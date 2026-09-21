"""GEMM/TRMM regrouping, m2 legalization and matching packing adapters.

Ported from the scoped GEMM and TRMM campaign constructors. Output domains,
reduction order and triangular bounds stay in the recovered source AST.
"""
import copy
from ..frontend import GEN,retarget_group,Unsupported
from ..scope import analyze
from ..transforms import split_consumers
from .common import containment,HEADER,level3_wrapper

def construct(ast,variant,profile):
    scope=analyze(ast,profile);accepted=[];rejected=[]
    for group in ([8,4,2] if scope['seeds'] else [8]):
        lm=1 if profile['lmul']==1 else 2
        name=(f'{profile["id"]}_gemm_L{lm}_group{group}' if variant=='gemm'
              else f'trmm_RU_{profile["id"]}_m{lm}_g{group}')
        try:
            node=copy.deepcopy(ast);events=[]
            if group!=8:
                node,events=split_consumers(node,group)
                if not events:raise Unsupported('eight-consumer regrouping schema absent')
            node=retarget_group(node,2,lm)
            cert=containment(scope,node)
            cert.update(seed_nodes=scope['seeds'],selected_statements=scope['selected_statements'],
                        total_statements=scope['total_statements'],events=events,
                        scope_model=scope['minimality'])
            accepted.append(dict(id=name,variant=variant,profile=profile['id'],lmul=lm,
                organization='retain' if group==8 else 'regroup_'+str(group),
                code=HEADER+GEN.visit(node)+'\n'+level3_wrapper(variant,lm),certificate=cert))
        except (Unsupported,AssertionError) as exc:
            rejected.append(dict(id=name,reason=str(exc)))
    return accepted,rejected,scope
