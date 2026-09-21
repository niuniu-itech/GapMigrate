"""Level 2 legalization and checked chunk-reduction candidates.

Ported from generate_level2 and generate_level2_hierarchy. Conservative scope
expansion includes vsetvlmax definitions when vector representation changes.
"""
import re
from ..frontend import GEN,parse_function,retarget_group,Unsupported
from ..scope import analyze
from ..transforms import legalize_reductions,reduction_helpers
from .common import containment

def legalize(ast,lm,profile):
    node=retarget_group(ast,8,lm)
    node,events=legalize_reductions(node,'native' if profile['reductions'] else 'scalar',lm)
    code=GEN.visit(node)
    helpers='' if profile['reductions'] else reduction_helpers(lm)
    if not profile['complex_memory']:
        for op in ['vlse32','vsse32']:
            old=f'__riscv_{op}_v_f32m{lm}'
            if old not in code:continue
            name=f'gm_{op}_{lm}';code=code.replace(old,name)
            if op=='vlse32':
                helpers+=f'\nstatic inline vfloat32m{lm}_t {name}(const float*p,long stride,size_t vl){{float a[vl];for(size_t i=0;i<vl;i++)a[i]=*(const float*)((const char*)p+i*stride);return __riscv_vle32_v_f32m{lm}(a,vl);}}\n'
            else:
                helpers+=f'\nstatic inline void {name}(float*p,long stride,vfloat32m{lm}_t v,size_t vl){{float a[vl];__riscv_vse32_v_f32m{lm}(a,v,vl);for(size_t i=0;i<vl;i++)*(float*)((char*)p+i*stride)=a[i];}}\n'
    return code,helpers,events

def chunk_reduction(code,lm):
    code,ninit=re.subn(r'vr = __riscv_vfmv_v_f_f32m\d\(0, (vlmax|vl_max)\);','gm_sum = 0.0f;',code)
    code,nupdate=re.subn(rf'vr = __riscv_vfmacc_vv_f32m{lm}_tu\(vr, ([a-z]+), ([a-z]+), vl\);',r'gm_sum += gm_chunk_dot(\1, \2, vl);',code)
    code,nreduce=re.subn(rf'v_res = (?:__riscv_vfredusum_vs_f32m{lm}_f32m1|gm_reduce_sum_{lm})\(vr, v_z0, (vlmax|vl_max)\);','v_res = __riscv_vfmv_s_f_f32m1(gm_sum, 1);',code)
    if not(ninit==nupdate==nreduce and ninit>0):raise Unsupported('chunk reduction schema mismatch')
    code=code.replace('{','{\nfloat gm_sum = 0.0f;',1)
    helper=f'''static inline float gm_chunk_dot(vfloat32m{lm}_t a,vfloat32m{lm}_t b,size_t vl){{
float lanes[vl];auto product=__riscv_vfmul_vv_f32m{lm}(a,b,vl);__riscv_vse32_v_f32m{lm}(lanes,product,vl);float s=0.0f;for(size_t i=0;i<vl;i++)s+=lanes[i];return s;}}
'''
    return code,helper

def construct(ast,variant,profile):
    initial=analyze(ast,profile);accepted=[];rejected=[]
    lms=[8] if not initial['seeds'] else ([1] if profile['lmul']==1 else [1,2,4,8])
    for lm in lms:
        extra=[n['id'] for n in initial['nodes'] if '__riscv_vsetvlmax_e32m8' in n['calls']] if initial['seeds'] and lm!=8 else []
        scope=analyze(ast,profile,additional_seeds=extra)
        base,helpers,events=legalize(ast,lm,profile)
        for organization in ['retain']+(['chunk_reduce'] if variant!='gemv_n' and scope['seeds'] else []):
            name=f'{variant}_{profile["id"]}_m{lm}_{organization}'
            try:
                code=base;helper=''
                if organization=='chunk_reduce':code,helper=chunk_reduction(code,lm)
                cert=containment(scope,parse_function(code))
                cert.update(seed_nodes=scope['seeds'],initial_scope_statements=initial['selected_statements'],
                    configuration_expansion_nodes=extra,selected_statements=scope['selected_statements'],
                    total_statements=scope['total_statements'],lowering=events,
                    numerical_policy='FP64 reference abs2e-5 rel2e-4; chunk reduction changes FP association')
                accepted.append(dict(id=name,variant=variant,profile=profile['id'],lmul=lm,
                    organization=organization,code='#include <stddef.h>\n#include <riscv_vector.h>\n'+helpers+helper+code,
                    certificate=cert))
            except (Unsupported,AssertionError) as exc:
                rejected.append(dict(id=name,reason=str(exc)))
    return accepted,rejected,initial
