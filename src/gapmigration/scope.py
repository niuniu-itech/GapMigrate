"""Conservative structured-C CFG and dependence-closed migration regions.

Least closure is relative to this graph, not a proof of globally minimal edits.
Names are conservatively merged across lexical scopes. Pointer alias groups are
conservative unless disjoint roots are explicitly part of the caller contract.
"""
import re
from collections import defaultdict
from pycparser import c_ast as C,c_generator
G=c_generator.CGenerator()

class UnsupportedControl(ValueError):pass

def facts(node):
    defs=set();uses=set();calls=[];types=set();memory=[]
    def read(n):
        if n is None:return
        if isinstance(n,C.ID):uses.add(n.name);return
        if isinstance(n,C.FuncCall):
            if not isinstance(n.name,C.ID):raise UnsupportedControl('indirect call')
            name=n.name.name;calls.append(name)
            args=n.args.exprs if n.args else []
            memop=re.match(r'^(?:__riscv_)?v([ls])(?:e\d|se\d|[uo]xei\d|seg\d|[uo]xseg\d|sseg\d)',name)
            if memop:
                root=G.visit(args[0]) if args else '?'
                ids=re.findall(r'[A-Za-z_]\w*',root);root=ids[0] if ids else '?'
                mode='write' if memop.group(1)=='s' else 'read'
                memory.append((root,mode))
            elif not name.startswith('__riscv_'):
                memory.append(('unknown-call:*','write'))
            for a in args:read(a)
            return
        if isinstance(n,C.Decl):
            defs.add(n.name)
            ty=G.visit(n.type)
            if 'vfloat' in ty or 'vint' in ty or 'vuint' in ty or 'vbool' in ty:types.add(n.name)
            read(n.init);return
        if isinstance(n,C.Assignment):
            if isinstance(n.lvalue,C.ID):
                defs.add(n.lvalue.name)
                if n.op!='=':uses.add(n.lvalue.name)
            else:read(n.lvalue);memory.append(('?','write'))
            read(n.rvalue);return
        if isinstance(n,C.UnaryOp) and n.op in ('p++','p--','++','--'):
            if isinstance(n.expr,C.ID):defs.add(n.expr.name)
            read(n.expr);return
        if isinstance(n,C.ArrayRef) or (isinstance(n,C.UnaryOp) and n.op=='*'):memory.append(('?','read'))
        for _,child in n.children():read(child)
    read(node)
    return dict(defs=sorted(defs),uses=sorted(uses),calls=calls,vector_defs=sorted(types),memory=memory)

class CFG:
    def __init__(self,ast):
        self.nodes=[];self.edges=set();self.loops={};self.loop_serial=0
        entry=self.add(None,'entry',[])
        exits=self.block(ast.body,{entry},[],None,None)
        self.exit=self.add(None,'exit',[])
        for q in exits:self.edges.add((q,self.exit))
        for q in self.nodes:
            if q['kind']=='return':self.edges.add((q['id'],self.exit))
    def add(self,node,kind,loops):
        i=len(self.nodes);f=facts(node) if node else dict(defs=[],uses=[],calls=[],vector_defs=[],memory=[])
        self.nodes.append(dict(id=i,kind=kind,line=getattr(getattr(node,'coord',None),'line',None),text=G.visit(node) if node else '',loops=list(loops),**f))
        for loop in loops:self.loops[loop].add(i)
        return i
    def connect(self,preds,node,kind,loops):
        i=self.add(node,kind,loops)
        self.edges.update((p,i) for p in preds);return i
    def block(self,n,preds,loops,break_to,continue_to):
        if n is None:return preds
        if isinstance(n,C.Compound):
            for child in n.block_items or []:preds=self.block(child,preds,loops,break_to,continue_to)
            return preds
        if isinstance(n,C.If):
            h=self.connect(preds,n.cond,'branch',loops)
            a=self.block(n.iftrue,{h},loops,break_to,continue_to)
            b=self.block(n.iffalse,{h},loops,break_to,continue_to) if n.iffalse else {h}
            return a|b
        if isinstance(n,(C.For,C.While)):
            k=self.loop_serial;self.loop_serial+=1;self.loops[k]=set();ls=loops+[k]
            if isinstance(n,C.For) and n.init is not None:preds=self.block(n.init,preds,ls,break_to,continue_to)
            h=self.connect(preds,n.cond,'loop_condition',ls)
            end=self.add(None,'loop_exit',loops)
            step=self.add(n.next,'loop_step',ls) if isinstance(n,C.For) and n.next is not None else h
            exits=self.block(n.stmt,{h},ls,end,step)
            self.edges.update((q,step) for q in exits)
            if step!=h:self.edges.add((step,h))
            self.edges.add((h,end));return {end}
        if isinstance(n,C.DoWhile):
            k=self.loop_serial;self.loop_serial+=1;self.loops[k]=set();ls=loops+[k]
            entry=self.connect(preds,None,'do_entry',ls)
            cond=self.add(n.cond,'loop_condition',ls)
            end=self.add(None,'loop_exit',loops)
            exits=self.block(n.stmt,{entry},ls,end,cond)
            self.edges.update((q,cond) for q in exits)
            self.edges.update([(cond,entry),(cond,end)])
            return {end}
        if isinstance(n,C.Break):
            if break_to is None:raise UnsupportedControl('break outside loop')
            i=self.connect(preds,n,'break',loops);self.edges.add((i,break_to));return set()
        if isinstance(n,C.Continue):
            if continue_to is None:raise UnsupportedControl('continue outside loop')
            i=self.connect(preds,n,'continue',loops);self.edges.add((i,continue_to));return set()
        if isinstance(n,(C.Goto,C.Label,C.Switch)):raise UnsupportedControl(type(n).__name__)
        i=self.connect(preds,n,'return' if isinstance(n,C.Return) else 'statement',loops)
        return set() if isinstance(n,C.Return) else {i}
    def dependencies(self):
        pred=defaultdict(set)
        for a,b in self.edges:pred[b].add(a)
        ins=[set() for _ in self.nodes];outs=[set() for _ in self.nodes]
        changed=True;iterations=0
        while changed:
            changed=False;iterations+=1
            for n in self.nodes:
                i=n['id'];incoming=set().union(*(outs[p] for p in pred[i])) if pred[i] else set()
                outgoing={(v,d) for v,d in incoming if v not in n['defs']}|{(v,i) for v in n['defs']}
                if ins[i]!=incoming or outs[i]!=outgoing:changed=True;ins[i]=incoming;outs[i]=outgoing
        deps=set()
        for n in self.nodes:
            for v,d in ins[n['id']]:
                if v in n['uses']:deps.add((d,n['id'],v,'RAW'))
                if v in n['defs']:deps.add((d,n['id'],v,'WAW'))
        # Flow-insensitive memory edges are explicit conservative barriers.
        mem=[n for n in self.nodes if n['memory']]
        for i,a in enumerate(mem):
            for b in mem[i+1:]:
                if any(ma=='write' or mb=='write' for _,ma in a['memory'] for _,mb in b['memory']):
                    deps.add((a['id'],b['id'],'memory:*','MEM'))
        return deps,iterations

def unsupported(call,profile):
    reasons=[]
    if profile.get('lmul')==1 and re.search(r'm(?:2|4|8)(?:_|$)',call):reasons.append('LMUL limit')
    if not profile.get('reductions',True) and re.search(r'v(?:f|w|fw)?(?:u|o)?red',call):reasons.append('vector reduction')
    if not profile.get('complex_memory',True) and re.search(r'v(?:ls|ss)e|v[ls](?:u|o)xei|v[ls](?:s|ux|ox)?seg',call):reasons.append('complex vector memory')
    return reasons

def analyze(ast,profile,mode='closed',additional_seeds=None):
    cfg=CFG(ast);deps,it=cfg.dependencies()
    seeds={n['id'] for n in cfg.nodes if any(unsupported(c,profile) for c in n['calls'])}
    vectors=set().union(*(set(n['vector_defs']) for n in cfg.nodes))
    selected=set(seeds);reasons={i:['unsupported intrinsic'] for i in seeds};rounds=0
    for i in additional_seeds or []:
        if not 0<=i<len(cfg.nodes):raise ValueError('invalid expansion node')
        selected.add(i);reasons.setdefault(i,[]).append('rule-required configuration expansion')
    if mode=='whole':selected={n['id'] for n in cfg.nodes if n['text']}
    elif mode not in ('closed','loop_only','seed_only'):raise ValueError(mode)
    if mode in ('closed','loop_only'):
        while True:
            rounds+=1;before=set(selected)
            for i in list(selected):
                if cfg.nodes[i]['loops']:
                    for j in cfg.loops[cfg.nodes[i]['loops'][-1]]:
                        if j not in selected:reasons.setdefault(j,[]).append('complete innermost loop')
                        selected.add(j)
            if mode=='closed':
                for a,b,v,k in deps:
                    if v in vectors or k=='MEM':
                        if a in selected or b in selected:
                            for j in (a,b):
                                if j not in selected:reasons.setdefault(j,[]).append(k+':'+v)
                                selected.add(j)
            if before==selected:break
    crossing=[dict(producer=a,consumer=b,value=v,kind=k) for a,b,v,k in deps if (a in selected)!=(b in selected)]
    leaks=[d for d in crossing if d['value'] in vectors or d['kind']=='MEM']
    return dict(status='analyzed',mode=mode,additional_seeds=sorted(additional_seeds or []),nodes=cfg.nodes,cfg_edges=sorted(cfg.edges),dependence_edges=[dict(producer=a,consumer=b,value=v,kind=k) for a,b,v,k in sorted(deps)],seeds=sorted(seeds),selected=sorted(selected),selection_reasons=reasons,closure_rounds=rounds,reaching_definition_rounds=it,unsafe_crossings=leaks,boundary_values=crossing,minimality='least fixed point of conservative vector/memory dependency and innermost-loop closure; not global edit minimality',limitations=['lexical names conservatively merged','memory aliases conservatively merged','unsupported unstructured control rejected'],selected_statements=sum(bool(cfg.nodes[i]['text']) for i in selected),total_statements=sum(bool(n['text']) for n in cfg.nodes))
