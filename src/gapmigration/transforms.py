"""Checked transformations over recovered source ASTs, not operator templates."""
import copy,re
from pycparser import c_ast as C
from .frontend import GEN, Unsupported, parse_function

def rewrite(node, fn):
    for key in node.__slots__:
        if key in ('coord','__weakref__'):continue
        val=getattr(node,key,None)
        if isinstance(val,C.Node):setattr(node,key,rewrite(val,fn))
        elif isinstance(val,list):setattr(node,key,[rewrite(x,fn) if isinstance(x,C.Node) else x for x in val])
    return fn(node)

def rematerialize_phase(ast):
    """Remove an in-place elementwise producer store and rebuild its later load.

    Accepted scope: separate row phases, identical local ptr alias, no producer
    side effects except its store and reduction, lane-wise exp/sub expression.
    """
    ast=copy.deepcopy(ast); blocks=ast.body.block_items
    producers=[]
    for bi,block in enumerate(blocks):
        if not isinstance(block,C.Compound):continue
        for loop in block.block_items or []:
            if not isinstance(loop,C.While):continue
            env={}; expr=None; store=None
            def expand(n):
                n=copy.deepcopy(n)
                def sub(x):return copy.deepcopy(env[x.name]) if isinstance(x,C.ID) and x.name in env else x
                return rewrite(n,sub)
            for st in loop.stmt.block_items:
                if isinstance(st,C.Decl) and st.init is not None:env[st.name]=expand(st.init)
                if isinstance(st,C.Assignment) and isinstance(st.lvalue,C.ID):env[st.lvalue.name]=expand(st.rvalue)
                if isinstance(st,C.FuncCall) and st.name.name.startswith('__riscv_vse32_'):
                    expr=expand(st.args.exprs[1]);store=st
            if expr is not None and 'exp_ps(' in GEN.visit(expr):
                text=GEN.visit(expr)
                if GEN.visit(store.args.exprs[0])!='ptr':raise Unsupported('materialization pointer alias')
                if not ('__riscv_vfsub_' in text and '__riscv_vle32_' in text):raise Unsupported('unsupported rematerialization expression')
                ids=set()
                class IDs(C.NodeVisitor):
                    def visit_ID(self,n):ids.add(n.name)
                    def visit_FuncCall(self,n):self.visit(n.args)
                IDs().visit(expr)
                if ids-{'ptr','_max','vl'}:raise Unsupported('unavailable rematerialization operands')
                producers.append((bi,expr))
    if len(producers)!=1:raise Unsupported('one checked materialized producer phase required')
    bi,expr=producers[0]; consumer_index=None; replaced=[0]; removed=[0]
    # Contract-bound specialization may introduce compounds between phases.
    for ci in range(bi+1,len(blocks)):
        block=blocks[ci]
        if isinstance(block,C.Compound) and 'while (' in GEN.visit(block) and '__riscv_vle32_' in GEN.visit(block):
            if consumer_index is not None:raise Unsupported('multiple later memory phases')
            consumer_index=ci
    if consumer_index is None:raise Unsupported('materialized value has no checked consumer')
    middle='\n'.join(GEN.visit(x) for x in blocks[bi+1:consumer_index])
    if re.search(r'\b_max\s*=',middle):raise Unsupported('producer invariant overwritten')
    def remove(x):
        if isinstance(x,C.FuncCall) and x.name.name.startswith('__riscv_vse32_'):
            if GEN.visit(x.args.exprs[0])!='ptr':raise Unsupported('producer writes another region')
            removed[0]+=1;return C.EmptyStatement()
        return x
    def substitute(x):
        if isinstance(x,C.FuncCall) and x.name.name.startswith('__riscv_vle32_'):
            if [GEN.visit(a) for a in x.args.exprs]!=['ptr','vl']:raise Unsupported('consumer address does not match')
            replaced[0]+=1;return copy.deepcopy(expr)
        return x
    blocks[bi]=rewrite(blocks[bi],remove);blocks[consumer_index]=rewrite(blocks[consumer_index],substitute)
    if removed[0]!=2 or replaced[0]!=1:raise Unsupported('full/tail producer and single consumer required')
    return ast,dict(kind='rematerialize_checked_row_value',producer_phase=bi,consumer_phase=consumer_index,removed_stores=removed[0],substituted_loads=replaced[0],expression=GEN.visit(expr))

def legalize_reductions(ast, mode, lm):
    ast=copy.deepcopy(ast);operations=[]
    def change(n):
        if isinstance(n,C.FuncCall) and re.match(r'__riscv_vfred(?:u?o?sum|max)_vs_',n.name.name):
            old=n.name.name;kind='max' if 'max' in old else 'sum'
            if mode=='scalar':n.name.name=f'gm_reduce_{kind}_{lm}'
            operations.append(dict(original=old,replacement=n.name.name,line=n.coord.line))
        return n
    return rewrite(ast,change),operations

def reduction_helpers(lm):
    return f'''
static inline vfloat32m1_t gm_reduce_sum_{lm}(vfloat32m{lm}_t x,vfloat32m1_t seed,size_t vl){{
 float lanes[64];__riscv_vse32_v_f32m{lm}(lanes,x,vl);
 float s=__riscv_vfmv_f_s_f32m1_f32(seed);
 for(size_t i=0;i<vl;i++)s+=lanes[i];
 return __riscv_vfmv_s_f_f32m1(s,__riscv_vsetvlmax_e32m1());
}}
static inline vfloat32m1_t gm_reduce_max_{lm}(vfloat32m{lm}_t x,vfloat32m1_t seed,size_t vl){{
 float lanes[64];__riscv_vse32_v_f32m{lm}(lanes,x,vl);
 float s=__riscv_vfmv_f_s_f32m1_f32(seed);
 for(size_t i=0;i<vl;i++)s=s>lanes[i]?s:lanes[i];
 return __riscv_vfmv_s_f_f32m1(s,__riscv_vsetvlmax_e32m1());
}}
'''

def split_consumers(ast, width):
    """Split witnessed packed GEMM consumer groups into serial reduction passes.

    Existing arithmetic and address expressions are retained. Each pass resets
    the shared producer pointer, keeps a subset of output accumulators, and
    completes the original reduction before advancing to the next lane tile.
    """
    ast=copy.deepcopy(ast);events=[]
    def transform(n):
        if not isinstance(n,C.For) or GEN.visit(n.next)!='i -= vl':return n
        stores=[]
        class S(C.NodeVisitor):
            def visit_FuncCall(self,x):
                if x.name.name.startswith('__riscv_vse32_'):
                    args=[GEN.visit(v) for v in x.args.exprs]
                    if re.fullmatch(r'C[0-7]',args[0]) and re.fullmatch(r'va[0-7]',args[1]):
                        acc='vres'+args[1][2:]
                        if not (re.search(r'\b'+args[1]+r'\s*=\s*__riscv_vfmacc_vf_\w+\('+args[1]+r', alpha, '+acc+r', vl\)',GEN.visit(n.stmt)) or re.search(r'\b'+args[1]+r'\s*=\s*__riscv_vfmul_vf_\w+\('+acc+r', alpha, vl\)',GEN.visit(n.stmt))):raise Unsupported('output recurrence not established')
                        stores.append((args[0],acc))
                self.generic_visit(x)
        S().visit(n.stmt)
        if len(stores)<=width:return n
        if len(set(stores))!=len(stores):raise Unsupported('duplicate output ownership')
        body_text=GEN.visit(n.stmt)
        if 'ptrbb = bb;' not in body_text or 'ptrba += vl;' not in body_text:raise Unsupported('producer stream/reset not verified')
        groups=[]
        for start in range(0,len(stores),width):
            kept=stores[start:start+width];drop={v for pair in stores if pair not in kept for v in pair}
            def filt(x):
                if isinstance(x,(C.Assignment,C.UnaryOp)) or (isinstance(x,C.FuncCall) and x.name.name.startswith('__riscv_vse32_')):
                    text=GEN.visit(x)
                    if any(re.search(r'\b'+re.escape(v)+r'\b',text) for v in drop):return C.EmptyStatement()
                return x
            group=rewrite(copy.deepcopy(n.stmt),filt)
            groups.append('ptrba = gm_saved_ba;\n'+GEN.visit(group))
        new='{float *gm_saved_ba=ptrba;\n'+'\n'.join(groups)+'\n}'
        n.stmt=parse_function('void temporary(void)'+new).body
        events.append(dict(kind='split_shared_producer_consumers',source_consumers=len(stores),target_consumers=width,reduction_passes=len(groups),outputs=stores))
        return n
    result=rewrite(ast,transform)
    return result,events

def separate_terminal_output(ast):
    """Retarget a terminal in-place store to a separate output buffer.

    The selected source has one store site in its last row phase; all earlier
    phases only read input. The final loop advances its pointer by active VL,
    so each output write is disjoint from every future input read.
    """
    ast=copy.deepcopy(ast);stores=[]
    class S(C.NodeVisitor):
        def visit_FuncCall(self,n):
            if n.name.name.startswith('__riscv_vse32_'):stores.append(n)
            self.generic_visit(n)
    S().visit(ast)
    if len(stores)!=1:raise Unsupported('output separation requires one terminal store site')
    if GEN.visit(stores[0].args.exprs[0])!='ptr':raise Unsupported('terminal store pointer')
    last=ast.body.block_items[-1]
    if not isinstance(last,C.Compound) or '__riscv_vse32_' not in GEN.visit(last):raise Unsupported('store not in terminal phase')
    if 'ptr += vl;' not in GEN.visit(last):raise Unsupported('terminal write ownership')
    extra=parse_function('void f(float *gm_output){}').decl.type.args.params[0]
    ast.decl.type.args.params.append(extra)
    prefix=parse_function('void f(void){float *gm_input_base=ptr;}').body.block_items[0]
    ast.body.block_items.insert(0,prefix)
    expr=C.BinaryOp('+',C.ID('gm_output'),C.BinaryOp('-',stores[0].args.exprs[0],C.ID('gm_input_base')))
    stores[0].args.exprs[0]=expr
    return ast,dict(kind='separate_terminal_output',source_store_sites=1,read_input_immutable=True,removed_wrapper_copy=True,output_address=GEN.visit(expr))


def stabilize_centered_normalization(ast):
    """Numerical-policy repair shared by fixed and reconstructed LN candidates.

    Preserve the recovered mean/centered-variance phases, but evaluate them on
    x - x[0]. Keep the centered value through normalization rather than forming
    a large, pre-rounded affine offset. This changes FP rounding, so it requires
    the declared tolerance contract and independent numerical qualification.
    """
    ast=copy.deepcopy(ast);s=GEN.visit(ast)
    checks=['_sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl)',
            '_p = __riscv_vfsub_vv_f32m8(_p, _mean, vl)',
            '_sqsum = __riscv_vfmadd_vv_f32m8(_p, _p, _sqsum, vl)',
            '_p = __riscv_vfmadd_vv_f32m8(_p, _gamma, _beta, vl)']
    if any(x not in s for x in checks):raise Unsupported('centered normalization arithmetic schema')
    anchor=parse_function('void f(void){const float gm_anchor=ptr[0];}').body.block_items[0]
    ast.body.block_items.insert(0,anchor);loads=[0];outputs=[0]
    def change(n):
        if isinstance(n,C.Decl) and n.name=='_p' and isinstance(n.init,C.FuncCall) and n.init.name.name=='__riscv_vle32_v_f32m8':
            args=n.init.args.exprs
            if GEN.visit(args[0]) not in ('ptr','ptr0') or GEN.visit(args[1]) not in ('vl','vlr'):
                raise Unsupported('normalization load ownership')
            n.init=C.FuncCall(C.ID('__riscv_vfsub_vf_f32m8'),C.ExprList([n.init,C.ID('gm_anchor'),copy.deepcopy(args[1])]))
            loads[0]+=1
        if isinstance(n,C.Assignment) and GEN.visit(n)=='_p = __riscv_vfmadd_vv_f32m8(_p, _a, _b, vl)':
            sub=C.FuncCall(C.ID('__riscv_vfsub_vv_f32m8'),C.ExprList([C.ID('_p'),C.ID('_mean'),C.ID('vl')]))
            n.rvalue=C.FuncCall(C.ID('__riscv_vfmul_vv_f32m8'),C.ExprList([sub,C.ID('_a'),C.ID('vl')]))
            outputs[0]+=1
        return n
    ast=rewrite(ast,change)
    if loads[0]!=5 or outputs[0]!=1:raise Unsupported('normalization phase coverage')
    # First-element outliers can leave large shifted means. Compensate the
    # FP32 lane sums without an extra input pass or a wider storage type.
    comp=parse_function('void f(void){vfloat32m8_t gm_comp=__riscv_vfmv_v_f_f32m8(0.f,__riscv_vsetvlmax_e32m8());}').body.block_items[0]
    pos=next(i for i,n in enumerate(ast.body.block_items) if isinstance(n,C.Decl) and n.name=='_sum')
    ast.body.block_items.insert(pos+1,comp);sum_sites=[0]
    def compensate(n):
        if not isinstance(n,C.Assignment):return n
        text=GEN.visit(n)
        if text=='_sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl)':
            sum_sites[0]+=1
            return parse_function('''void f(void){
vfloat32m8_t gm_y=__riscv_vfsub_vv_f32m8(_p,gm_comp,vl);
vfloat32m8_t gm_t=__riscv_vfadd_vv_f32m8(_sum,gm_y,vl);
gm_comp=__riscv_vfsub_vv_f32m8(__riscv_vfsub_vv_f32m8(gm_t,_sum,vl),gm_y,vl);
_sum=gm_t;
}''').body
        if text=='_sum = __riscv_vfadd_vv_f32m8_tu(_sum, _sum, _p, vlr)':
            sum_sites[0]+=1
            return parse_function('''void f(void){
vfloat32m8_t gm_y=__riscv_vfsub_vv_f32m8(_p,gm_comp,vlr);
_sum=__riscv_vfadd_vv_f32m8_tu(_sum,_sum,gm_y,vlr);
}''').body
        return n
    ast=rewrite(ast,compensate)
    if sum_sites[0]!=2:raise Unsupported('compensated full/tail mean coverage')
    return ast,dict(kind='shared_numeric_policy_shifted_compensated_centering',shifted_loads=loads[0],normalized_output_sites=outputs[0],compensated_mean_sites=sum_sites[0],anchor='row input[0]',precision='FP32',same_policy_for_both_methods=True)
