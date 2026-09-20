"""Bounded C AST recovery for selected upstream RVV intrinsic entrypoints.

The AST is the executable source record. Recovery never selects an operator
template by its entrypoint name. Contracts specialize unpacked/nonaliasing FP32
entries; every source operation and accepted specialization is retained.
"""
from pathlib import Path
import copy, hashlib, io, json, re, sys
from pcpp import Preprocessor
from pycparser import c_parser, c_ast as C, c_generator
GEN=c_generator.CGenerator()

class Unsupported(ValueError): pass

def function_fragment(text, entry):
    m=re.search(r'\b(?:static\s+)?(?:void|int)\s+'+re.escape(entry)+r'\s*\(',text)
    if not m:raise Unsupported('entrypoint absent: '+entry)
    start=m.start(); brace=text.index('{',m.end()); depth=1; pos=brace+1
    # Comments in accepted upstream fragments do not contain unbalanced braces.
    while depth:
        if text[pos]=='{':depth+=1
        elif text[pos]=='}':depth-=1
        pos+=1
    return text[start:pos],text.count('\n',0,start)+1

def preprocess(text, name='source.c'):
    pp=Preprocessor();pp.line_directive=None
    for definition in ('__riscv_vector 1','__riscv_xtheadvector 0','FLOAT float','IFLOAT float','BLASLONG long','CNAME source_kernel'):
        pp.define(definition)
    pp.parse(re.sub(r'^\s*#include[^\n]*','',text,flags=re.M),source=name)
    out=io.StringIO();pp.write(out)
    if pp.return_code:raise Unsupported('preprocessor rejected source')
    return out.getvalue()

def parse_function(text):
    types='typedef unsigned long size_t;\n'
    types+='\n'.join(f'typedef int vfloat32m{x}_t;' for x in (1,2,4,8))+'\n'
    tree=c_parser.CParser().parse(types+text)
    funcs=[n for n in tree.ext if isinstance(n,C.FuncDef)]
    if len(funcs)!=1:raise Unsupported('exactly one selected function required')
    return funcs[0]

def specialize(node, constants):
    """Only contract-bound control flow is removed; data-dependent tails remain."""
    def known(x):
        if isinstance(x,C.ID):return constants.get(x.name)
        if isinstance(x,C.Constant):
            try:return int(x.value)
            except ValueError:return None
        if isinstance(x,C.BinaryOp):
            a,b=known(x.left),known(x.right)
            if a is None or b is None:return None
            if x.op=='==':return int(a==b)
            if x.op=='&&':return int(bool(a) and bool(b))
        return None
    if isinstance(node,C.If):
        v=known(node.cond)
        if v is not None:
            chosen=node.iftrue if v else node.iffalse
            return specialize(chosen,constants) if chosen else C.EmptyStatement()
    for key in node.__slots__:
        if key in ('coord','__weakref__'):continue
        v=getattr(node,key,None)
        if isinstance(v,C.Node):setattr(node,key,specialize(v,constants))
        elif isinstance(v,list):setattr(node,key,[specialize(q,constants) if isinstance(q,C.Node) else q for q in v])
    return node

def describe(ast, source, entry, offset, guards):
    records=[];writes={}; edges=[];loops=[]; calls=[]
    class V(C.NodeVisitor):
        def visit_For(self,n):loops.append(dict(kind='for',condition=GEN.visit(n.cond),line=n.coord.line));self.generic_visit(n)
        def visit_While(self,n):loops.append(dict(kind='while',condition=GEN.visit(n.cond),line=n.coord.line));self.generic_visit(n)
        def visit_FuncCall(self,n):
            if not isinstance(n.name,C.ID):raise Unsupported('indirect function call')
            name=n.name.name
            if not (name.startswith('__riscv_') or name in ('exp_ps','csrr_vlenb')):raise Unsupported('unmodeled external call '+name)
            calls.append(dict(name=name,args=[GEN.visit(x) for x in n.args.exprs] if n.args else [],line=n.coord.line))
            self.generic_visit(n)
        def record(self,lhs,rhs,line):
            if rhs is None:return
            ids=[]
            class IDs(C.NodeVisitor):
                def visit_ID(self,n):ids.append(n.name)
                def visit_FuncCall(self,n):
                    if n.args:self.visit(n.args)
            IDs().visit(rhs)
            rid=len(records)
            records.append(dict(id=rid,lhs=lhs,rhs=GEN.visit(rhs),line=line,reads=sorted(set(ids))))
            for ident in set(ids):
                if ident in writes:edges.append(dict(producer=writes[ident],consumer=rid,value=ident))
            writes[lhs]=rid
        def visit_Decl(self,n):self.record(n.name,n.init,n.coord.line);self.generic_visit(n)
        def visit_Assignment(self,n):self.record(GEN.visit(n.lvalue),n.rvalue,n.coord.line);self.generic_visit(n)
    V().visit(ast)
    # Edges document local source def/use only. They are not a whole-program SSA
    # proof across loops; transformations additionally check their own scopes.
    return dict(source_sha256=hashlib.sha256(source).hexdigest(),entry=entry,original_entry_start_line=offset,guards=guards,operations=calls,statements=records,local_def_use=edges,loops=loops,relationship_scope='local expressions and explicitly checked phase/loop transforms',specialized_source=GEN.visit(ast))

def recover_ncnn(path,entry):
    raw=Path(path).read_bytes();frag,line=function_fragment(raw.decode(),entry)
    ast=parse_function(preprocess(frag))
    constants={'elempack':1,'packn':8,'gamma_ptr':1,'beta_ptr':1}
    specialize(ast,constants)
    # Substitute only the contract parameter, not pointer values.
    class Replace(C.NodeVisitor):
        def visit_Decl(self,n):
            if n.name=='packn':n.init=C.Constant('int','8')
            self.generic_visit(n)
    Replace().visit(ast)
    guards=['RVV VLEN=256 bits','FP32','elempack=1','positive row length','nonaliasing input/affine buffers','gamma and beta nonnull when present']
    return ast,describe(ast,raw,entry,line,guards)

def recover_openblas(path):
    raw=Path(path).read_bytes();ast=parse_function(preprocess(raw.decode()))
    return ast,describe(ast,raw,'CNAME',1,['FP32','TRMMKERNEL disabled','positive dimensions','declared packed-layout contract','nonaliasing buffers'])

def retarget_group(ast, old, new):
    result=copy.deepcopy(ast)
    def rename(value):
        return re.sub(r'(?<=[0-9])m'+str(old)+r'(?=\b|_)','m'+str(new),value)
    def transform(node):
        if isinstance(node,C.ID):node.name=rename(node.name)
        if isinstance(node,C.IdentifierType):node.names=[rename(n) for n in node.names]
        for key in node.__slots__:
            if key in ('coord','__weakref__'):continue
            v=getattr(node,key,None)
            if isinstance(v,C.Node):setattr(node,key,transform(v))
            elif isinstance(v,list):setattr(node,key,[transform(x) if isinstance(x,C.Node) else x for x in v])
        if isinstance(node,C.FuncCall) and isinstance(node.name,C.ID):
            # A one-register insert at offset zero, immediately gathered from
            # lane zero, becomes the inserted one-register value.
            if node.name.name=='__riscv_vset_v_f32m1_f32m1':
                assert GEN.visit(node.args.exprs[1])=='0'
                return node.args.exprs[2]
        return node
    return transform(result)

def rename_function(ast,name):
    ast.decl.name=name;ast.decl.type.type.declname=name
    return ast
