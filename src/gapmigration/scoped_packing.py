"""Region-scoped packing reconstruction from recovered OpenBLAS AST.

The affine packing contract requires an outer panel loop boundary. That rule
expansion is explicit, rather than calling the raw dependence slice minimal.
"""
from pathlib import Path
import copy,hashlib,json,sys
from pycparser import c_ast as C
from .frontend import recover_openblas,GEN,parse_function
from .scope import analyze

def generate(source,profile,lmul,mode):
 if lmul not in (1,2,4,8):raise ValueError('unsupported LMUL')
 ast,ir=recover_openblas(source);scope=analyze(ast,profile)
 body=ast.body.block_items
 loops=[(i,n) for i,n in enumerate(body) if isinstance(n,C.For) and GEN.visit(n.init)=='j = n']
 if len(loops)!=1:raise ValueError('panel loop not uniquely established')
 ix,loop=loops[0]
 required=['a_offset1, lda * (sizeof(float)), vl','a_offset += vl * lda','a_offset1++','b_offset += vl']
 if any(t not in GEN.visit(loop) for t in required):raise ValueError('packing relation guard')
 if mode not in ('gather','tile'):raise ValueError(mode)
 if profile.get('lmul')==1 and lmul!=1:raise ValueError('LMUL gate')
 if mode=='gather':
  replacement=f'''for(j=n;j>0;j-=vl){{
   vl=j<16?j:16; a_offset1=a_offset; a_offset+=vl*lda;
   for(i=m;i>0;i--){{
    for(size_t q=0;q<vl;){{size_t avl=vl-q; if(avl>{8*lmul})avl={8*lmul};size_t z=__riscv_vsetvl_e32m{lmul}(avl);float lane[16];
     for(size_t r=0;r<z;r++)lane[r]=a_offset1[(q+r)*lda];
     vfloat32m{lmul}_t v=__riscv_vle32_v_f32m{lmul}(lane,z);
     __riscv_vse32_v_f32m{lmul}(b_offset+q,v,z);q+=z;}}
    a_offset1++; b_offset+=vl;
   }}
  }}'''
 else:
  replacement=f'''for(j=n;j>0;j-=vl){{vl=j<16?j:16;a_offset1=a_offset;a_offset+=vl*lda;
   for(long i0=0;i0<m;i0+=32){{long h=m-i0<32?m-i0:32;float tile[512];float lane[64];
    for(size_t q=0;q<vl;q++)for(long t=0;t<h;){{size_t z=__riscv_vsetvl_e32m{lmul}(h-t);
     vfloat32m{lmul}_t v=__riscv_vle32_v_f32m{lmul}(a_offset1+i0+t+q*lda,z);
     __riscv_vse32_v_f32m{lmul}(lane,v,z);for(size_t r=0;r<z;r++)tile[(t+r)*vl+q]=lane[r];t+=z;}}
    for(size_t t=0;t<h*vl;){{size_t z=__riscv_vsetvl_e32m{lmul}(h*vl-t);vfloat32m{lmul}_t v=__riscv_vle32_v_f32m{lmul}(tile+t,z);__riscv_vse32_v_f32m{lmul}(b_offset+i0*vl+t,v,z);t+=z;}}
   }}b_offset+=m*vl;
  }}'''
 fresh=parse_function('void replacement(void){'+replacement+'}').body.block_items[0]
 original=[GEN.visit(n) for n in body];body[ix]=fresh
 updated=[GEN.visit(n) for n in body]
 preserved=[i for i in range(len(body)) if i!=ix]
 assert all(original[i]==updated[i] for i in preserved)
 # The source function signature, outer declarations, initialization and return
 # survive. Only the checked panel loop is replaced.
 cert=dict(family='packing',mode=mode,lmul=lmul,source_sha256=ir['source_sha256'],seed_nodes=scope['seeds'],initial_scope_nodes=scope['selected'],rule_scope='outer panel loop',scope_expansion_reason='16-column output panel contract and traversal realization',replaced_top_level_statement=ix,preserved_top_level_statements=len(preserved),preserved_source_bytes=sum(len(original[i].encode()) for i in preserved),original_source_bytes=len(GEN.visit(recover_openblas(source)[0]).encode()),outside_region_ast_identical=True,scope_claim='single rule-admissible panel region, not global minimum',fixed_tile_height=32)
 code='#include <riscv_vector.h>\n#include <stddef.h>\n'+GEN.visit(ast)+'\n'
 code+='extern "C" void kernel(int M,int N,int K,const float*A,const float*B,float*C,float*W){source_kernel(M,N,(float*)A,M,C);}\n'
 cert['generated_sha256']=hashlib.sha256(code.encode()).hexdigest()
 return code,cert
