"""Shared source recovery, containment records and complete-call adapters."""
from collections import Counter
from pathlib import Path
import hashlib
import re
from ..frontend import GEN, parse_function, preprocess, Unsupported
from ..scope import CFG

HEADER = '#include <riscv_vector.h>\n#include <stddef.h>\n#include <stdint.h>\n#include <math.h>\n#include <float.h>\n#include <string.h>\n'

def load_source(path):
    raw=Path(path).read_text(encoding='utf-8-sig')
    ast=parse_function(preprocess(raw))
    if ast.decl.name!='source_kernel':raise Unsupported('expected source_kernel entrypoint')
    notice=re.match(r'\s*(/\*.*?\*/)',raw,re.S)
    return ast,(notice[1]+'\n' if notice else '')

def containment(scope,changed):
    protected=Counter(n['text'] for n in scope['nodes'] if n['id'] not in scope['selected'] and n['text'])
    remaining=Counter(n['text'] for n in CFG(changed).nodes if n['text'])
    missing=list((protected-remaining).elements())
    if missing:raise Unsupported('outside-region statements changed: '+repr(missing))
    return dict(protected_statements=sum(protected.values()),protected_multiset_preserved=True,
                limitation='statement multiset containment is not semantic equivalence proof')

def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def level3_wrapper(variant,lmul):
    """Pack with the vector configuration used by the recovered microkernel."""
    call='source_kernel(M,N,K,1.0f,PA,PB,C,M'+(',0' if variant=='trmm' else '')+');'
    return f'''extern "C" __attribute__((noinline)) void kernel(int M,int N,int K,const float*A,const float*B,float*C,float*W){{
float*PA=W;float*PB=W+(size_t)M*K;size_t z=0;
for(int i=0;i<M;){{size_t vl=__riscv_vsetvl_e32m{lmul}(M-i);for(int k=0;k<K;k++)for(size_t r=0;r<vl;r++)PA[z++]=A[(i+r)*K+k];i+=vl;}}
z=0;for(int j=0;j<N;){{int n=N-j>=8?8:(N-j>=4?4:(N-j>=2?2:1));for(int k=0;k<K;k++)for(int q=0;q<n;q++)PB[z++]=B[k*N+j+q];j+=n;}}
for(size_t t=0;t<(size_t)M*N;t++)C[t]=0.0f;{call}
}}
'''
