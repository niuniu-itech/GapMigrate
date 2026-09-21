"""Boundary copies around source-derived retained cores with explicit contracts."""
import copy
import re
from ..frontend import Unsupported

def construct(retained):
    if retained['organization']!='retain' or retained['profile'] not in ('P3','P4'):
        raise Unsupported('materialization requires a retained P3/P4 candidate')
    text=retained['code'];variant=retained['variant']
    if text.count('int source_kernel(')!=1 or len(re.findall(r'\bbuffer\b',text))!=1:
        raise Unsupported('source workspace is already used or entrypoint is ambiguous')
    text=text.replace('int source_kernel(', 'static inline __attribute__((always_inline)) int gm_retained_core(',1)
    nx='m' if variant!='gemv_n' else 'n';ny='n' if variant=='gemv_t' else 'm'
    if variant.startswith('gemv'):
        sig='long m,long n,long dummy,float alpha,float*a,long lda,float*x,long inc_x,float*y,long inc_y,float*buffer'
        call='gm_retained_core(m,n,dummy,alpha,a,lda,px,1,py,1,buffer);'
    else:
        sig='long m,long offset,float alpha,float*a,long lda,float*x,long inc_x,float*y,long inc_y,float*buffer'
        call='gm_retained_core(m,offset,alpha,a,lda,px,1,py,1,buffer);'
    wrapper=f'''\nint source_kernel({sig}){{
const long nx={nx},ny={ny};float*px=x;float*py=y;
if(inc_x!=1){{px=buffer;for(long i=0;i<nx;i++)px[i]=x[i*inc_x];}}
if(inc_y!=1){{py=buffer+nx;for(long i=0;i<ny;i++)py[i]=y[i*inc_y];}}
{call}
if(inc_y!=1)for(long i=0;i<ny;i++)y[i*inc_y]=py[i];
return 0;
}}\n'''
    row=copy.deepcopy(retained)
    row.update(id=retained['id'].replace('_retain','_materialize'),organization='materialize_vectors',code=text+wrapper)
    row['certificate'].update(
        required_contract='nonaliasing A/x/y/workspace; positive increments; workspace >= nx+ny floats; input values unchanged until consumers finish',
        checked_guard='original buffer identifier occurs only in the function signature',
        scope_extension='entry-boundary preparation and output restoration around the retained mapped core')
    return row
