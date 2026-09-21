// Source-layout packing adapter for the original FP32 m2 microkernels.
// A and B are row-major; C is column-major. Packing and output initialization
// are inside kernel(), and therefore included in the benchmark's timed call.
#include <riscv_vector.h>
#include <stddef.h>
#ifdef TRMM_BENCH
int source_kernel(long,long,long,float,float*,float*,float*,long,long);
#else
int source_kernel(long,long,long,float,float*,float*,float*,long);
#endif
extern "C" __attribute__((noinline))
void kernel(int M,int N,int K,const float*A,const float*B,float*C,float*W) {
    float* PA=W;
    float* PB=W+(size_t)M*K;
    size_t z=0;
    for(int i=0;i<M;) {
        size_t vl=__riscv_vsetvl_e32m2(M-i);
        for(int k=0;k<K;k++)
            for(size_t r=0;r<vl;r++) PA[z++]=A[(i+r)*K+k];
        i+=vl;
    }
    z=0;
    for(int j=0;j<N;) {
        int n=N-j>=8?8:(N-j>=4?4:(N-j>=2?2:1));
        for(int k=0;k<K;k++)
            for(int q=0;q<n;q++) PB[z++]=B[k*N+j+q];
        j+=n;
    }
    for(size_t t=0;t<(size_t)M*N;t++) C[t]=0;
#ifdef TRMM_BENCH
    source_kernel(M,N,K,1.f,PA,PB,C,M,0);
#else
    source_kernel(M,N,K,1.f,PA,PB,C,M);
#endif
}
