#include <riscv_vector.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <vector>
extern "C" void kernel(int,int,int,const float*,const float*,float*,float*);
static uint64_t state=20260916;
static float random_float(){state^=state<<13;state^=state>>7;state^=state<<17;return (float)((state>>16)%20001)/10000.f-1.f;}
static double now(){timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t hash(const std::vector<float>&v){uint64_t h=1469598103934665603ULL;const unsigned char*p=(const unsigned char*)v.data();for(size_t i=0;i<v.size()*4;i++)h=(h^p[i])*1099511628211ULL;return h;}
int main(int argc,char**argv){
 if(argc<6)return 2;
 int M=atoi(argv[2]),N=atoi(argv[3]),K=atoi(argv[4]),seed=atoi(argv[5]);
 if(M<=0||N<=0||K<=0)return 64;
 bool diagnose=argc>6&&atoi(argv[6]);state=20260916+(uint64_t)seed;
 if(__riscv_vsetvlmax_e32m1()!=8){puts("{\"status\":\"vlen_contract_failed\"}");return 3;}
 // Environment/format probe is outside the kernel timing and legality scope.
 size_t na=(size_t)M*K;
 size_t nb=(size_t)K*N;
 std::vector<float>A(na),B(nb),C((size_t)M*N+32,123456.f),W((size_t)M*K+(size_t)K*N+(size_t)M*N+4096,0.f);
 for(auto&x:A)x=random_float();for(auto&x:B)x=random_float();
#ifdef TRMM_BENCH
 if(K!=N)return 64;
 for(int k=0;k<K;k++)for(int j=0;j<N;j++)if(k>j)B[(size_t)k*N+j]=0.f;
#endif
 if(diagnose)for(auto&x:A)x+=10000.f;
 uint64_t ha=hash(A),hb=hash(B);std::vector<double>ref((size_t)M*N);
 {for(int j=0;j<N;j++)for(int i=0;i<M;i++){double s=0;for(int k=0;k<K;k++)s+=(double)A[(size_t)i*K+k]*B[(size_t)k*N+j];ref[(size_t)j*M+i]=s;}}
 auto execute=[&](){kernel(M,N,K,A.data(),B.data(),C.data(),W.data());};
 double atol=2e-4,rtol=2e-4;
 double max_abs=0,max_scaled=0;size_t bad=0;
 auto check=[&](){bad=0;max_abs=max_scaled=0;for(size_t i=0;i<ref.size();i++){double e=fabs((double)C[i]-ref[i]),tol=atol+rtol*fabs(ref[i]);max_abs=std::max(max_abs,e);max_scaled=std::max(max_scaled,tol?e/tol:e);if(!std::isfinite(C[i])||e>tol)bad++;}for(size_t i=ref.size();i<C.size();i++)if(C[i]!=123456.f)bad++;if(hash(A)!=ha||hash(B)!=hb)bad++;};
 fprintf(stderr,"qualification_begin\n");fflush(stderr);
 execute();check();if(bad){printf("{\"status\":\"numerical_failed\",\"bad\":%zu,\"max_abs\":%.12g,\"max_scaled\":%.12g,\"diagnostic\":%s}\n",bad,max_abs,max_scaled,diagnose?"true":"false");return 4;}
 if(diagnose){printf("{\"status\":\"pass\",\"diagnostic\":true,\"max_abs\":%.12g}\n",max_abs);return 0;}
 fprintf(stderr,"timing_begin\n");fflush(stderr);
 execute();double t0=now();execute();double one=now()-t0;int batch=(int)std::min(2000.,std::max(1.,0.003/std::max(one,1e-8)));
 std::vector<double>times;for(int s=0;s<7;s++){t0=now();for(int b=0;b<batch;b++)execute();times.push_back((now()-t0)*1e6/batch);}
 check();auto raw=times;std::sort(times.begin(),times.end());
 printf("{\"status\":\"%s\",\"latency_us\":%.9g,\"min_us\":%.9g,\"max_us\":%.9g,\"batch\":%d,\"samples\":7,\"max_abs\":%.12g,\"max_scaled\":%.12g,\"input_a_fnv64\":\"%016llx\",\"input_b_fnv64\":\"%016llx\",\"raw_us\":[",bad?"postcheck_failed":"pass",times[3],times.front(),times.back(),batch,max_abs,max_scaled,(unsigned long long)ha,(unsigned long long)hb);
 for(size_t i=0;i<raw.size();i++)printf("%s%.9g",i?",":"",raw[i]);puts("]}");return bad?5:0;
}
