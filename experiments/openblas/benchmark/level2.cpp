#include <riscv_vector.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#ifndef FAMILY
#define FAMILY 0
#endif
#if FAMILY<2
int source_kernel(long,long,long,float,float*,long,float*,long,float*,long,float*);
#else
int source_kernel(long,long,float,float*,long,float*,long,float*,long,float*);
#endif
int main(int argc,char**argv){
 if(argc!=6){fprintf(stderr,"usage: benchmark M N inc_x inc_y seed\n");return 64;}
 if(__riscv_vsetvlmax_e32m1()!=8){puts("{\"status\":\"vlen_contract_failed\"}");return 3;}
 int m=atoi(argv[1]),n=atoi(argv[2]),sx=atoi(argv[3]),sy=atoi(argv[4]),seed=atoi(argv[5]);
 if(m<=0||n<=0||sx<=0||sy<=0)return 64;
 if(FAMILY>=2)n=m;int nx=FAMILY==1?m:n,ny=FAMILY==1?n:m;int lda=m+3;
 std::vector<float>a(lda*n+16,12345),x(nx*sx+16,12345),y(ny*sy+16,12345),work(2*m*n+64,0);
 uint64_t state=0x9e3779b97f4a7c15ULL+(uint64_t)seed;
 auto random_value=[&](){state^=state<<13;state^=state>>7;state^=state<<17;return float(int((state>>16)%20001)-10000)/20000.f;};
 for(int j=0;j<n;j++)for(int i=0;i<m;i++)a[i+j*lda]=random_value();
 for(int i=0;i<nx;i++)x[i*sx]=random_value();
 for(int i=0;i<ny;i++)y[i*sy]=random_value();
 auto aa=a,xx=x,yy=y;std::vector<double>ref(ny);float alpha=.75f;
 for(int i=0;i<ny;i++){double sum=0;for(int j=0;j<nx;j++){
 double v=FAMILY==1?a[j+i*lda]:a[i+j*lda];
 if(FAMILY==2)v=a[std::max(i,j)+std::min(i,j)*lda];
 if(FAMILY==3)v=a[std::min(i,j)+std::max(i,j)*lda];
 sum+=v*x[j*sx];}ref[i]=y[i*sy]+alpha*sum;}
 auto call=[&](){
#if FAMILY<2
 source_kernel(m,n,0,alpha,a.data(),lda,x.data(),sx,y.data(),sy,work.data());
#else
 source_kernel(m,m,alpha,a.data(),lda,x.data(),sx,y.data(),sy,work.data());
#endif
 };
 fprintf(stderr,"qualification_begin\n");fflush(stderr);
 call();double err=0;bool ok=a==aa&&x==xx;
 for(int i=0;i<ny;i++){double e=std::abs(y[i*sy]-ref[i]);err=std::max(err,e);if(!std::isfinite(y[i*sy])||e>2e-5+2e-4*std::abs(ref[i]))ok=false;}
 for(size_t i=0;i<y.size();i++)if((i%sy!=0||i/size_t(sy)>=size_t(ny))&&y[i]!=yy[i])ok=false;
 if(!ok){printf("{\"status\":\"numerical_failed\",\"max_abs_error\":%.9g}\n",err);return 2;}
 fprintf(stderr,"timing_begin\n");fflush(stderr);
 std::vector<double>samples;for(int s=0;s<7;s++){y=yy;auto t=std::chrono::steady_clock::now();for(int r=0;r<20;r++)call();samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count()/20);}
 bool post_ok=a==aa&&x==xx;double post_error=0;
 for(int i=0;i<ny;i++){double expected=yy[i*sy]+20*(ref[i]-yy[i*sy]);double e=std::abs(y[i*sy]-expected);post_error=std::max(post_error,e);if(!std::isfinite(y[i*sy])||e>20*2e-5+2e-4*std::abs(expected))post_ok=false;}
 for(size_t i=0;i<y.size();i++)if((i%sy!=0||i/size_t(sy)>=size_t(ny))&&y[i]!=yy[i])post_ok=false;
 if(!post_ok){printf("{\"status\":\"postcheck_failed\",\"max_abs_error\":%.9g}\n",post_error);return 3;}
 auto sorted=samples;std::sort(sorted.begin(),sorted.end());printf("{\"status\":\"pass\",\"latency_us\":%.9g,\"max_abs_error\":%.9g,\"samples_us\":[",sorted[3],err);for(int i=0;i<7;i++)printf("%s%.9g",i?",":"",samples[i]);printf("]}\n");
}
