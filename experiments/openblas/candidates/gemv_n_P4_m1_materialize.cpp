/***************************************************************************
Copyright (c) 2022, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/
// Historical source-derived candidate snapshot. See catalog.json.
#include <stddef.h>
#include <riscv_vector.h>

static inline vfloat32m1_t gm_reduce_sum_1(vfloat32m1_t x,vfloat32m1_t seed,size_t vl){
 float lanes[64];__riscv_vse32_v_f32m1(lanes,x,vl);
 float s=__riscv_vfmv_f_s_f32m1_f32(seed);
 for(size_t i=0;i<vl;i++)s+=lanes[i];
 return __riscv_vfmv_s_f_f32m1(s,__riscv_vsetvlmax_e32m1());
}
static inline vfloat32m1_t gm_reduce_max_1(vfloat32m1_t x,vfloat32m1_t seed,size_t vl){
 float lanes[64];__riscv_vse32_v_f32m1(lanes,x,vl);
 float s=__riscv_vfmv_f_s_f32m1_f32(seed);
 for(size_t i=0;i<vl;i++)s=s>lanes[i]?s:lanes[i];
 return __riscv_vfmv_s_f_f32m1(s,__riscv_vsetvlmax_e32m1());
}

static inline vfloat32m1_t gm_vlse32_1(const float*p,long stride,size_t vl){float a[vl];for(size_t i=0;i<vl;i++)a[i]=*(const float*)((const char*)p+i*stride);return __riscv_vle32_v_f32m1(a,vl);}

static inline void gm_vsse32_1(float*p,long stride,vfloat32m1_t v,size_t vl){float a[vl];__riscv_vse32_v_f32m1(a,v,vl);for(size_t i=0;i<vl;i++)*(float*)((char*)p+i*stride)=a[i];}
static inline __attribute__((always_inline)) int gm_retained_core(long m, long n, long dummy1, float alpha, float *a, long lda, float *x, long inc_x, float *y, long inc_y, float *buffer)
{
  if (n < 0)
    return 0;
  float *a_ptr;
  float *x_ptr;
  long i;
  vfloat32m1_t va;
  vfloat32m1_t vy;
  if (inc_y == 1)
  {
    for (size_t vl; m > 0; m -= vl, y += vl, a += vl)
    {
      vl = __riscv_vsetvl_e32m1(m);
      a_ptr = a;
      x_ptr = x;
      vy = __riscv_vle32_v_f32m1(y, vl);
      for (i = 0; i < n; i++)
      {
        va = __riscv_vle32_v_f32m1(a_ptr, vl);
        vy = __riscv_vfmacc_vf_f32m1(vy, alpha * (*x_ptr), va, vl);
        a_ptr += lda;
        x_ptr += inc_x;
      }

      __riscv_vse32_v_f32m1(y, vy, vl);
    }

  }
  else
  {
    long stride_y = inc_y * (sizeof(float));
    for (size_t vl; m > 0; m -= vl, y += vl * inc_y, a += vl)
    {
      vl = __riscv_vsetvl_e32m1(m);
      a_ptr = a;
      x_ptr = x;
      vy = gm_vlse32_1(y, stride_y, vl);
      for (i = 0; i < n; i++)
      {
        va = __riscv_vle32_v_f32m1(a_ptr, vl);
        vy = __riscv_vfmacc_vf_f32m1(vy, alpha * (*x_ptr), va, vl);
        a_ptr += lda;
        x_ptr += inc_x;
      }

      gm_vsse32_1(y, stride_y, vy, vl);
    }

  }
  return 0;
}


int source_kernel(long m,long n,long dummy,float alpha,float*a,long lda,float*x,long inc_x,float*y,long inc_y,float*buffer){
  const long nx=n,ny=m;float*px=x;float*py=y;
  if(inc_x!=1){px=buffer;for(long i=0;i<nx;i++)px[i]=x[i*inc_x];}
  if(inc_y!=1){py=buffer+nx;for(long i=0;i<ny;i++)py[i]=y[i*inc_y];}
  gm_retained_core(m,n,dummy,alpha,a,lda,px,1,py,1,buffer);
  if(inc_y!=1)for(long i=0;i<ny;i++)y[i*inc_y]=py[i];
  return 0;
}
