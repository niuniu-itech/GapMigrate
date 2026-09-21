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
static inline float gm_chunk_dot(vfloat32m1_t a,vfloat32m1_t b,size_t vl){
    float lanes[vl];auto product=__riscv_vfmul_vv_f32m1(a,b,vl);__riscv_vse32_v_f32m1(lanes,product,vl);float s=0.0f;for(size_t i=0;i<vl;i++)s+=lanes[i];return s;}
int source_kernel(long m, long offset, float alpha, float *a, long lda, float *x, long inc_x, float *y, long inc_y, float *buffer)
{
float gm_sum = 0.0f;
  long i;
  long j;
  long k;
  long ix;
  long iy;
  long jx;
  long jy;
  float temp1;
  float *a_ptr = a;
  vfloat32m1_t v_res;
  vfloat32m1_t v_z0;
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  size_t vl;
  v_z0 = __riscv_vfmv_v_f_f32m1(0, vlmax);
  vlmax = __riscv_vsetvlmax_e32m1();
  vfloat32m1_t va;
  vfloat32m1_t vx;
  vfloat32m1_t vy;
  vfloat32m1_t vr;
  long stride_x;
  long stride_y;
  long inc_xv;
  long inc_yv;
  if ((inc_x == 1) && (inc_y == 1))
  {
    for (j = 0; j < offset; j++)
    {
      temp1 = alpha * x[j];
      y[j] += temp1 * a_ptr[j];
      i = j + 1;
      gm_sum = 0.0f;
      for (k = m - i; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m1(k);
        va = __riscv_vle32_v_f32m1(&a_ptr[i], vl);
        vy = __riscv_vle32_v_f32m1(&y[i], vl);
        vy = __riscv_vfmacc_vf_f32m1(vy, temp1, va, vl);
        __riscv_vse32_v_f32m1(&y[i], vy, vl);
        vx = __riscv_vle32_v_f32m1(&x[i], vl);
        gm_sum += gm_chunk_dot(vx, va, vl);
      }

      v_res = __riscv_vfmv_s_f_f32m1(gm_sum, 1);
      y[j] += alpha * __riscv_vfmv_f_s_f32m1_f32(v_res);
      a_ptr += lda;
    }

  }
  else
    if (inc_x == 1)
  {
    jy = 0;
    stride_y = inc_y * (sizeof(float));
    for (j = 0; j < offset; j++)
    {
      temp1 = alpha * x[j];
      y[jy] += temp1 * a_ptr[j];
      iy = jy + inc_y;
      i = j + 1;
      gm_sum = 0.0f;
      for (k = m - i; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m1(k);
        inc_yv = inc_y * vl;
        va = __riscv_vle32_v_f32m1(&a_ptr[i], vl);
        vy = __riscv_vlse32_v_f32m1(&y[iy], stride_y, vl);
        vy = __riscv_vfmacc_vf_f32m1(vy, temp1, va, vl);
        __riscv_vsse32_v_f32m1(&y[iy], stride_y, vy, vl);
        vx = __riscv_vle32_v_f32m1(&x[i], vl);
        gm_sum += gm_chunk_dot(vx, va, vl);
        iy += inc_yv;
      }

      v_res = __riscv_vfmv_s_f_f32m1(gm_sum, 1);
      y[jy] += alpha * __riscv_vfmv_f_s_f32m1_f32(v_res);
      jy += inc_y;
      a_ptr += lda;
    }

  }
  else
    if (inc_y == 1)
  {
    jx = 0;
    stride_x = inc_x * (sizeof(float));
    for (j = 0; j < offset; j++)
    {
      temp1 = alpha * x[jx];
      y[j] += temp1 * a_ptr[j];
      ix = jx + inc_x;
      i = j + 1;
      gm_sum = 0.0f;
      for (k = m - i; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m1(k);
        inc_xv = inc_x * vl;
        va = __riscv_vle32_v_f32m1(&a_ptr[i], vl);
        vy = __riscv_vle32_v_f32m1(&y[i], vl);
        vy = __riscv_vfmacc_vf_f32m1(vy, temp1, va, vl);
        __riscv_vse32_v_f32m1(&y[i], vy, vl);
        vx = __riscv_vlse32_v_f32m1(&x[ix], stride_x, vl);
        gm_sum += gm_chunk_dot(vx, va, vl);
        ix += inc_xv;
      }

      v_res = __riscv_vfmv_s_f_f32m1(gm_sum, 1);
      y[j] += alpha * __riscv_vfmv_f_s_f32m1_f32(v_res);
      jx += inc_x;
      a_ptr += lda;
    }

  }
  else
  {
    stride_x = inc_x * (sizeof(float));
    stride_y = inc_y * (sizeof(float));
    jx = 0;
    jy = 0;
    for (j = 0; j < offset; j++)
    {
      temp1 = alpha * x[jx];
      y[jy] += temp1 * a_ptr[j];
      ix = jx + inc_x;
      iy = jy + inc_y;
      i = j + 1;
      gm_sum = 0.0f;
      for (k = m - i; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m1(k);
        inc_xv = inc_x * vl;
        inc_yv = inc_y * vl;
        va = __riscv_vle32_v_f32m1(&a_ptr[i], vl);
        vy = __riscv_vlse32_v_f32m1(&y[iy], stride_y, vl);
        vy = __riscv_vfmacc_vf_f32m1(vy, temp1, va, vl);
        __riscv_vsse32_v_f32m1(&y[iy], stride_y, vy, vl);
        vx = __riscv_vlse32_v_f32m1(&x[ix], stride_x, vl);
        gm_sum += gm_chunk_dot(vx, va, vl);
        ix += inc_xv;
        iy += inc_yv;
      }

      v_res = __riscv_vfmv_s_f_f32m1(gm_sum, 1);
      y[jy] += alpha * __riscv_vfmv_f_s_f32m1_f32(v_res);
      jx += inc_x;
      jy += inc_y;
      a_ptr += lda;
    }

  }
  return 0;
}

