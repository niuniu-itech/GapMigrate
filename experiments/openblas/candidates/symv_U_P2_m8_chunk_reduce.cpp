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

static inline vfloat32m1_t gm_reduce_sum_8(vfloat32m8_t x,vfloat32m1_t seed,size_t vl){
 float lanes[64];__riscv_vse32_v_f32m8(lanes,x,vl);
 float s=__riscv_vfmv_f_s_f32m1_f32(seed);
 for(size_t i=0;i<vl;i++)s+=lanes[i];
 return __riscv_vfmv_s_f_f32m1(s,__riscv_vsetvlmax_e32m1());
}
static inline vfloat32m1_t gm_reduce_max_8(vfloat32m8_t x,vfloat32m1_t seed,size_t vl){
 float lanes[64];__riscv_vse32_v_f32m8(lanes,x,vl);
 float s=__riscv_vfmv_f_s_f32m1_f32(seed);
 for(size_t i=0;i<vl;i++)s=s>lanes[i]?s:lanes[i];
 return __riscv_vfmv_s_f_f32m1(s,__riscv_vsetvlmax_e32m1());
}
static inline float gm_chunk_dot(vfloat32m8_t a,vfloat32m8_t b,size_t vl){
    float lanes[vl];auto product=__riscv_vfmul_vv_f32m8(a,b,vl);__riscv_vse32_v_f32m8(lanes,product,vl);float s=0.0f;for(size_t i=0;i<vl;i++)s+=lanes[i];return s;}
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
  size_t vl_max = __riscv_vsetvlmax_e32m1();
  size_t vl;
  v_z0 = __riscv_vfmv_v_f_f32m1(0, vl_max);
  vl_max = __riscv_vsetvlmax_e32m8();
  vfloat32m8_t va;
  vfloat32m8_t vx;
  vfloat32m8_t vy;
  vfloat32m8_t vr;
  long stride_x;
  long stride_y;
  long inc_xv;
  long inc_yv;
  long m1 = m - offset;
  if ((inc_x == 1) && (inc_y == 1))
  {
    a_ptr += m1 * lda;
    for (j = m1; j < m; j++)
    {
      temp1 = alpha * x[j];
      i = 0;
      gm_sum = 0.0f;
      for (k = j; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m8(k);
        vy = __riscv_vle32_v_f32m8(&y[i], vl);
        va = __riscv_vle32_v_f32m8(&a_ptr[i], vl);
        vy = __riscv_vfmacc_vf_f32m8(vy, temp1, va, vl);
        __riscv_vse32_v_f32m8(&y[i], vy, vl);
        vx = __riscv_vle32_v_f32m8(&x[i], vl);
        gm_sum += gm_chunk_dot(vx, va, vl);
      }

      v_res = __riscv_vfmv_s_f_f32m1(gm_sum, 1);
      y[j] += (temp1 * a_ptr[j]) + (alpha * __riscv_vfmv_f_s_f32m1_f32(v_res));
      a_ptr += lda;
    }

  }
  else
    if (inc_x == 1)
  {
    jy = m1 * inc_y;
    a_ptr += m1 * lda;
    stride_y = inc_y * (sizeof(float));
    for (j = m1; j < m; j++)
    {
      temp1 = alpha * x[j];
      iy = 0;
      i = 0;
      gm_sum = 0.0f;
      for (k = j; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m8(k);
        inc_yv = inc_y * vl;
        vy = __riscv_vlse32_v_f32m8(&y[iy], stride_y, vl);
        va = __riscv_vle32_v_f32m8(&a_ptr[i], vl);
        vy = __riscv_vfmacc_vf_f32m8(vy, temp1, va, vl);
        __riscv_vsse32_v_f32m8(&y[iy], stride_y, vy, vl);
        vx = __riscv_vle32_v_f32m8(&x[i], vl);
        gm_sum += gm_chunk_dot(vx, va, vl);
        iy += inc_yv;
      }

      v_res = __riscv_vfmv_s_f_f32m1(gm_sum, 1);
      y[jy] += (temp1 * a_ptr[j]) + (alpha * __riscv_vfmv_f_s_f32m1_f32(v_res));
      a_ptr += lda;
      jy += inc_y;
    }

  }
  else
    if (inc_y == 1)
  {
    jx = m1 * inc_x;
    a_ptr += m1 * lda;
    stride_x = inc_x * (sizeof(float));
    for (j = m1; j < m; j++)
    {
      temp1 = alpha * x[jx];
      ix = 0;
      i = 0;
      gm_sum = 0.0f;
      for (k = j; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m8(k);
        inc_xv = inc_x * vl;
        vy = __riscv_vle32_v_f32m8(&y[i], vl);
        va = __riscv_vle32_v_f32m8(&a_ptr[i], vl);
        vy = __riscv_vfmacc_vf_f32m8(vy, temp1, va, vl);
        __riscv_vse32_v_f32m8(&y[i], vy, vl);
        vx = __riscv_vlse32_v_f32m8(&x[ix], stride_x, vl);
        gm_sum += gm_chunk_dot(vx, va, vl);
        ix += inc_xv;
      }

      v_res = __riscv_vfmv_s_f_f32m1(gm_sum, 1);
      y[j] += (temp1 * a_ptr[j]) + (alpha * __riscv_vfmv_f_s_f32m1_f32(v_res));
      a_ptr += lda;
      jx += inc_x;
    }

  }
  else
  {
    jx = m1 * inc_x;
    jy = m1 * inc_y;
    a_ptr += m1 * lda;
    stride_x = inc_x * (sizeof(float));
    stride_y = inc_y * (sizeof(float));
    for (j = m1; j < m; j++)
    {
      temp1 = alpha * x[jx];
      ix = 0;
      iy = 0;
      i = 0;
      gm_sum = 0.0f;
      for (k = j; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m8(k);
        inc_xv = inc_x * vl;
        inc_yv = inc_y * vl;
        vy = __riscv_vlse32_v_f32m8(&y[iy], stride_y, vl);
        va = __riscv_vle32_v_f32m8(&a_ptr[i], vl);
        vy = __riscv_vfmacc_vf_f32m8(vy, temp1, va, vl);
        __riscv_vsse32_v_f32m8(&y[iy], stride_y, vy, vl);
        vx = __riscv_vlse32_v_f32m8(&x[ix], stride_x, vl);
        gm_sum += gm_chunk_dot(vx, va, vl);
        ix += inc_xv;
        iy += inc_yv;
      }

      v_res = __riscv_vfmv_s_f_f32m1(gm_sum, 1);
      y[jy] += (temp1 * a_ptr[j]) + (alpha * __riscv_vfmv_f_s_f32m1_f32(v_res));
      a_ptr += lda;
      jx += inc_x;
      jy += inc_y;
    }

  }
  return 0;
}

