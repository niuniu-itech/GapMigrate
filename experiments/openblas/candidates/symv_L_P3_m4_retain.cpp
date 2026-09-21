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

static inline vfloat32m4_t gm_vlse32_4(const float*p,long stride,size_t vl){float a[vl];for(size_t i=0;i<vl;i++)a[i]=*(const float*)((const char*)p+i*stride);return __riscv_vle32_v_f32m4(a,vl);}

static inline void gm_vsse32_4(float*p,long stride,vfloat32m4_t v,size_t vl){float a[vl];__riscv_vse32_v_f32m4(a,v,vl);for(size_t i=0;i<vl;i++)*(float*)((char*)p+i*stride)=a[i];}
int source_kernel(long m, long offset, float alpha, float *a, long lda, float *x, long inc_x, float *y, long inc_y, float *buffer)
{
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
  vlmax = __riscv_vsetvlmax_e32m4();
  vfloat32m4_t va;
  vfloat32m4_t vx;
  vfloat32m4_t vy;
  vfloat32m4_t vr;
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
      vr = __riscv_vfmv_v_f_f32m4(0, vlmax);
      for (k = m - i; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m4(k);
        va = __riscv_vle32_v_f32m4(&a_ptr[i], vl);
        vy = __riscv_vle32_v_f32m4(&y[i], vl);
        vy = __riscv_vfmacc_vf_f32m4(vy, temp1, va, vl);
        __riscv_vse32_v_f32m4(&y[i], vy, vl);
        vx = __riscv_vle32_v_f32m4(&x[i], vl);
        vr = __riscv_vfmacc_vv_f32m4_tu(vr, vx, va, vl);
      }

      v_res = __riscv_vfredusum_vs_f32m4_f32m1(vr, v_z0, vlmax);
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
      vr = __riscv_vfmv_v_f_f32m4(0, vlmax);
      for (k = m - i; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m4(k);
        inc_yv = inc_y * vl;
        va = __riscv_vle32_v_f32m4(&a_ptr[i], vl);
        vy = gm_vlse32_4(&y[iy], stride_y, vl);
        vy = __riscv_vfmacc_vf_f32m4(vy, temp1, va, vl);
        gm_vsse32_4(&y[iy], stride_y, vy, vl);
        vx = __riscv_vle32_v_f32m4(&x[i], vl);
        vr = __riscv_vfmacc_vv_f32m4_tu(vr, vx, va, vl);
        iy += inc_yv;
      }

      v_res = __riscv_vfredusum_vs_f32m4_f32m1(vr, v_z0, vlmax);
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
      vr = __riscv_vfmv_v_f_f32m4(0, vlmax);
      for (k = m - i; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m4(k);
        inc_xv = inc_x * vl;
        va = __riscv_vle32_v_f32m4(&a_ptr[i], vl);
        vy = __riscv_vle32_v_f32m4(&y[i], vl);
        vy = __riscv_vfmacc_vf_f32m4(vy, temp1, va, vl);
        __riscv_vse32_v_f32m4(&y[i], vy, vl);
        vx = gm_vlse32_4(&x[ix], stride_x, vl);
        vr = __riscv_vfmacc_vv_f32m4_tu(vr, vx, va, vl);
        ix += inc_xv;
      }

      v_res = __riscv_vfredusum_vs_f32m4_f32m1(vr, v_z0, vlmax);
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
      vr = __riscv_vfmv_v_f_f32m4(0, vlmax);
      for (k = m - i; k > 0; k -= vl, i += vl)
      {
        vl = __riscv_vsetvl_e32m4(k);
        inc_xv = inc_x * vl;
        inc_yv = inc_y * vl;
        va = __riscv_vle32_v_f32m4(&a_ptr[i], vl);
        vy = gm_vlse32_4(&y[iy], stride_y, vl);
        vy = __riscv_vfmacc_vf_f32m4(vy, temp1, va, vl);
        gm_vsse32_4(&y[iy], stride_y, vy, vl);
        vx = gm_vlse32_4(&x[ix], stride_x, vl);
        vr = __riscv_vfmacc_vv_f32m4_tu(vr, vx, va, vl);
        ix += inc_xv;
        iy += inc_yv;
      }

      v_res = __riscv_vfredusum_vs_f32m4_f32m1(vr, v_z0, vlmax);
      y[jy] += alpha * __riscv_vfmv_f_s_f32m1_f32(v_res);
      jx += inc_x;
      jy += inc_y;
      a_ptr += lda;
    }

  }
  return 0;
}

