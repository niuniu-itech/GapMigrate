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
int source_kernel(long m, long n, long dummy1, float alpha, float *a, long lda, float *x, long inc_x, float *y, long inc_y, float *buffer)
{
  long i;
  long j;
  float *a_ptr;
  float *x_ptr;
  vfloat32m1_t va;
  vfloat32m1_t vx;
  vfloat32m1_t vr;
  vfloat32m1_t v_res;
  vfloat32m1_t v_z0;
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  v_z0 = __riscv_vfmv_v_f_f32m1(0, vlmax);
  vlmax = __riscv_vsetvlmax_e32m1();
  if (inc_x == 1)
  {
    for (i = 0; i < n; i++)
    {
      j = m;
      a_ptr = a;
      x_ptr = x;
      vr = __riscv_vfmv_v_f_f32m1(0, vlmax);
      for (size_t vl; j > 0; j -= vl, a_ptr += vl, x_ptr += vl)
      {
        vl = __riscv_vsetvl_e32m1(j);
        va = __riscv_vle32_v_f32m1(a_ptr, vl);
        vx = __riscv_vle32_v_f32m1(x_ptr, vl);
        vr = __riscv_vfmacc_vv_f32m1_tu(vr, va, vx, vl);
      }

      v_res = __riscv_vfredusum_vs_f32m1_f32m1(vr, v_z0, vlmax);
      *y += alpha * __riscv_vfmv_f_s_f32m1_f32(v_res);
      y += inc_y;
      a += lda;
    }

  }
  else
  {
    long stride_x = inc_x * (sizeof(float));
    for (i = 0; i < n; i++)
    {
      j = m;
      a_ptr = a;
      x_ptr = x;
      vr = __riscv_vfmv_v_f_f32m1(0, vlmax);
      for (size_t vl; j > 0; j -= vl, a_ptr += vl, x_ptr += vl * inc_x)
      {
        vl = __riscv_vsetvl_e32m1(j);
        va = __riscv_vle32_v_f32m1(a_ptr, vl);
        vx = __riscv_vlse32_v_f32m1(x_ptr, stride_x, vl);
        vr = __riscv_vfmacc_vv_f32m1_tu(vr, va, vx, vl);
      }

      v_res = __riscv_vfredusum_vs_f32m1_f32m1(vr, v_z0, vlmax);
      *y += alpha * __riscv_vfmv_f_s_f32m1_f32(v_res);
      y += inc_y;
      a += lda;
    }

  }
  return 0;
}

