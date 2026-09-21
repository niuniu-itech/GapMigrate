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
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <string.h>
int source_kernel(long bm, long bn, long bk, float alpha, float *ba, float *bb, float *C, long ldc)
{
  long i;
  long j;
  long k;
  float *C0;
  float *C1;
  float *C2;
  float *C3;
  float *C4;
  float *C5;
  float *C6;
  float *C7;
  float *ptrba;
  float *ptrbb;
  vfloat32m2_t va0;
  vfloat32m2_t va1;
  vfloat32m2_t va2;
  vfloat32m2_t va3;
  vfloat32m2_t va4;
  vfloat32m2_t va5;
  vfloat32m2_t va6;
  vfloat32m2_t va7;
  vfloat32m2_t vres0;
  vfloat32m2_t vres1;
  vfloat32m2_t vres2;
  vfloat32m2_t vres3;
  vfloat32m2_t vres4;
  vfloat32m2_t vres5;
  vfloat32m2_t vres6;
  vfloat32m2_t vres7;
  size_t vl;
  for (j = bn / 8; j > 0; j--)
  {
    C0 = C;
    C1 = C0 + ldc;
    C2 = C1 + ldc;
    C3 = C2 + ldc;
    C4 = C3 + ldc;
    C5 = C4 + ldc;
    C6 = C5 + ldc;
    C7 = C6 + ldc;
    ptrba = ba;
    for (i = bm; i > 0; i -= vl)
    {
      vl = __riscv_vsetvl_e32m2(i);
      ptrbb = bb;
      vres0 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres1 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres2 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres3 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres4 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres5 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres6 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres7 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      for (k = bk / 8; k > 0; k--)
      {
        va0 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        va1 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va0, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va0, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va0, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va0, vl);
        vres4 = __riscv_vfmacc_vf_f32m2(vres4, *(ptrbb + 4), va0, vl);
        vres5 = __riscv_vfmacc_vf_f32m2(vres5, *(ptrbb + 5), va0, vl);
        vres6 = __riscv_vfmacc_vf_f32m2(vres6, *(ptrbb + 6), va0, vl);
        vres7 = __riscv_vfmacc_vf_f32m2(vres7, *(ptrbb + 7), va0, vl);
        ptrbb += 8;
        va2 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va1, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va1, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va1, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va1, vl);
        vres4 = __riscv_vfmacc_vf_f32m2(vres4, *(ptrbb + 4), va1, vl);
        vres5 = __riscv_vfmacc_vf_f32m2(vres5, *(ptrbb + 5), va1, vl);
        vres6 = __riscv_vfmacc_vf_f32m2(vres6, *(ptrbb + 6), va1, vl);
        vres7 = __riscv_vfmacc_vf_f32m2(vres7, *(ptrbb + 7), va1, vl);
        ptrbb += 8;
        va3 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va2, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va2, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va2, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va2, vl);
        vres4 = __riscv_vfmacc_vf_f32m2(vres4, *(ptrbb + 4), va2, vl);
        vres5 = __riscv_vfmacc_vf_f32m2(vres5, *(ptrbb + 5), va2, vl);
        vres6 = __riscv_vfmacc_vf_f32m2(vres6, *(ptrbb + 6), va2, vl);
        vres7 = __riscv_vfmacc_vf_f32m2(vres7, *(ptrbb + 7), va2, vl);
        ptrbb += 8;
        va4 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va3, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va3, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va3, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va3, vl);
        vres4 = __riscv_vfmacc_vf_f32m2(vres4, *(ptrbb + 4), va3, vl);
        vres5 = __riscv_vfmacc_vf_f32m2(vres5, *(ptrbb + 5), va3, vl);
        vres6 = __riscv_vfmacc_vf_f32m2(vres6, *(ptrbb + 6), va3, vl);
        vres7 = __riscv_vfmacc_vf_f32m2(vres7, *(ptrbb + 7), va3, vl);
        ptrbb += 8;
        va5 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va4, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va4, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va4, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va4, vl);
        vres4 = __riscv_vfmacc_vf_f32m2(vres4, *(ptrbb + 4), va4, vl);
        vres5 = __riscv_vfmacc_vf_f32m2(vres5, *(ptrbb + 5), va4, vl);
        vres6 = __riscv_vfmacc_vf_f32m2(vres6, *(ptrbb + 6), va4, vl);
        vres7 = __riscv_vfmacc_vf_f32m2(vres7, *(ptrbb + 7), va4, vl);
        ptrbb += 8;
        va6 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va5, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va5, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va5, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va5, vl);
        vres4 = __riscv_vfmacc_vf_f32m2(vres4, *(ptrbb + 4), va5, vl);
        vres5 = __riscv_vfmacc_vf_f32m2(vres5, *(ptrbb + 5), va5, vl);
        vres6 = __riscv_vfmacc_vf_f32m2(vres6, *(ptrbb + 6), va5, vl);
        vres7 = __riscv_vfmacc_vf_f32m2(vres7, *(ptrbb + 7), va5, vl);
        ptrbb += 8;
        va7 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va6, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va6, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va6, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va6, vl);
        vres4 = __riscv_vfmacc_vf_f32m2(vres4, *(ptrbb + 4), va6, vl);
        vres5 = __riscv_vfmacc_vf_f32m2(vres5, *(ptrbb + 5), va6, vl);
        vres6 = __riscv_vfmacc_vf_f32m2(vres6, *(ptrbb + 6), va6, vl);
        vres7 = __riscv_vfmacc_vf_f32m2(vres7, *(ptrbb + 7), va6, vl);
        ptrbb += 8;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va7, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va7, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va7, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va7, vl);
        vres4 = __riscv_vfmacc_vf_f32m2(vres4, *(ptrbb + 4), va7, vl);
        vres5 = __riscv_vfmacc_vf_f32m2(vres5, *(ptrbb + 5), va7, vl);
        vres6 = __riscv_vfmacc_vf_f32m2(vres6, *(ptrbb + 6), va7, vl);
        vres7 = __riscv_vfmacc_vf_f32m2(vres7, *(ptrbb + 7), va7, vl);
        ptrbb += 8;
      }

      for (k = bk & 7; k > 0; k--)
      {
        va0 = __riscv_vle32_v_f32m2(ptrba, vl);
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va0, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va0, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va0, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va0, vl);
        vres4 = __riscv_vfmacc_vf_f32m2(vres4, *(ptrbb + 4), va0, vl);
        vres5 = __riscv_vfmacc_vf_f32m2(vres5, *(ptrbb + 5), va0, vl);
        vres6 = __riscv_vfmacc_vf_f32m2(vres6, *(ptrbb + 6), va0, vl);
        vres7 = __riscv_vfmacc_vf_f32m2(vres7, *(ptrbb + 7), va0, vl);
        ptrbb += 8;
        ptrba += vl;
      }

      va0 = __riscv_vle32_v_f32m2(C0, vl);
      va0 = __riscv_vfmacc_vf_f32m2(va0, alpha, vres0, vl);
      __riscv_vse32_v_f32m2(C0, va0, vl);
      va1 = __riscv_vle32_v_f32m2(C1, vl);
      va1 = __riscv_vfmacc_vf_f32m2(va1, alpha, vres1, vl);
      __riscv_vse32_v_f32m2(C1, va1, vl);
      va2 = __riscv_vle32_v_f32m2(C2, vl);
      va2 = __riscv_vfmacc_vf_f32m2(va2, alpha, vres2, vl);
      __riscv_vse32_v_f32m2(C2, va2, vl);
      va3 = __riscv_vle32_v_f32m2(C3, vl);
      va3 = __riscv_vfmacc_vf_f32m2(va3, alpha, vres3, vl);
      __riscv_vse32_v_f32m2(C3, va3, vl);
      va4 = __riscv_vle32_v_f32m2(C4, vl);
      va4 = __riscv_vfmacc_vf_f32m2(va4, alpha, vres4, vl);
      __riscv_vse32_v_f32m2(C4, va4, vl);
      va5 = __riscv_vle32_v_f32m2(C5, vl);
      va5 = __riscv_vfmacc_vf_f32m2(va5, alpha, vres5, vl);
      __riscv_vse32_v_f32m2(C5, va5, vl);
      va6 = __riscv_vle32_v_f32m2(C6, vl);
      va6 = __riscv_vfmacc_vf_f32m2(va6, alpha, vres6, vl);
      __riscv_vse32_v_f32m2(C6, va6, vl);
      va7 = __riscv_vle32_v_f32m2(C7, vl);
      va7 = __riscv_vfmacc_vf_f32m2(va7, alpha, vres7, vl);
      __riscv_vse32_v_f32m2(C7, va7, vl);
      C0 += vl;
      C1 += vl;
      C2 += vl;
      C3 += vl;
      C4 += vl;
      C5 += vl;
      C6 += vl;
      C7 += vl;
    }

    bb += bk << 3;
    C += ldc << 3;
  }

  if (bn & 4)
  {
    C0 = C;
    C1 = C0 + ldc;
    C2 = C1 + ldc;
    C3 = C2 + ldc;
    ptrba = ba;
    for (i = bm; i > 0; i -= vl)
    {
      vl = __riscv_vsetvl_e32m2(i);
      ptrbb = bb;
      vres0 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres1 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres2 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres3 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      for (k = bk / 8; k > 0; k--)
      {
        va0 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        va1 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va0, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va0, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va0, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va0, vl);
        ptrbb += 4;
        va2 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va1, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va1, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va1, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va1, vl);
        ptrbb += 4;
        va3 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va2, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va2, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va2, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va2, vl);
        ptrbb += 4;
        va4 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va3, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va3, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va3, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va3, vl);
        ptrbb += 4;
        va5 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va4, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va4, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va4, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va4, vl);
        ptrbb += 4;
        va6 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va5, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va5, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va5, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va5, vl);
        ptrbb += 4;
        va7 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va6, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va6, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va6, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va6, vl);
        ptrbb += 4;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va7, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va7, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va7, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va7, vl);
        ptrbb += 4;
      }

      for (k = bk & 7; k > 0; k--)
      {
        va0 = __riscv_vle32_v_f32m2(ptrba, vl);
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va0, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va0, vl);
        vres2 = __riscv_vfmacc_vf_f32m2(vres2, *(ptrbb + 2), va0, vl);
        vres3 = __riscv_vfmacc_vf_f32m2(vres3, *(ptrbb + 3), va0, vl);
        ptrbb += 4;
        ptrba += vl;
      }

      va0 = __riscv_vle32_v_f32m2(C0, vl);
      va0 = __riscv_vfmacc_vf_f32m2(va0, alpha, vres0, vl);
      __riscv_vse32_v_f32m2(C0, va0, vl);
      va1 = __riscv_vle32_v_f32m2(C1, vl);
      va1 = __riscv_vfmacc_vf_f32m2(va1, alpha, vres1, vl);
      __riscv_vse32_v_f32m2(C1, va1, vl);
      va2 = __riscv_vle32_v_f32m2(C2, vl);
      va2 = __riscv_vfmacc_vf_f32m2(va2, alpha, vres2, vl);
      __riscv_vse32_v_f32m2(C2, va2, vl);
      va3 = __riscv_vle32_v_f32m2(C3, vl);
      va3 = __riscv_vfmacc_vf_f32m2(va3, alpha, vres3, vl);
      __riscv_vse32_v_f32m2(C3, va3, vl);
      C0 += vl;
      C1 += vl;
      C2 += vl;
      C3 += vl;
    }

    bb += bk << 2;
    C += ldc << 2;
  }
  if (bn & 2)
  {
    C0 = C;
    C1 = C0 + ldc;
    ptrba = ba;
    for (i = bm; i > 0; i -= vl)
    {
      vl = __riscv_vsetvl_e32m2(i);
      ptrbb = bb;
      vres0 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      vres1 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      for (k = bk / 8; k > 0; k--)
      {
        va0 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        va1 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va0, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va0, vl);
        ptrbb += 2;
        va2 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va1, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va1, vl);
        ptrbb += 2;
        va3 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va2, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va2, vl);
        ptrbb += 2;
        va4 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va3, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va3, vl);
        ptrbb += 2;
        va5 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va4, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va4, vl);
        ptrbb += 2;
        va6 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va5, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va5, vl);
        ptrbb += 2;
        va7 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va6, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va6, vl);
        ptrbb += 2;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va7, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va7, vl);
        ptrbb += 2;
      }

      for (k = bk & 7; k > 0; k--)
      {
        va0 = __riscv_vle32_v_f32m2(ptrba, vl);
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va0, vl);
        vres1 = __riscv_vfmacc_vf_f32m2(vres1, *(ptrbb + 1), va0, vl);
        ptrbb += 2;
        ptrba += vl;
      }

      va0 = __riscv_vle32_v_f32m2(C0, vl);
      va0 = __riscv_vfmacc_vf_f32m2(va0, alpha, vres0, vl);
      __riscv_vse32_v_f32m2(C0, va0, vl);
      va1 = __riscv_vle32_v_f32m2(C1, vl);
      va1 = __riscv_vfmacc_vf_f32m2(va1, alpha, vres1, vl);
      __riscv_vse32_v_f32m2(C1, va1, vl);
      C0 += vl;
      C1 += vl;
    }

    bb += bk << 1;
    C += ldc << 1;
  }
  if (bn & 1)
  {
    C0 = C;
    ptrba = ba;
    for (i = bm; i > 0; i -= vl)
    {
      vl = __riscv_vsetvl_e32m2(i);
      ptrbb = bb;
      vres0 = __riscv_vfmv_v_f_f32m2(0.0, vl);
      for (k = bk / 8; k > 0; k--)
      {
        va0 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        va1 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va0, vl);
        ptrbb += 1;
        va2 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va1, vl);
        ptrbb += 1;
        va3 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va2, vl);
        ptrbb += 1;
        va4 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va3, vl);
        ptrbb += 1;
        va5 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va4, vl);
        ptrbb += 1;
        va6 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va5, vl);
        ptrbb += 1;
        va7 = __riscv_vle32_v_f32m2(ptrba, vl);
        ptrba += vl;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va6, vl);
        ptrbb += 1;
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va7, vl);
        ptrbb += 1;
      }

      for (k = bk & 7; k > 0; k--)
      {
        va0 = __riscv_vle32_v_f32m2(ptrba, vl);
        vres0 = __riscv_vfmacc_vf_f32m2(vres0, *(ptrbb + 0), va0, vl);
        ptrbb += 1;
        ptrba += vl;
      }

      va0 = __riscv_vle32_v_f32m2(C0, vl);
      va0 = __riscv_vfmacc_vf_f32m2(va0, alpha, vres0, vl);
      __riscv_vse32_v_f32m2(C0, va0, vl);
      C0 += vl;
    }

    bb += bk;
    C += ldc;
  }
  return 0;
}


extern "C" __attribute__((noinline)) void kernel(int M,int N,int K,const float*A,const float*B,float*C,float*W){
float*PA=W;float*PB=W+(size_t)M*K;size_t z=0;
for(int i=0;i<M;){size_t vl=__riscv_vsetvl_e32m2(M-i);for(int k=0;k<K;k++)for(size_t r=0;r<vl;r++)PA[z++]=A[(i+r)*K+k];i+=vl;}
z=0;for(int j=0;j<N;){int n=N-j>=8?8:(N-j>=4?4:(N-j>=2?2:1));for(int k=0;k<K;k++)for(int q=0;q<n;q++)PB[z++]=B[k*N+j+q];j+=n;}
for(size_t t=0;t<(size_t)M*N;t++)C[t]=0.0f;source_kernel(M,N,K,1.0f,PA,PB,C,M);
}
