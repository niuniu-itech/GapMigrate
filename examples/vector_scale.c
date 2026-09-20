/* Synthetic example authored for this repository, not an upstream BLAS kernel. */
#include <stddef.h>
#include <riscv_vector.h>

void source_kernel(float *a, float *out, int n, float alpha) {
    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t x = __riscv_vle32_v_f32m2(a + i, vl);
        vfloat32m2_t y = __riscv_vfmul_vf_f32m2(x, alpha, vl);
        __riscv_vse32_v_f32m2(out + i, y, vl);
        i += vl;
    }
}
