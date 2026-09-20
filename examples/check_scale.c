/* Original numerical smoke test for vector_scale.c and its migrated candidate. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void source_kernel(float *a, float *out, int n, float alpha);

int main(void) {
    const int sizes[] = {0, 1, 7, 31, 95, 191, 512, 2049};
    for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        int n = sizes[s];
        float *a = malloc((n + 2) * sizeof(*a));
        float *out = malloc((n + 2) * sizeof(*out));
        if (!a || !out) { free(a); free(out); return 2; }
        for (int i = 0; i < n + 2; ++i) {
            a[i] = (float)(i % 29 - 14) / 8.0f;
            out[i] = 12345.0f;
        }
        source_kernel(a + 1, out + 1, n, 1.25f);
        int failed = out[0] != 12345.0f || out[n + 1] != 12345.0f;
        for (int i = 1; i <= n; ++i) {
            double expected = (double)a[i] * 1.25;
            if (!isfinite(out[i]) || fabs(out[i] - expected) > 1e-6)
                failed = 1;
        }
        free(a); free(out);
        if (failed) { fprintf(stderr, "FAIL n=%d\n", n); return 1; }
        printf("PASS n=%d\n", n);
    }
    return 0;
}
