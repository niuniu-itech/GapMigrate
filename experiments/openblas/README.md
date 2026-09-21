# Standalone OpenBLAS RVV kernels and benchmarks

This directory contains the six FP32 kernel variants used in the paper: GEMM
and right-upper, non-transposed, non-unit-diagonal TRMM (BLAS Level 3), plus
GEMV-N, GEMV-T, SYMV-L and SYMV-U (BLAS Level 2).

## Files and provenance

- `upstream/`: six **unmodified, macro-bearing** OpenBLAS C files, their
  BSD license and SHA-256 / immutable Git blob identifiers in `provenance.json`.
  GEMM comes from the recorded v0.3.30 snapshot; the other files come from the
  recorded development snapshot. Do not describe all six as one release.
- `include/common.h`: a minimal **benchmark-only compatibility header** selecting
  FP32 and `CNAME=source_kernel`. It is not the full OpenBLAS header.
  TRMM additionally defines `TRMMKERNEL`; `DOUBLE`, `LEFT`, and `TRANSA` stay
  undefined. Original macro branches remain in the source files.
- `benchmark/`: FP64-reference checks, input-integrity and output-sentinel checks,
  and latency measurement. The Level 3 wrapper packs source-layout inputs and
  initializes outputs inside the timed call.
- `configs/inputs.json`: all 90 registered input configurations, 15 per variant,
  including dimensions and increments. `profiles.json` records P0–P4.
- `candidates/`: 109 historical source-derived candidate snapshots. Each includes
  helpers and, for Level 3, its matching packing wrapper. The catalog records
  variant, intended profile, LMUL, organization and checksums. These are pool
  artifacts, **not a claim that every artifact is qualified or was selected**.
  The historical checksum excludes the restored copyright notice; the published
  checksum includes it. No computation was changed when packaging the snapshots.

The drivers are extracted from the report harnesses, with argument and target
checks added and unrelated operator branches removed. This package enables fresh
correctness checks; it does not recreate the full historical search campaign.

## Native build and execution

Use Python 3 and RISC-V GCC/G++ with RVV 1.0 intrinsics (the paper used GCC 13.2).
Run on physical RVV Linux with **VLEN=256**. The harness rejects other VLENs because
these packaged candidate contracts were evaluated at 256 bits.

From the repository root:

```sh
python3 experiments/openblas/run.py verify
python3 experiments/openblas/run.py list

# Original macro-bearing GEMM plus source-layout packing adapter.
python3 experiments/openblas/run.py build --variant gemm \
  --cxx g++ --output outputs/openblas/gemm_source
python3 experiments/openblas/run.py run --build outputs/openblas/gemm_source

# Historical migrated m1/group-4 implementation, with its own matching adapter.
python3 experiments/openblas/run.py build --candidate P1_gemm_L1_group4 \
  --cxx g++ --output outputs/openblas/gemm_migrated
python3 experiments/openblas/run.py run --build outputs/openblas/gemm_migrated

# TRMM and a strided matrix-vector variant use their corresponding drivers.
python3 experiments/openblas/run.py build --variant trmm \
  --output outputs/openblas/trmm_source
python3 experiments/openblas/run.py run --build outputs/openblas/trmm_source
python3 experiments/openblas/run.py build --candidate gemv_t_P4_m1_chunk_reduce \
  --output outputs/openblas/gemv_t_migrated
python3 experiments/openblas/run.py run --build outputs/openblas/gemv_t_migrated
```

Other original variants are `gemv_n`, `gemv_t`, `symv_L`, and `symv_U`.
Use `--smoke` for only the first registered input. The default is all 15 inputs
with three independent executable invocations each. Use `--repeats 5` for five
invocations, and optionally `--cpu N` to pin execution to an available core.
Do not run timing jobs concurrently on that core.

## Cross-compilation on an x86-64 Linux host

```sh
python3 experiments/openblas/run.py build --variant gemv_n \
  --cxx riscv64-linux-gnu-g++ --output outputs/openblas/gemv_n_source
```

Transfer that output directory and this repository to the physical RVV machine,
then run the `run` command there. The build contains a dynamically linked binary,
so the target needs compatible Linux C/C++ runtime libraries. No benchmark binary
is executed on the x86-64 host. Compiler flags and all actual build/link commands
are recorded in `manifest.json` and `build_commands.json`.

## Validate another migrated implementation

```sh
python3 experiments/openblas/run.py build --variant gemm \
  --source path/to/complete_migrated.cpp --output outputs/openblas/my_gemm
python3 experiments/openblas/run.py run --build outputs/openblas/my_gemm
```

The source must include all its helpers and meet the corresponding ABI:

```cpp
// GEMM/TRMM: A[M*K], B[K*N] row-major; C[M*N] column-major.
// W has M*K + K*N + M*N + 4096 float elements. Inputs do not alias outputs.
// Compute C = A*B, not C += A*B. TRMM has K=N and an upper-triangular B.
extern "C" void kernel(int M, int N, int K, const float* A,
                       const float* B, float* C, float* W);

// GEMV-N/T: column-major A with lda=M+3. Compute y += alpha*op(A)*x.
int source_kernel(long m, long n, long dummy, float alpha, float* A,
                  long lda, float* x, long inc_x, float* y, long inc_y,
                  float* work);

// SYMV-L/U: n=m; use only the indicated triangle. Compute y += alpha*A*x.
int source_kernel(long m, long n, float alpha, float* A, long lda,
                  float* x, long inc_x, float* y, long inc_y, float* work);
```

Do not attach the original m2 packing wrapper to an m1 Level 3 kernel: its packed
layout must match its vector configuration. Historical Level 3 snapshots already
contain the appropriate complete-call wrapper. Level 2 signatures use C++ linkage
in the existing harness, so the runner compiles all kernels as C++.

## Results and interpretation

`results.jsonl` records every invocation, input ID, seed, exit status, numerical
error and timing samples, including failures and timeouts. The process returns
nonzero if any check fails and refuses to overwrite existing results.

- Level 2 checks use `2e-5 + 2e-4*abs(reference)` tolerance, seven batches of
  20 calls, and a post-batch check of the accumulated output.
- Level 3 checks use `2e-4 + 2e-4*abs(reference)` tolerance, seven batches and
  1–2000 calls per batch calibrated toward 3 ms. Packing and output initialization
  are included. Caller-owned workspace allocation occurs before timing.
- Both check input integrity and inactive output/sentinel values. These finite
  tests do not prove general equivalence or cover every OpenBLAS macro setting.
- Numerical passing is separate from target-profile admission. `run.py` does
  **not** label a passing candidate P1–P4 qualified. Inspect kernel objects and
  helpers with the repository's admission tools before making that claim.
- No newly measured RVV result is bundled with this packaging change. Run the
  commands above to generate results for your hardware and compiler.

OpenBLAS notices are retained in original and derived files. This does not change
the repository's undecided license for original GapMigrate code.
