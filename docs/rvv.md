# RISC-V GCC and physical RVV execution

The public path uses RVV 1.0, `riscv_vector.h` and `__riscv_` intrinsics.
Use a GCC build supporting these intrinsics and the target machine's ABI.
The capability profiles restrict generated instructions in software.

## Generate on the analysis host

```sh
gapmigration candidates examples/vector_scale.c --profile P1 --output outputs/candidates
```

## Compile and run on an RVV Linux machine

Copy the repository and generated candidates to the physical machine. From the
repository root, compile the example and its migrated candidate separately.

```sh
mkdir -p outputs
gcc -O2 -march=rv64gcv -mabi=lp64d -fno-tree-vectorize -fno-fast-math \
  examples/vector_scale.c examples/check_scale.c -lm -o outputs/source_check
gcc -O2 -march=rv64gcv -mabi=lp64d -fno-tree-vectorize -fno-fast-math \
  outputs/candidates/g8_m1.c examples/check_scale.c -lm -o outputs/migrated_check
timeout 30s ./outputs/source_check
timeout 30s ./outputs/migrated_check
```

On a cross-compilation host, replace `gcc` with `riscv64-linux-gnu-gcc`, use a
sysroot matching the physical machine, and copy the resulting executables there
before running them. GCC, the sysroot and binaries are not bundled.

The smoke test checks tail lengths and output bounds against a scalar reference.
It is not the paper benchmark and does not measure performance. A timeout is an
incomplete run, not a measured latency. Record compiler version, flags, CPU,
VLEN and numerical results when conducting an experiment. Instruction admission
and repeated full-call latency measurements are additional required steps before
reporting qualified performance.
