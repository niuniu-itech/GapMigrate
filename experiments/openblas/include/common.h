// Minimal standalone benchmark configuration, not OpenBLAS's full common.h.
#pragma once
#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>
#define FLOAT float
#define IFLOAT float
#define BLASLONG long
#define CNAME source_kernel
