# Generate, migrate and validate the registered RVV kernels

This first-stage public implementation connects source analysis, candidate
construction, instruction checks, numerical validation, measured selection and
independent reports. The packaged contracts are six FP32 variants at VLEN=256,
with positive dimensions/increments and nonaliasing inputs, outputs and scratch.
OpenBLAS itself does not need to be linked.

## Install

From the repository root, with Python 3.10 or newer:

```sh
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
```

On a physical RVV Linux machine, install GCC/G++ supporting the RVV 1.0 intrinsic
API (the paper used GCC 13.2) and matching binutils `objdump`. The native runner
checks the host architecture; the benchmark checks VLEN=256. Compilation and
measurement use the same machine in this public search runner.

## Regenerate the complete candidate pool on any analysis host

```sh
python3 scripts/generate_candidates.py --output outputs/generated
```

This reads the six files in `experiments/openblas/kernels/`, recovers their ASTs
and dependence scopes, and constructs the 109 registered candidates. It does
not copy the historical files in `experiments/openblas/candidates/`. Those files
serve as independent regression fixtures. The tests compare every regenerated
candidate with its historical counterpart after removing comments, includes and
whitespace, while retaining operation, type, index and literal tokens.

The generated directory contains:

| File | Meaning |
| --- | --- |
| `sources/*.cpp` | Complete candidate sources, helpers and Level 3 packing adapters |
| `analysis/*.json` | CFG, reaching-definition/dependence edges, seeds, conservative scopes and boundaries |
| `catalog.json` | Source/output hashes, construction profile, organization, LMUL and containment records |
| `rejections.json` | Failed schema/containment checks, including their reasons |
| `construction.json` | Construction time, separate from the search budget |

Constructed candidates are not yet qualified. Scope closure is conservative,
and preservation of statement text outside a scope is not an equivalence proof.
The rules are implemented for these recovered source schemas; new kernels may
need new guards and rules.

## Native migration, search and independent reports

Start with one registered GEMM input and its m1 target profile:

```sh
python3 scripts/run_migration.py \
  --case gemm_01 --profile P1 --mode independent \
  --output outputs/gemm_independent
```

This performs independent fixed-organization and full-mapping searches with equal
120-second budgets. Both arms use the same source-derived pool and numerical
policy, with separate candidate compilation/measurement. Arm order is shuffled
deterministically. The fixed arm retains organization while allowing the eligible
legalization and vector parameters. The full arm also admits regrouping.

Selections are frozen before five held-out report processes with distinct seeds.
Report order is interleaved across selected executables. A failing frozen candidate
is recorded as a failure; the report does not select a replacement.

The same command supports Level 2 variants, for example:

```sh
python3 scripts/run_migration.py \
  --variant gemv_t --profile P3 --mode pool \
  --output outputs/gemv_t_pool
```

`pool` mode runs one full search and compares retained organization and fixed LMUL
within its observed eligible pool. These are **shared-pool controls**, not
independent equal-budget comparisons. Fixed LMUL means m1 under P1/P4 and m8
otherwise, with organization reselected inside that restricted pool.

`independent` mode is available for both levels. New Level 2 independent runs are
new experiments; do not describe them as the paper's existing pool measurements.

Omit `--case` and `--variant` to use all 90 registered inputs. Omit `--profile` to
consider all five profiles. Each input/arm has one shared profile-pool budget,
not a separate budget for every profile. A full campaign can take hours.

Useful options:

```sh
# Reuse a previously generated pool, still compiling candidates in each search.
python3 scripts/run_migration.py --candidate-dir outputs/generated \
  --case trmm_01 --profile P1 --mode independent \
  --cpu 0 --cxx g++ --objdump objdump --output outputs/trmm_independent

# Host-only construction and job inspection. This creates no measurements.
python3 scripts/run_migration.py --case gemm_01 --profile P1 \
  --mode independent --plan-only --output outputs/inspect_plan
```

Installed equivalents are `gapmigration-generate` and `gapmigration-run`.
Run from the repository root or supply explicit benchmark/config paths.

## Budgets and qualification

`configs/search.json` records budget, timeout and seed settings. `--budget` can
override search seconds for a new experiment. Compilation, object inspection,
linking, numerical/interface validation, timing and selection count against the
search deadline. Source analysis, candidate construction and common benchmark
driver compilation are separately recorded and excluded. This is not a unified
end-to-end migration-time comparison.

Every measured candidate must first pass object admission for a profile under
which it was constructed. The runner checks the complete candidate object,
including helpers and Level 3 wrappers. Unresolved external calls cause rejection.
The benchmark driver and system runtime are outside that kernel admission scope.

Identical candidate bytes share one trial inside a search, while retaining their
construction-profile memberships. Results finishing after the deadline cannot
update the incumbent. Failures, timeouts and empty eligible sets have no assigned
latency. The overall exit code is nonzero if a requested method/profile has no
qualified held-out result; partial successes remain in the output.

The benchmark checks FP64-reference error, input integrity, output sentinels and
repeated-call behavior. Tolerances, argument contracts and batching are described
in the [benchmark guide](../experiments/openblas/README.md).

## Inspect and reuse outputs

| Location | Contents |
| --- | --- |
| `plan.json`, `environment.json` | Effective protocol, inputs, tool versions, flags and selected CPU |
| `drivers/` | Shared driver objects and compilation logs |
| `cases/<input>/<arm>/trials.jsonl` | Every attempted candidate, stage, command, failures and measurements |
| `cases/<input>/<arm>/<candidate>/disassembly.txt` | Inspected candidate object disassembly |
| `cases/<input>/<arm>/frozen.json` | Committed selection and explicitly labeled pool controls |
| `cases/<input>/reports/` | Independent report results, mappings and shared-observation records |
| `cases/<input>/selected/<method>/<profile>/` | Qualified complete source, executable and benchmark-compatible manifest |
| `summary.json` | Per-input/profile/method qualification and median report latency |

Keep the entire output directory. Existing output directories are not overwritten.
Fresh measurement processes that share one executable are recorded together so
profile aliases cannot be mistaken for independent replications.

To cross-compile a generated candidate on x86-64, use the existing standalone
builder, then transfer its output and run it on physical RVV hardware:

```sh
python3 experiments/openblas/run.py build --variant gemm \
  --source outputs/generated/sources/P1_gemm_L1_group4.cpp \
  --cxx riscv64-linux-gnu-g++ --output outputs/cross_gemm
# On the RVV machine after transfer:
python3 experiments/openblas/run.py run --build outputs/cross_gemm
```

This split path validates that candidate; it does not provide a remote equal-budget
search protocol. Compiler/sysroot and target runtime libraries must be compatible.

## Optional packing-loop rule

The original source-derived packing rule is also available:

```sh
python3 scripts/generate_packing.py --profile P3 --lmul 1 \
  --mode tile --output outputs/packing_tile
```

It checks the recovered panel-loop relation and preserves the surrounding AST,
with a fixed 16-column layout and 32-row tile. Its input is the preprocessed
`auxiliary/packing.cpp`. This auxiliary kernel is not a seventh paper variant,
and the generated rule output is marked unqualified until separately compiled,
audited and tested. The normal six-variant pipeline already includes its required
Level 3 call adapters; this optional packing rule is a separate analysis tool.

## Implementation provenance and validation status

| Public implementation | Historical functionality ported |
| --- | --- |
| `mapping/level2.py` | `generate_level2.py`, `generate_level2_hierarchy.py` |
| `mapping/materialize.py` | `generate_materialize.py` |
| `mapping/level3.py`, `mapping/common.py` | GEMM portion of `scoped_construct.py`, `generate_trmm.py`, relevant `generate_campaign.py` adapters |
| `scoped_packing.py` | Checked packing-loop scope expansion and rewrite |
| `search.py`, `runtime.py`, `migration.py` | Local compile/audit/search/report functionality of formal comparison, independent-arm and phase-runner scripts |

Private machine paths and unrelated baseline integrations were removed. The new
runner does not claim byte-for-byte replay of the historical orchestration or
reproduction of published speedups. It uses the original rule outputs and exposes
the effective settings for fresh experiments. TVM and LLM/IntrinTrans campaigns
are outside this first-stage release.

Host tests verify all 109 regenerated candidate token streams, rule rejections,
deadline behavior, selection eligibility, held-out failure handling and pipeline
wiring with mocked hardware. This release has **not yet been rerun on physical
RVV hardware**. The tests do not supply new numerical qualification or performance
results for the native runner.
