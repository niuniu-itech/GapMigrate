# Third-party components

This repository does not bundle compiler, TVM or OpenBLAS source/binaries. Dependencies must be installed separately. Their licenses remain independent of this project's currently undecided license.

| Component | Use | Upstream license / source |
| --- | --- | --- |
| LLVM / Clang | Optional external AST inspection and compilation | Apache-2.0 WITH LLVM-exception; https://llvm.org/docs/DeveloperPolicy.html#license |
| Apache TVM | Research dependency; no tuning implementation bundled in this package | Apache-2.0; https://github.com/apache/tvm/blob/main/LICENSE |
| pycparser | Existing structured-C AST/CFG prototype | BSD-3-Clause; https://github.com/eliben/pycparser/blob/main/LICENSE |
| pcpp | Frontend preprocessing | BSD-3-Clause; https://github.com/ned14/pcpp/blob/master/LICENSE |
| OpenBLAS | User-supplied isolated benchmark kernels | BSD-3-Clause; https://github.com/OpenMathLib/OpenBLAS/blob/develop/LICENSE |

Check the exact installed versions and their additional third-party notices before redistributing dependencies. If upstream source is copied or modified, retain the applicable copyright, LICENSE and NOTICE material and identify changes. Merely invoking these tools does not select a license for original GapMigrate code.

## Project license status

No open-source license has been selected by the authors. No top-level LICENSE is provided. Public visibility alone is not a grant of broad reuse rights. Third-party rights are not modified by this status.

## Branding

The GapMigrate icon and wordmark in assets/brand are original vector artwork for this repository. They do not imply endorsement by LLVM, Apache TVM or OpenBLAS. No third-party font files are bundled.
