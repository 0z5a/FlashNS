# Local source synchronization, September 13, 2026

The comparison used GitHub `main` at `255058062000dd500b0c86e73ceec93acb177bb5` (71 files). The local `flashns` directory was a source workspace without its own Git history, so differences were checked against Git blob contents rather than inferred from commit counts.

## What differed

| Local source set | Identical to the 71 GitHub files | Different | Absent locally |
| --- | ---: | ---: | ---: |
| Main local `flashns` workspace | 37 | 10 | 24 |
| Frozen SM120 `first-vjp-wgrad-ready/flashns` package | 57 | 7 | 7 |

GitHub already contained the G1/G2 kernels, shared coordinate headers, host references, tests, and validation runners that were absent from the main local directory. The frozen package also predates some published runtime compatibility and documentation adjustments. Replacing GitHub wholesale with either directory would regress that release. This supplement retains the published files and adds missing material; all **60 entries in the original [tested-source manifest](tested-source-sha256.json) still match byte for byte**.

The supplemental material includes:

- H100 native, Graph, cuBLASLt, cuEquivariance, PhysicsNeMo, and MathDx experiment drivers; Hopper kernels and their validation; earlier A4000 data-parallel and CUDA fusion experiments.
- The complete eight-backend Kovasznay solver, frozen protocols, per-run JSON, original budget failures, uniform budget extension, and summary/acceptance evidence.
- Local affine-wave, stable Euler spline, symbolic and high-precision references, formal-source adapters, source registries, and their CPU tests.
- Missing CUDA/HOST validation and packaging entry points, the original v8 reference package, and selected historical reports, raw JSON, logs, and plots.

Environments, downloaded third-party archives, generated binaries, large input arrays, checkpoints, and machine-specific remote orchestration directories remain outside the source release. Historical JSON retains the original source and binary hashes and may refer to artifacts held in the original local archive. The linked reports state these boundaries; this supplement is not a complete offline image of those machines.

## Description changes

The English and Chinese READMEs now lead with the measured improvements over existing library paths. [The comparison page](library-comparison.md) separates complete-solver results from fixed-input gradient replay and explains the smaller incremental gain over the existing library-GEMM jet baseline. SM120's inconclusive G1/G2 speed results remain visible, and ordinary Torch dispatch remains the default.

Package metadata now describes the broader library comparisons and exposes the restored `flashns` CLI. The empty `euler` extra preserves compatibility with the historical runner because Torch is already a required dependency. The dependency lock is generated for the published package name; the old local `flashns-repro` metadata is not copied over the existing `flashns` package.

## Verification for this publication

- **235 CPU tests passed, no skips**, using the pinned upstream Euler evaluator from the existing local source archive. Without that optional downloaded reference, the source-only checkout has 233 passes and two explicit skips.
- **27/27 HOST C++ residual/seed cases passed** against the independent tensor reference.
- **23/23 v8 reference/integration checks passed**, including the original shared stable-jet header compiled as HOST C++.
- The original 60 tested-source hashes match. Shell syntax, repository documentation links, source whitespace, and the computed comparison values were checked.
- GPU benchmarks and Sanitizers were not rerun during this publication. GPU statements refer to the linked historical evidence and its stated coverage.

See [publication validation](publication-validation-20260913.json) and [supplemental provenance](supplement-source-sha256.json). The provenance records both the original local file hashes and their published hashes; documentation edits for portable links are distinguishable from unchanged source/evidence bytes.
