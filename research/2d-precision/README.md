# 2D precision study: FP64, FP32 and TF32 on H100 (pre-registered)

The external review asked for FP32/TF32 measurements: most PINN users train in FP32, the speedups may differ, and the measurements also answer whether FP64 is needed at all. This study compares compiled Q10 (cell A) and native GL8 (cell C) on the 2D Kovasznay objective in three precisions:
- FP64;
- FP32;
- FP32 with TF32 matrix products.

[protocol-2d-precision-20261016.json](protocol-2d-precision-20261016.json) was committed before the batch ran.

## Source change

The pinned 2D source (commit `286d7e394c8b`) accepted only FP64. [src/fp32-dtype.patch](src/fp32-dtype.patch) changes dtype checks only, 28 lines in eight modules:
- the Q10, GL8 and native paths accept FP32 or FP64, consistently across data and parameters;
- the native extension is built for the data dtype, through its existing FP32 code path.

[src/test_precision_cpu.py](src/test_precision_cpu.py) checks the patch on CPU, both locally (Torch 2.14) and in the batch runtime (Torch 2.12), on two formal seeds:
- FP64 gradients are bit-identical to the unpatched source;
- FP32 Q10 and GL8 gradients differ from the FP64 dense-Q10 oracle by 1.5e-7 to 3.2e-7 (relative L2);
- the unpatched source rejects FP32.

## Design

[`src/precision_run.py`](src/precision_run.py) builds the FP64 data and formal initializations unchanged, verifies them against the frozen hashes, then casts them to the arm's dtype.

- **Replay**, at 2,048, 16,384 and 65,536 points, 5 fresh processes per cell:
  - the `bench_gpu` measurement is unchanged;
  - FP64 arms keep the FP64 oracle check;
  - FP32 and TF32 arms report their error against the FP64 dense-Q10 oracle. They fail only on a non-finite value or a relative L2 gradient error of at least 0.1, a sanity gate against broken kernels.
- **Complete solves**, at 2,048 and 65,536 points on the six formal seeds:
  - the solve program, tolerances included, is unchanged;
  - the loss/gradient engine and the optimizer run in the arm's dtype;
  - every stop check and the final acceptance evaluate the parameters cast to FP64 on the FP64 problem, with the unchanged thresholds.
- **Batch.** 162 tasks on one four-GPU node: 72 solves and 90 replays. Both native extensions are prebuilt before the queue starts.

## Registered decisions

[read_precision.py](read_precision.py) applies three rules.
- **Training sufficiency.** A reduced precision is sufficient for a cell and size if 6/6 solves pass FP64 acceptance, and insufficient if at most 4/6 do.
- **Solve speed.** Optimization-only FP64/reduced geometric mean above 1.03, with at least 5 jointly accepted pairs and at least 5 favouring the reduced arm. The same rule decides C/A within each precision.
- **Replay speed.** An effect when the bootstrap interval excludes [0.97, 1.03].

Expectations (not gates):
- FP32 errors near 1e-7, TF32 much larger;
- little solve gain at 2,048 points, where the fixed L-BFGS cost dominates, and a possible gain at 65,536 points;
- FP32 training expected to pass the 1–2% acceptance gates, TF32 uncertain.

## Status

Pre-registered; the GPU batch is pending.
