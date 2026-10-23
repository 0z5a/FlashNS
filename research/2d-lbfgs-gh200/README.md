# 2D L-BFGS execution study on GH200 (pre-registered)

## Question

On H100, the [solve-scale study](../2d-solve-scale/README.md) found that at the formal 2,048 interior points a faster gradient does not shorten a solve: the L-BFGS phase costs 5.9–7.1 ms per evaluation in every cell, against 0.17–0.28 ms of gradient replay. Post hoc, the archived solves put 93% of the 2,048-point optimization time into that phase, outside the gradient.

The cause is in the optimizer, not the model. `torch.optim.LBFGS` keeps the two-loop coefficients as device tensors and passes each one as `alpha=`, which converts it to a host number: about 109 device synchronizations and 555 small operators per iteration (history 50). This study asks whether replacing only the two-loop recursion by a device-side version, replayed as a CUDA Graph, shortens complete solves **without changing them**.

[protocol-2d-lbfgs-gh200-20261023.json](protocol-2d-lbfgs-gh200-20261023.json) was committed before the GPU batch ran. Its v1 pinned the x86 initialization hashes of the H100 studies; the first attempt was refused at admission (before any GPU work) because `problem.parameters(seed)` draws `torch.randn` on the CPU and torch's CPU normal sampler is not bitwise identical between x86-64 and aarch64 (the GH200 node's Grace CPU), while the `torch.rand` collocation data are. v2 pins the aarch64 draws from a dedicated check job and keeps the x86 hashes for reference; the GH200 and H100 studies therefore share the data but not the initial parameters bitwise. The second attempt passed admission and failed in the prebuild: the wrapper read the `torch.optim.lbfgs` module through an attribute that `torch.optim` deletes; the wrapper now binds the module at import (a CPU test imports it with the 2D modules stubbed). No scored task had run.

## Cells

| Cell | Optimizer |
| --- | --- |
| stock | `torch.optim.LBFGS` as installed (torch 2.14.1) |
| light | `lbfgs_light.LightLBFGS(mode="graph", form="fma")` |

`LightLBFGS` ([src/lbfgs_light.py](src/lbfgs_light.py)) rewrites the installed `LBFGS.step` source and replaces exactly the two-loop block; the memory update, the strong-Wolfe line search, the stop tests and the state carried across `step()` calls are the runtime's own code, and a torch whose block differs is refused. In the replacement the coefficients stay on the device: `torch.dot` as before, then one Triton kernel per history entry computes `al = dot * ro` and the update `q <- q - al * y` with a single rounding (fma), which reproduces `add_(y, alpha=-al)` bitwise (development check: 0 of 909,400 elements differ; the round-twice form differs in 25%). Once the history is full, the rows are mirrored into a ring buffer of padded, aligned copies and one captured CUDA Graph per ring position replays the whole two-loop.

Everything else is the unchanged 2D solve: native GL8 (`gl8_native`) in FP64, `solve.py`'s optimizer program (4,000 Adam updates, strong-Wolfe L-BFGS in blocks of at most 20 iterations / 25 evaluations, history 50, 500-block extension) and acceptance, run through [src/light_solve.py](src/light_solve.py) in the pinned 2D source tree (commit `286d7e394c8b`, same as the H100 studies) exactly as the solve-scale wrapper did.

## Design

- **Sizes and seeds.** 2,048, 16,384 and 65,536 interior points; the six formal initializations. 36 solves, one fresh process each, in one pre-randomized order over the four GPUs of one GH200 node.
- **Prebuild.** Before the queue, on lane 0: the native extension, then one unscored light solve at 2,048 points, so every timed task starts with the native extension and the Triton kernels compiled.
- **Drift.** The first four tasks of the randomized order are repeated at the end.
- **Admission.** Manifest, plan and protocol hashes; the pinned source commit with exactly the two new files; torch, Triton and the `torch.optim.lbfgs` module hash; the frozen initialization and data hashes of all three sizes; four idle GH200s with one exclusive lock file each.
- **Registered statistics**, per size: pairs with identical final parameter SHA-256, evaluation count, L-BFGS iteration count and acceptance; paired geometric mean (range) of stock over light optimization-only seconds and of setup plus optimization over jointly accepted pairs; pairs favouring light; phase costs per evaluation.
- **Decision**, per size: `EXECUTION_GAIN_IDENTICAL` if all six pairs are identical and jointly accepted, the geometric mean exceeds 1.03 and at least 5 of 6 pairs favour light; `IDENTICAL_NO_GAIN` otherwise when identical; `TRAJECTORY_DIFFERS` if any pair is not identical (no execution claim at that size; the seeds are listed); `NOT_DECIDABLE` below 5 jointly accepted pairs. A drift repeat that is not identical to its first run, or whose optimization time is outside 5%, marks the batch's timing `NOT_VALID`; identity results stand.
- **Predictions** (expectations, not gates), from three development pairs on GH200 and the H100 phase model: stock over light optimization 2.4–2.9 at 2,048, 2.0–2.6 at 16,384 and 1.4–1.6 at 65,536 points; all 36 pairs identical.

## Results

The batch ran once on one four-GPU GH200 node (7.1 minutes, four lanes). Admission verified the manifest (14 files), the plan and protocol hashes, the pinned source commit with exactly the two new files, torch 2.14.1 / Triton 3.8.0 and the `torch.optim.lbfgs` module hash, the six aarch64 initialization hashes and the fifteen data hashes; the prebuild compiled the native extension and ran one unscored light solve. All 40 tasks completed and every solve passed acceptance; the archive (`321d7f77…`, 162 members) was re-read and verified.

| Points | Identical pairs | Optimization stock/light GM (range) | Setup + opt. GM | Favour light | Adam ms/eval stock / light | L-BFGS ms/eval stock / light |
| ---: | ---: | --- | ---: | ---: | --- | --- |
| 2,048 | 6/6 | **2.21** (1.96–2.38) | 2.17 | 6/6 | 0.22 / 0.22 | 6.64 / 2.93 |
| 16,384 | 6/6 | **1.82** (1.68–2.15) | 1.81 | 6/6 | 0.70 / 0.70 | 6.97 / 3.43 |
| 65,536 | 6/6 | **1.48** (1.33–1.59) | 1.47 | 6/6 | 2.24 / 2.24 | 8.49 / 4.91 |

- **Identity.** All 36 pairs are identical: same final parameter SHA-256, evaluation count, L-BFGS iteration count and acceptance. The light optimizer changes execution only.
- **Speed.** Stock over light optimization is 2.21 (1.96–2.38) at 2,048 points, 1.82 (1.68–2.15) at 16,384 and 1.48 (1.33–1.59) at 65,536, with all six pairs favouring light at every size. The Adam phase is unchanged (0.22 / 0.70 / 2.24 ms per evaluation in both cells); the L-BFGS phase falls from 6.6 / 7.0 / 8.5 ms to 2.9 / 3.4 / 4.9 ms per evaluation. Setup is 0.4–0.7 s in both cells (one outlier each, 1.3 s).
- **Registered decision.** `EXECUTION_GAIN_IDENTICAL` at every size by identity, geometric mean and favour counts, **but the timing is `NOT_VALID` by the registered drift rule**: the four drift repeats are all identical to their first runs, three are within 5% in optimization time (1.005, 0.976, 0.995), and one stock repeat at 16,384 points ran 5.7% faster the second time (0.943). The decision is therefore reported with the `(NOT_VALID timing)` label the protocol prescribes. The variance is on the stock side, whose L-BFGS phase is host-bound and ran beside three other lanes on the same node; the light repeat was within 0.5%. A re-run with one lane per node would be needed to validate the timing under the registered rule; the identity result stands.
- **Predictions.** 2.21 at 2,048 points is below the predicted 2.4–2.9 (from three development pairs run one at a time); 1.82 at 16,384 and 1.48 at 65,536 are inside the predicted 2.0–2.6 and 1.4–1.6.
- **Coverage.** In every light solve the two-loop ran eagerly for the 49 iterations before the history was full and as a graph replay afterwards (one ring refill, no history-unchanged two-loops).

Drift repeats (optimization time second/first): light-n65536-s862102 1.005; stock-n16384-s862105 0.943; stock-n2048-s862106 0.976; stock-n2048-s862104 0.995.

Files:
- [lbfgs-result-20261023.json](lbfgs-result-20261023.json): registered reader output (work-directory prefix and host identifiers normalized).
- [driver-state-20261023.json](driver-state-20261023.json): admission checks, lanes, per-task lane/exit/time (command lines removed).
- [release-source-map-20261023.json](release-source-map-20261023.json): byte identity of every published file with its executed copy, the executed copies of the two new source files, the archive digest and the two earlier attempts.

## Files

- `make_lbfgs_plan.py` builds the frozen plan [plan-2d-lbfgs-gh200-20261023.json](plan-2d-lbfgs-gh200-20261023.json) (36 solves + 4 drift repeats).
- `run_lbfgs_study.py` is the cluster-side driver; `entry_lbfgs_study.sh` and `start_lbfgs_study.sbatch` start it inside its own allocation.
- `read_lbfgs_study.py` is the reader and registered analysis.
- `test_lbfgs_study.py` (CPU, in CI): the rewrite against the installed torch and against a foreign source, the plan builder, the reader's decisions on synthetic rows, and the driver's control flow in self-test mode.

All numbers are GH200 numbers; nothing is pooled with the H100 or H20 campaigns.
