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

## Files

- `make_lbfgs_plan.py` builds the frozen plan [plan-2d-lbfgs-gh200-20261023.json](plan-2d-lbfgs-gh200-20261023.json) (36 solves + 4 drift repeats).
- `run_lbfgs_study.py` is the cluster-side driver; `entry_lbfgs_study.sh` and `start_lbfgs_study.sbatch` start it inside its own allocation.
- `read_lbfgs_study.py` is the reader and registered analysis.
- `test_lbfgs_study.py` (CPU, in CI): the rewrite against the installed torch and against a foreign source, the plan builder, the reader's decisions on synthetic rows, and the driver's control flow in self-test mode.

All numbers are GH200 numbers; nothing is pooled with the H100 or H20 campaigns.
