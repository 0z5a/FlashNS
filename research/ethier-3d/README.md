# Ethier–Steinman transient 3D PINN solves (protocol v4)

This directory holds the frozen source, data and protocol of the first accepted three-dimensional FlashNS solve comparison, together with its audited result. The problem is the zero-force Ethier–Steinman Beltrami flow (`a=0.5`, `d=1`, `nu=1`) on the cube `[-0.5,0.5]^3` for `t in [0,0.25]`, with pressure gauge `p(x,t)-p(0,0,0,t)`. The source is Ethier and Steinman (1994), Eq. 15, also recorded in the Cardinal verification equations. At four points, evaluated at 80 and 160 decimal digits, the analytic field satisfies all 16 momentum, divergence and residual-gradient observations to below 1e-65, and the FP64 implementation agrees with the high-precision values to 2e-15 ([source-verification](source-verification/); its recorded `source_sha256` matches `src/ethier_source.py`, and the historical verifier imports it from its own directory and needs mpmath).

A `4-64-64-4` tanh network (4,740 parameters) predicts velocity and pressure directly. The FP64 objective is the mean of the squared momentum/divergence residual plus `0.1` times its squared spatial gradient at 512 interior points, plus velocity terms at 384 boundary and 256 initial points (weight 10) and a pressure-gauge term at 64 points (weight 1). Acceptance requires all eight gates on separate validation points: normalized momentum, divergence, boundary, initial and gauge RMS at most 0.01; normalized residual-gradient RMS at most 0.02; velocity relative L2 at most 0.02; pressure relative L2 at most 0.04. A final independent scalar-coordinate-AD evaluation must confirm acceptance.

## Arms

| Arm | Derivative state per hidden scalar | Parameter gradient |
| --- | --- | --- |
| `c17_graph` (FlashNS) | C17: value, spatial gradient (3), packed spatial Hessian (6), gradient of the spatial Laplacian (3), `d/dt` (1), spatial gradient of `d/dt` (3) | Explicit transpose, including the fourth tanh derivative |
| `a24_graph` (internal reference) | A24: complete spatial third-order jet (20) plus `d/dt` and its spatial gradient | PyTorch autograd |
| `naive_q35_graph` (ablation) | All 35 monomials of `(x,y,z,t)` through degree three | PyTorch autograd |

All arms share data, initial parameter bytes, objective, optimizer program, CUDA Graph replay and gates. These are internal baselines; no external 3D library is compared here.

## Protocol history

- **v2**, development seed 642201 on an NVIDIA H20, 100 L-BFGS blocks: all three arms passed seven gates and failed only pressure relative L2 (6.95%, 6.95%, 7.04% against 4%). Removing each model's own origin offset did not change the error, and pressure error was still decreasing at block 100.
- **v3** changes only `optimizer.lbfgs_blocks` from 100 to 400. A CPU development diagnostic on seed 642201 accepted C17 at block 195 (pressure 0.03996). Seed 642201 is therefore development-only.
- **v4** ([src/protocol-v4.json](src/protocol-v4.json)) freezes six formal seeds 642202–642207. Seeds 642204–642207 are new draws from the original initializer; seeds 642201–642203 regenerate bit-identically. Data arrays from v1 are unchanged.

Optimizer: 2,000 Adam updates (learning rate 1e-3), then up to 400 strong-Wolfe L-BFGS blocks of at most 20 iterations, nominally 25 evaluations, history 50. Stop checks run every 200 Adam updates and every 5 L-BFGS blocks.

## Execution

`src/run_v4.py` runs the batch once. It checks the job, node, source manifest, original per-GPU lock inodes and idle GPUs, then holds two GPU lane locks. Each lane runs three seeds; the three arms of a seed run on the same GPU, and the six arm orders are the six permutations, so every arm occupies every position twice. Every case is a fresh process with no retries. After all children are joined the driver releases the locks, writes a raw-file manifest, and runs `src/audit.py`. This is an independent NumPy reader that rechecks every qualification gradient, the final fields against a separate 35-monomial Taylor oracle, all metrics and the acceptance decisions. Finally the driver writes and verifies a complete archive.

`release-source-map.json` lists the executed hash of every source file. The public copies differ only in a site-specific activation path (`entry.sh`, now `FLASHNS_ACTIVATE`) and a hardware site label (`protocol-v4.json`).

## Result

On 2026-10-14 all 18 planned solves (6 formal seeds × 3 arms) ran on two NVIDIA H100 GPUs (FP64, TF32 disabled, Torch 2.4.0a0, NumPy 1.26.4), and all 18 passed every gate in both the stop check and the independent final check. Times are setup plus optimization. The ratio is the geometric mean of six per-seed baseline/C17 ratios, not a ratio of medians.

| Arm | Accepted | Median s | Median evaluations | Paired ratio vs C17 | Range |
| --- | ---: | ---: | ---: | ---: | --- |
| FlashNS C17 | 6/6 | 55.7 | 6,676 | 1.000 | – |
| A24 reference | 6/6 | 210.1 | 7,220 | 3.602 | 3.353–4.284 |
| Naive Q35 | 6/6 | 386.6 | 6,675 | 6.957 | 6.750–7.436 |

- The arms need similar numbers of loss/gradient evaluations: acceptance falls between L-BFGS blocks 165 and 270. The gain therefore comes mainly from cheaper complete loss/gradient evaluations, not from fewer optimizer steps. Per evaluation (setup-plus-optimization time divided by evaluations, paired per seed), A24 and naive Q35 cost 3.537× (3.311–3.713) and 7.034× (6.757–7.300) as much as C17, which spends 8.0–8.7 ms per evaluation.
- The ratios measure the contracted state and the explicit parameter transpose together, because both baselines also use autograd for the parameter gradient; neither change is isolated.
- Pressure relative L2 is the binding gate. Its largest final values are 0.03966 (C17), 0.03972 (A24) and 0.039999 (naive Q35) against 0.04, while every other gate stays at or below 22% of its limit.
- The independent NumPy audit ([audit-v4-20261014.json](audit-v4-20261014.json)) verifies 14 source files and 91 raw files. It passes 1,512 comparisons over 2,227,032 numbers, with a maximum absolute discrepancy of 1.8e-14.
- [result-v4-20261014.json](result-v4-20261014.json) lists every case with its independent metrics, evaluation count and the L-BFGS block at acceptance.
- The complete 126-member archive (19,218,796 bytes, sha256 `1671ffc5…80dd41`) has verified copies on cluster scratch, persistent project storage and locally. It includes 8 macOS `._*` metadata files carried over from the upload; they are not in the source manifest and played no role in the run.

Scope:
- Both baselines are internal, and the problem is small (4,740 parameters, 512 interior points).
- C17 is a velocity–pressure contract. It is not the C36 curl-potential contract, and this result says nothing about C36 training.
- The two GPU lanes ran concurrently on one node whose other two GPUs carried another workload, so absolute times include that shared-node environment. Paired ratios compare arms of the same seed on the same GPU.

## External derivative baselines (pre-registered 2026-10-16)

The review asked for an external baseline on the three-dimensional C36 objective. C36 has no accepted solve, so this batch uses the accepted Ethier–Steinman benchmark instead; the C36 item stays open. [src/protocol-ext-20261016.json](src/protocol-ext-20261016.json) was committed before the batch ran. It keeps the v4 data, six formal seeds, initial parameter bytes, objective, optimizer program, stop checks and acceptance.

| Arm | Derivatives | Parameter gradient |
| --- | --- | --- |
| `c17_graph` (FlashNS) | C17 contracted state | explicit transpose |
| `torch_ad_graph` (external) | nested scalar-coordinate PyTorch autograd, as an ordinary PINN code and the 2D `torch_ad` adapter write it | autograd |
| `jet_collapsed_graph` (external) | Dangel et al.'s public `jet` (collapsed Taylor mode, pinned snapshot): spatial first derivatives and spatial Laplacian; a first-order jet gives d/dt; residual gradient by autograd, as in the 2D adapter | autograd |

- **Code.** [src/external.py](src/external.py) implements both external arms; neither uses FlashNS code.
- **Execution.** [src/solve_ext.py](src/solve_ext.py) is `solve.py` with this protocol, the arm table and `warn_only` determinism.
  - Every arm replays its complete loss/gradient as one CUDA Graph, as every v4 arm did; no arm is compiled.
  - All arms, C17 included, run in one batch on the Torch 2.12 runtime: one exclusive four-GPU node, the three arms of a seed on the same GPU, arm orders permuted over seeds.
  - [src/run_ext.py](src/run_ext.py) runs the batch once and stops starting cases 30 minutes before its deadline.
- **Analysis.** [src/audit_ext.py](src/audit_ext.py) is the v4 NumPy audit.
  - Its primary statistic is unchanged: baseline/C17 setup-plus-optimization geometric means over jointly accepted seeds.
  - It adds optimization-only and per-evaluation ratios.
- **Pre-submission checks.**
  - On CPU in the batch runtime, both external arms match independent scalar-coordinate AD within the v4 budget on 7-point and full data: at most 1.4e-4 of the budget.
  - The driver self-test runs all 18 cases as CPU smoke runs, every qualification passing at no more than 2.3e-4 of the budget. The audit then passes 1,512 comparisons, and the archive is verified.
  - The self-test caught one bug before submission: a stale `jet` cache keyed on a reused object id. It was removed.
- **Expectations** (not gates). Both external arms are expected to cost more per evaluation than C17; there is no numeric prediction. Pressure relative L2 was the binding gate in v4, so some external solves may fail acceptance. Ratios use jointly accepted pairs only.

### Result (2026-10-16)

The batch ran once on one four-GPU H100 node in about 8 minutes. All 18 solves exited normally and **all 18 are accepted** by both validators. The independent audit ([audit-ext-20261016.json](audit-ext-20261016.json)) passes 1,512 comparisons, with a maximum absolute discrepancy of 1.8e-14. The archive `575a586c…` (150 members) was re-read and verified.

| Arm | Accepted | Median s | Median evaluations | Paired ratio vs C17 (registered) | Optimization only | Per evaluation |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| FlashNS C17 | 6/6 | 47.2 | 6,728 | 1.000 | – | – |
| Collapsed jet (Dangel et al.) | 6/6 | 52.6 | 6,610 | **1.085** (0.98–1.19) | 1.082 | 1.096 |
| Nested torch AD | 6/6 | 89.0 | 6,709 | **1.901** (1.67–2.14) | 1.897 | 1.911 |

- **Binding gate.** Pressure relative L2 binds again: the largest values are 0.03994 (C17), 0.03997 (jet) and 0.03983 (torch AD) against 0.04. Every other gate stays at or below 22% of its limit.
- **Phase split** (post hoc, [ext-phase-split-20261016.json](ext-phase-split-20261016.json), `posthoc_ext_phase_split.py`).

  | Time per evaluation | C17 | Collapsed jet | Nested torch AD |
  | --- | ---: | ---: | ---: |
  | Adam phase (gradient, Adam step, stop checks) | 2.4–2.7 ms | 3.6–3.7 ms (1.37×) | 8.3–10.1 ms (3.62×) |
  | L-BFGS phase | 8.4–9.3 ms | 9.0–9.8 ms (1.07×) | 14.2–15.5 ms (1.70×) |

  - The L-BFGS phase adds about 6 ms per evaluation in every arm, as in 2D, and compresses the solve ratios.
  - Against a strong public Taylor-mode implementation, C17 shortens this small problem's solve by only about 8%, although its Adam-phase evaluation, which contains the gradient, is 1.37× faster.
  - One seed favours `jet`, and the protocol registered no gain rule, so 1.08× is a measured ratio, not a decided gain.
  - No arm was compiled, whereas the 2D comparison let each baseline use its faster qualified default or compiled variant. Compiled external arms and larger problems, where the gradient dominates, were not run.
- **Records.** [release-source-map-ext-20261016.json](release-source-map-ext-20261016.json): all 17 published files are byte-identical to their executed copies.
