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

On 2026-10-14 all 18 planned solves (6 formal seeds × 3 arms) ran on two NVIDIA H100 GPUs (FP64), and all 18 passed every gate in both the stop check and the independent final check. Times are setup plus optimization. The ratio is the geometric mean of six per-seed baseline/C17 ratios, not a ratio of medians.

| Arm | Accepted | Median s | Median evaluations | Paired ratio vs C17 | Range |
| --- | ---: | ---: | ---: | ---: | --- |
| FlashNS C17 | 6/6 | 55.7 | 6,676 | 1.000 | – |
| A24 reference | 6/6 | 210.1 | 7,220 | 3.602 | 3.353–4.284 |
| Naive Q35 | 6/6 | 386.6 | 6,675 | 6.957 | 6.750–7.436 |

- The arms need similar numbers of loss/gradient evaluations: acceptance falls between L-BFGS blocks 165 and 270. The gain therefore comes mainly from cheaper complete loss/gradient evaluations, not from fewer optimizer steps.
- Pressure relative L2 is the binding gate. Its largest final values are 0.03966 (C17), 0.03972 (A24) and 0.039999 (naive Q35) against 0.04, while every other gate stays at or below 22% of its limit.
- The independent NumPy audit ([audit-v4-20261014.json](audit-v4-20261014.json)) verifies 14 source files and 91 raw files. It passes 1,512 comparisons over 2,227,032 numbers, with a maximum absolute discrepancy of 1.8e-14.
- [result-v4-20261014.json](result-v4-20261014.json) lists every case with its independent metrics, evaluation count and the L-BFGS block at acceptance.
- The complete 126-member archive (19,218,796 bytes, sha256 `1671ffc5…80dd41`) has verified copies on cluster scratch, persistent project storage and locally.

Scope:
- Both baselines are internal, and the problem is small (4,740 parameters, 512 interior points).
- C17 is a velocity–pressure contract. It is not the C36 curl-potential contract, and this result says nothing about C36 training.
- The two GPU lanes ran concurrently on one node whose other two GPUs carried another workload, so absolute times include that shared-node environment. Paired ratios compare arms of the same seed on the same GPU.
