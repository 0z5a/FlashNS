# FlashNS

FP64 Taylor-jet kernels and explicit parameter gradients for Navier–Stokes PINNs. FlashNS reduces the cost of third-order spatial derivatives and full parameter gradients through stable Taylor jets, packed interior/boundary data, and optional CUDA fusion, while retaining PyTorch/cuBLAS GEMMs where they are competitive.

On the recorded **H100 NVL Kovasznay solve**, FlashNS Hopper TMA reached the fixed accuracy criteria in **22.206 s**, versus **96.660 s for PyTorch nested autodiff** and **118.790 s for PhysicsNeMo**: **4.35× and 5.35× ratios of median accepted-run times**. These are three-seed results for the specified problem after a uniform budget extension; the original budget converged in 17/24 cases, and the extension reached 24/24. Failed-attempt costs and differing optimization trajectories are retained in the [complete-solver report](docs/pinn-solver-results.md).

[中文说明](README.zh-CN.md) · [Library comparison and evidence](docs/library-comparison.md) · [Local/GitHub synchronization](docs/source-sync-20260913.md)

## Improvements over existing library paths

All rows use FP64 and the same objective within their own experiment. Full solves include data preparation, backend/Graph setup, optimization, line searches, stopping checks, and independent acceptance; gradient replay rows exclude setup and optimization.

| Measurement | Existing library path | Baseline | FlashNS | Observed ratio |
| --- | --- | ---: | ---: | ---: |
| H100 NVL complete solve, median of 3 accepted runs | PhysicsNeMo PhysicsInformer AD | 118.790 s | Hopper TMA: 22.206 s | 5.35× |
| Same complete solve | PyTorch nested coordinate AD | 96.660 s | Hopper TMA: 22.206 s | 4.35× |
| Same complete solve | Compiled PyTorch Taylor jets | 25.941 s | Hopper TMA: 22.206 s | 1.17× |
| Same complete solve | cuEquivariance polynomial jets | 25.794 s | Hopper TMA: 22.206 s | 1.16× |
| H100 complete-gradient Graph replay, 16,384 points | Compiled PyTorch Taylor jets | 2.467 ms | F: 1.832 ms | 1.348× paired |
| Same gradient replay | cuEquivariance polynomial jets | 2.193 ms | F: 1.832 ms | 1.197× paired |
| Separate H100, gradient replay, 65,536 points | MathDx/cuBLASDx M64 adapter | 5.588 ms | F+: 4.282 ms | 1.305× paired |

The large gains over nested AD come primarily from the derivative representation. Against FlashNS's own library-GEMM jet baseline B1, Hopper TMA's complete-solver median improves only from 22.528 s to 22.206 s (1.014×). MathDx uses the documented finite GEMM adapter and a different toolchain; these results do not establish the tuning limit of a whole library. [Methods, raw JSON, and limitations](docs/library-comparison.md) distinguish these measurements from SM120 results.

The implementation adds:

- **Explicit derivative propagation:** factorial-normalized Q10/Q20 jets and parameter VJPs avoid repeated nested coordinate autodiff for the measured third-order PINN objective.
- **Stable saturated activations:** retained tanh derivative auxiliaries prevent reconstruction from a rounded, saturated activation value; CPU high-precision and GPU checks cover the stated tail cases.
- **Packed boundary work:** interior points carry full jets while boundary points carry values. For 2,048 interior and 512 boundary points, compact rows decrease from 25,600 to 20,992 (18%); this is a logical workload reduction, not a measured 18% solver or memory speedup.
- **Optional CUDA fusion:** dgrad/VJP, Hopper TMA, coordinate-gradient, and first-layer VJP/wgrad candidates have independent validation and timing entry points.

The implementation supports full third-order jets and value-only boundary points, packed residual seeds, stable tanh derivatives, and optional first-layer coordinate contractions. Optimizations remain explicit choices; the ordinary Torch path is the default.

## Install and run CPU checks

Python 3.11 or newer and PyTorch are required. From a source checkout:

```bash
uv sync --group dev
PYTHONPATH=src uv run pytest -q
PYTHONPATH=src uv run python experiments/pinn_v8/validate_host.py --help
```

The experiment drivers live in the source checkout and are not installed as package entry points. CPU tests do not establish GPU correctness or performance.

## CUDA validation

Use an existing Linux CUDA environment with `nvcc`, `compute-sanitizer`, `flock`, and GNU `timeout`. The scripts require an idle, explicitly selected SM120 device and an output directory that does not exist:

```bash
FLASHNS_PYTHON=python bash scripts/run_sm120_coordinate_wgrad.sh artifacts/g1 GPU_UUID
FLASHNS_PYTHON=python bash scripts/run_sm120_first_vjp_wgrad.sh artifacts/g2 GPU_UUID
```

Replace `GPU_UUID` with the selected device UUID from `nvidia-smi -L`. These entry points build fresh sources, perform numerical checks and sanitizer checks, then run local screening. They do not by themselves establish a complete-solver speedup.

For independent solves, see `experiments/pinn_v8/first_wgrad_solve.py --help` and the frozen protocol in `sources/flashns_v8_reference/evidence/frozen_solver_protocol.json`. Baseline and candidate must use separate builds and fresh processes, identical inputs and stopping criteria, and fixed comparison order.

## SM120 results and scope

The September 12 validation completed 12 independent solves for each comparison, all meeting the frozen accuracy criteria:

| Comparison | Full-solver result | Interpretation |
| --- | --- | --- |
| Torch first-layer gradient → G1 coordinate gradient | Geometric mean time decreased 3.842%; two of three seeds regressed | Stable speedup is unconfirmed; optimization trajectories differ |
| G1 → G2 fused VJP and coordinate gradient | Geometric mean time increased 0.068% | No confirmed speedup; final parameters and work counts match |

Kernel and route improvements must be reported separately from full-solver outcomes. See [validation details](docs/validation.md), including uncertainty and sanitizer coverage. These finite numerical experiments are not a mathematical proof of a fluid regularity or blow-up result.

The [tested source manifest](docs/tested-source-sha256.json) records the unchanged implementation and reference bytes used for the G1/G2 comparison. The supplemental release adds the local H100/Hopper/A4000 experiment drivers, full solver, affine-wave and Euler spline adapters, and selected historical evidence. Downloaded third-party source archives, environments, binaries, and checkpoints are not bundled; historical reports may describe such locally retained artifacts. This publication does not claim a new GPU run.

## Additional experiments

- [H100 library A/B](experiments/cuda_jet_h100/README.md), [Hopper kernels](experiments/cuda_jet_hopper/README.md), and [eight-backend complete solver](experiments/pinn_solver/README.md).
- [A4000 sample data parallelism](experiments/cuda_jet_v5/README.md) and [earlier CUDA fusion](experiments/cuda_jet_v2/README.md).
- [Euler spline and affine-wave reproduction](docs/day0-results.md). Run `PYTHONPATH=src uv run python -m flashns.cli fetch-sources` to retrieve hash-pinned inputs before source-dependent checks. The 4M-point Euler comparison measured 2.694× paired speedup over stable CPU evaluation, including GPU input/output transfers; initialization and comparison costs are reported separately in the [H100 report](docs/cuda-h100-results.md).
- [Formal-source reproduction and independent numerical adapter](experiments/openai_ns_formal/README.md), with [scope and results](docs/openai-ns-formal-results.md).

## Research manuscript

The [October 14 research draft](research/manuscript/README.md) keeps the accepted 2D H20 solves as the main result. It adds the first accepted 3D solve comparison and reports the C36 extension (full derivative qualification, matched H100/CPU objective-gradient timing, and the failed 3D training and physical acceptance). It is restructured after an external review; no earlier measured value changed.

**Accepted 3D solves ([research/ethier-3d](research/ethier-3d/README.md)).** On the transient Ethier–Steinman benchmark (FP64, H100), the 17-channel C17 contract passes all eight gates in 18 of 18 solves, together with the internal baselines. Paired setup-plus-optimization speedups are **3.602×** over the A24 full spatial-jet reference (range 3.353–4.284) and **6.957×** over a naive 35-monomial ablation (range 6.750–7.436), six seed pairs each. Both baselines are internal, and the network is small (4,740 parameters).

**C36 development run ([details](research/c36-potential-proof/README.md#development-run-with-a-larger-l-bfgs-budget-v2)).** The C36 problem has the same data, model and gates as before; only the L-BFGS budget grows 40×. All four runs (two engines × two seeds) pass the training gate. In the two fully observed final checkpoints (analytic engine), every physical gate passes except stratified corrected pressure (about 2.5× over its limit) and, for one seed, the trajectory endpoint (7% over). Under its pre-registered rule the outcome is mechanically branch (b). The analytic engine reaches each loss threshold 1.30–1.33× sooner than parameter AD. These are development seeds, so no formal time-to-accuracy claim is made.

**C36 checkpoints on every data plane, and a smoothness continuation ([details](research/c36-potential-proof/README.md#all-plane-observations-along-the-training-trajectory)).**
- **Observations.** 72 observations cover every saved development checkpoint and 26 continuation checkpoints. Each is checked on all data planes and re-read by the independent reader (8,424 checks, all pass). No checkpoint passes physical acceptance.
- **Errors still falling.** The physical errors fall with the loss and are still falling at the end of the budget. The remaining failure, stratified pressure inside the moving box, falls more slowly than the loss. So branch (b)'s "errors stop falling" does not hold either.
- **Smoothness continuation.** A continuation that starts with a slower, smoother box motion (transition width w = 1, then 3/4, then the original 1/2) changes where training meets the loss gate:
  - One schedule reaches the gate with 0.6–0.7× the baseline's stratified pressure error.
  - Another reaches loss thresholds in 8–36% fewer evaluations.
  - Neither changes which acceptance test fails. Under the registered rule, the slower schedule is harmful and the other fits no branch cleanly.

### Naming

| Name | Where | Meaning |
| --- | --- | --- |
| Q10 / Q20 | paper, code | Complete third-order Taylor state in 2D (10 channels) / 3D (20 channels) |
| GL8 | paper | 2D consumer state: value, gradient, packed Hessian, gradient of the Laplacian (8 channels) |
| C36 | paper, `research/c36-potential-proof` | 3D transient curl-potential consumer state (36 channels) |
| C17 / A24 | `research/c36-potential-proof/src` | Earlier 3D velocity-model contract (17 channels) and its full-spatial-third-order reference (24 channels) |
| B1 | `docs/cuda-*` | Library GEMM plus separate native jet forward/VJP (the library-GEMM jet baseline) |
| U / F | `docs/cuda-*` | Same CUTLASS main loop with a separate VJP (U) or fused dgrad/VJP (F) |
| B1+ / F+ | `docs/cuda-h100-results.md` | B1 / F with a finite Torch/cuBLASLt GEMM plan search |
| G1 / G2 | `docs/validation.md` | SM120 first-layer coordinate-gradient kernel (G1) and its fused VJP/wgrad variant (G2) |
