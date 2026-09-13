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
