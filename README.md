# FlashNS

FP64 Taylor-jet kernels and explicit parameter gradients for a Kovasznay Navier–Stokes PINN. This initial source release includes the tested SM120 coordinate-gradient and fused VJP experiments, their CPU reference tests, CUDA validation entry points, and the frozen scientific stopping protocol.

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

## Results and scope

The September 12 validation completed 12 independent solves for each comparison, all meeting the frozen accuracy criteria:

| Comparison | Full-solver result | Interpretation |
| --- | --- | --- |
| Torch first-layer gradient → G1 coordinate gradient | Geometric mean time decreased 3.842%; two of three seeds regressed | Stable speedup is unconfirmed; optimization trajectories differ |
| G1 → G2 fused VJP and coordinate gradient | Geometric mean time increased 0.068% | No confirmed speedup; final parameters and work counts match |

Kernel and route improvements must be reported separately from full-solver outcomes. See [validation details](docs/validation.md), including uncertainty and sanitizer coverage. These finite numerical experiments are not a mathematical proof of a fluid regularity or blow-up result.

The [tested source manifest](docs/tested-source-sha256.json) records the unchanged implementation and reference bytes used for the G1/G2 comparison. Large generated artifacts, binaries, downloaded dependencies, and machine-specific orchestration records are excluded from this source release.
