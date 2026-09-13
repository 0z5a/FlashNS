# Improvements over existing library paths

FlashNS specializes the derivative representation and memory traffic of a third-order FP64 Navier–Stokes PINN. The comparisons below measure the included adapters on fixed workloads. They distinguish complete solves, complete-gradient replay, and incremental fusion gains.

## Complete solve: H100 NVL

The Kovasznay problem uses a 2–64–64–3 network, 2,048 interior points, 512 boundary points, viscosity 0.07, and a loss containing residual gradients. Each backend starts from the same three initialization seeds. Six frozen accuracy thresholds plus an independent scalar-autodiff acceptance check determine when a run is complete.

| Backend | Median accepted-run total, seconds | Backend / FlashNS Hopper TMA median |
| --- | ---: | ---: |
| PhysicsNeMo PhysicsInformer AD | 118.790 | 5.35× |
| PyTorch nested coordinate AD | 96.660 | 4.35× |
| Compiled PyTorch Taylor jets | 25.941 | 1.17× |
| cuEquivariance polynomial jets | 25.794 | 1.16× |
| cuBLASLt jet adapter | 22.723 | 1.023× |
| FlashNS B1, library GEMMs and separate jet operations | 22.528 | 1.014× |
| FlashNS F, fused dgrad/VJP | 22.427 | 1.010× |
| FlashNS Hopper TMA | 22.206 | 1.000× |

The ratios above divide each backend's three-run median by Hopper TMA's three-run median; they are not medians of paired ratios or confidence bounds. The first budget completed all 24 attempts but converged in 17. A separate extension raised only the common L-BFGS limit from 200 to 500 blocks: the seven failures were rerun from the original initialization, and all 24 backend/seed combinations then met the original criteria. The reported median uses each accepted full run. It does not charge those seven initial failed attempts to the successful run; their cumulative process cost is preserved separately.

Total solve time includes data preparation, backend/Graph setup, all optimizer and line-search work, stopping checks, and independent acceptance. Compilation and dependency caches were warm. Different floating-point reduction orders can change the L-BFGS trajectory and evaluation count, so these results do not isolate a single kernel's speed. Three seeds in one recorded workload do not establish a general library ranking.

The roughly 4–5× gains over nested AD are principally a derivative-representation improvement. Incremental fusion over an already efficient library-GEMM jet path is much smaller, as the B1/F/Hopper rows show.

Evidence: [full protocol and interpretation](pinn-solver-results.md), [per-seed summary and all-attempt costs](../artifacts/hopper_followup/solver-summary.json), [original suite](../experiments/pinn_solver/artifacts/formal1/suite.json), [extension suite](../experiments/pinn_solver/artifacts/extension1/suite.json). The suites and per-run JSON preserve failures as well as successes.

## Complete parameter gradients: H100 Graph replay

These runs compute the same FP64 loss and all parameter gradients at fixed inputs. They exclude optimizer steps, independent validation, compilation, plan search, and capture from the timed replay. Each ratio is the recorded median of within-run paired time ratios.

| Experiment | Existing path, median ms | FlashNS, median ms | Paired speedup |
| --- | ---: | ---: | ---: |
| 16,384 points, compiled PyTorch Taylor jets | 2.467 | F: 1.832 | 1.348× |
| 16,384 points, cuEquivariance polynomial jets | 2.193 | F: 1.832 | 1.197× |
| Separate H100, 65,536 points, MathDx M64 adapter | 5.588 | F+: 4.282 | 1.305× |

F fuses dgrad with the activation VJP; F+ additionally uses the finite cuBLASLt plan search. The MathDx/cuBLASDx experiment uses its documented M32/M64 static GEMM adapter and retains Torch for wgrad. Its CUDA 13 toolchain differs from the native CUDA 12.8 toolchain, and it does not exhaust MathDx's tuning space. Both sides of each A/B were measured on the same device; raw times from different H100 sessions are not combined.

cuEquivariance is competitive on some local VJP shapes; the complete-gradient benefit includes forward, backward, and common GEMMs. The scientific-library matrix covers 4K/16K points, while MathDx has its own 4K–262K matrix. These are different experiment sets.

Evidence: [H100 report and reproduction commands](cuda-h100-results.md), [scientific-library raw measurements](../experiments/cuda_jet_h100/artifacts/scientific_benchmark.json), [scientific validation](../experiments/cuda_jet_h100/artifacts/scientific_validation_v2.json), [MathDx raw measurements](../experiments/cuda_jet_h100/artifacts_h100b/mathdx_benchmark.json), [MathDx validation](../experiments/cuda_jet_h100/artifacts_h100b/mathdx_validation.json).

## Implementation improvements and validation limits

- Full Q10/Q20 factorial-normalized jets and explicit parameter VJPs reduce repeated high-order coordinate autodiff. The supported contract is first-order parameter VJP, not HVP or double backward.
- Stable tanh auxiliaries retain small derivatives when tanh itself rounds to saturation. Fixed FP64 tolerance and high-precision tail checks accompany the implementation.
- Compact v8 layout gives interior points full jets and boundary points only values: 25,600 → 20,992 logical rows for the frozen problem. The 18% reduction describes this tensor layout, not measured total allocator usage or solver time.
- Coordinate-gradient and VJP/wgrad fusion remain explicit SM120 candidates. Torch→G1 averaged 3.842% less time but two seeds regressed; G1→G2 averaged 0.068% more time. Neither cleared the predefined benefit gate. [SM120 evidence](validation.md) does not support changing the default.

The independent Euler spline experiment also reduced stable FP64 residual evaluation from 41.918 s on CPU to 15.558 s on GPU over 4M frozen points (2.694× paired, including transfers). It is a separate workload, with initialization and output-comparison overhead itemized in the [H100 report](cuda-h100-results.md). Invalid legacy numerical paths are excluded from effective speedup ratios.

## Recompute the summary

```bash
python scripts/summarize_library_comparison.py
```

This standard-library-only command reads the published JSON, checks completion/convergence, prints the ratios, and records input SHA-256 values. It does not rerun GPU benchmarks. Historical build paths, binary hashes, and local archive references in the raw evidence describe the original measurement environment; binaries, checkpoints, large input archives, and external toolchains must be obtained or rebuilt separately.
