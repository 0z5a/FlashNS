# SM120 validation, September 12, 2026

The two independent comparisons used RTX PRO 4000 Blackwell 24 GB, SM120, Torch 2.13.0+cu130, SciPy 1.18.1, and nvcc 13.0.88. Computation was FP64 with `--fmad=false` and no fast math. Each comparison used three fixed seed quartets in APPA/PAAP/APPA order, with separate builds and fresh processes, models, optimizers, and CUDA Graphs. Pilot runs were excluded.

## Complete solver

Kovasznay steady Navier–Stokes, viscosity 0.07, network 2–64–64–3, compact layout, 2,048 interior points, 512 boundary points, 8,192 validation points, data seed 642701. Adam runs for at most 4,000 steps, followed by at most 500 L-BFGS blocks of 20 iterations using strong Wolfe search.

All six thresholds must pass: PDE RMS 0.02, boundary velocity RMS 0.01, outlet pressure RMS 0.01, and relative L2 u/v/p errors 0.01/0.02/0.02. Results must also pass an independent scalar-autodiff check. All 24 solves across the two comparisons passed.

Total time includes data preparation, Graph/backend setup, optimization, line searches, stopping checks, and final independent validation. It excludes process startup, initial CUDA loading, compilation, and preflight; it is not cold-start timing.

| Torch → G1 seed | Torch seconds, geometric mean | G1 seconds, geometric mean | Time reduction |
| --- | ---: | ---: | ---: |
| 86201 | 23.846939 | 23.958537 | -0.4680% |
| 86202 | 26.344604 | 27.328636 | -3.7352% |
| 86203 | 35.530992 | 30.311832 | +14.6890% |

Aggregate time reduction was 3.8418%; the exploratory cluster-bootstrap 95% range was [-3.7352%, 14.6890%]. Three independent seeds in one session do not support a broad speed claim. G1 changes reduction order, trajectories, and gradient evaluation counts.

| G1 → G2 seed | G1 seconds, geometric mean | G2 seconds, geometric mean | Time reduction |
| --- | ---: | ---: | ---: |
| 75101 | 33.332058 | 33.127310 | +0.6143% |
| 75102 | 34.642766 | 34.623173 | +0.0566% |
| 75103 | 27.327228 | 27.567820 | -0.8804% |

Aggregate time increased 0.0680%; the exploratory 95% time-reduction range was [-0.8804%, 0.6143%]. G1 and G2 final parameter bytes, gradient evaluation counts, and recorded training checkpoints matched per seed. Neither comparison cleared the predefined 1% lower-bound benefit gate. Default dispatch remains unchanged.

## Correctness and sanitizer coverage

The original GPU runs covered Q10/Q20, full/value-only/empty inputs, current streams, boundary guards, nonfinite inputs, saturated activations, Graph input updates, and three real Adam updates in every layout against nested autodiff. Fresh builds passed memcheck, racecheck, initcheck, and synccheck with zero errors and no racecheck warnings. Sanitizers covered kernel lifetime and Graph updates, not the complete optimizer execution.

Source and loaded-binary hashes, independent process identity, frozen protocol, six acceptance metrics, and all final checkpoint hashes were audited in the original runs. The release manifest preserves the source bytes; this publication does not claim a new GPU rerun.

The historical static metadata field `logical_gemm_calls_per_evaluation` remains 8; G1/G2 actually execute 7 GEMMs per segment. That field was not used for timing or convergence decisions.
