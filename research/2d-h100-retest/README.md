# 2D H100 retest under the H20 protocol (pre-registered)

This study runs the paper's two-dimensional H20 solve campaign unchanged on an NVIDIA H100. The review asked for this "2D H100 re-test following the H20 protocol".

- **Unchanged from H20.** The 460 source, adapter and snapshot files ([H20-expectations.json](H20-expectations.json)), the frozen data and the six formal initialization byte strings ([frozen-H20-inputs.json](frozen-H20-inputs.json)), the optimizer budget and the acceptance gates.
- **Candidate.** GL8 native, in every lane.
- **Preparation.** The batch was prepared on 2026-10-13 and cancelled while pending; it never ran. This attempt changes only the Slurm job name in the batch configuration. Every input and runtime file is hash-checked at start.

[protocol-2d-h100-retest-20261016.json](protocol-2d-h100-retest-20261016.json) was committed before the batch ran.

## Batch

[finite-plan.json](finite-plan.json): one node, four H100 lanes, a fresh process per case, no retries.

| Lane | Group | Qualification | Cases |
| ---: | --- | ---: | ---: |
| 0 | public paper methods: Collapsed Taylor, Cao ED, exact STDE, torch AD, compact Q10, GL8 | 9 | 36 solves |
| 1 | NVIDIA methods: PhysicsNeMo, cuDNN, cuEquivariance, GL8 | 8 | 24 solves |
| 2 | representation: Q10, GL8, GL8 chunk-512 | 3 | 18 solves |
| 3 | memory and replay: Q10, GL8, GL8 chunk-512 at 2,048 and 16,384 points | 3 | 30 benchmarks |

- **Variant selection.** Each backend runs its solves with the faster of its GPU-qualified default and compiled variants.
- **Drivers.** `h20_node.py` runs the four lanes (`h20_gpu_lane.py`) and the audit, then writes a verified archive. `h20_case.py` runs one case with the frozen inputs.

## Analysis

- **Primary.** `audit_h20.py`, unchanged:
  - a NumPy recheck of the qualification gradients;
  - a complete CPU scalar-coordinate-AD recheck of every accepted checkpoint;
  - per lane, the paired geometric mean of baseline/GL8 setup-plus-optimization time over jointly accepted seeds.
- **Supplementary.** [read_h20_h100_supplement.py](read_h20_h100_supplement.py), registered before the run, adds for the same pairs:
  - optimization-only and per-evaluation ratios;
  - the Adam/L-BFGS phase cost of every solve;
  - each solve's compiler-cache state.
- **Expectation** (not a gate). The [solve-scale study](../2d-solve-scale/README.md) found a fixed L-BFGS cost of 5.9–7.1 ms per evaluation at 2,048 points on H100. GL8's optimization-time advantage should therefore compress toward 1 against baselines with cheap gradients, and persist against baselines whose gradient costs several milliseconds or more.
- **Comparison with H20.** H20 ratios are reported beside the H100 ones and never pooled.

The public `h20_node.py` reads the node-local scratch root from `NODE_LOCAL_SCRATCH`; the executed copy names it directly.

## Results (2026-10-16)

The batch ran once in 85 minutes; all four lanes and the audit exited 0.
- **Accepted solves.** All 78 solves are accepted and rechecked by complete CPU scalar AD: 36 public-method, 24 NVIDIA-method and 18 representation solves.
- **Qualification.** Every qualification gradient is within 5.5e-4 of its budget. 21 variants qualified. The two cuEquivariance-native variants failed qualification: the fused tensor product has no registered double-backward formula. The cuEquivariance polynomial arm ran.
- **Variants that ran the solves.** The compiled variant for ED, collapsed Taylor, torch AD, PhysicsNeMo and cuDNN; the default for STDE, Q10 and GL8.
  - Compiling sped up qualification replay as follows: collapsed Taylor 2.36 → 0.75 ms, ED 3.04 → 0.61 ms, torch AD 4.91 → 2.11 ms, PhysicsNeMo 4.00 → 1.72 ms.
- **Cache state.** Every solve started after a same-backend qualification on its lane, so with a warm compiler cache.
- **Archive.** The archive `c3672aa2…` (8,851 members, including per-lane caches) was verified.

| Baseline / GL8 | H100 setup + opt. (registered) | H100 optimization only | H100 per evaluation | H20 setup + opt. |
| --- | --- | ---: | ---: | ---: |
| Q10 compact (papers lane) | 1.22 (1.12–1.39) | 1.02 | 1.01 | 1.66 |
| Collapsed Taylor | 1.37 (1.28–1.63) | 1.21 | 1.18 | 1.73 |
| Cao ED | 1.29 (1.13–1.42) | 1.09 | 1.10 | 2.31 |
| Torch coordinate AD | 1.90 (1.79–2.00) | 1.67 | 1.64 | 2.99 |
| STDE exact (2D) | 1.69 (1.45–1.88) | 1.58 | 1.61 | 3.37 |
| PhysicsNeMo | 1.93 (1.51–2.43) | 1.52 | 1.51 | 3.11 |
| cuEquivariance polynomial | 1.49 (1.29–1.94) | 1.34 | 1.33 | 1.56 |
| cuDNN convolution AD | 30.1 (25.9–35.7) | 30.4 | 30.3 | 22.9 |
| GL8 chunk-512 (representation lane) | 1.10 (1.04–1.31) | 1.10 | 1.11 | 1.42 |
| Q10 compact (representation lane) | 1.24 (1.13–1.42) | 1.02 | 1.01 | 1.66 |

- **GL8 is still faster than every baseline, by much less than on H20,** as the registered expectation said.
  - In optimization alone, Q10 (1.02) and ED (1.09) are nearly level. The remaining Q10 gap is setup: 4.8 s with a warm compiler cache against 0.3–0.6 s native.
  - Consistency: the representation lane agrees with the ablation and the precision study at 2,048 points. Q10/GL8 is 0.95–1.02 in optimization alone and 1.16–1.32 in setup plus optimization across the three batches.
- **Phase split** (supplementary).
  - Adam phase, per evaluation: GL8 0.24–0.25 ms, Q10 0.37–0.40, ED 0.69–0.73, collapsed Taylor 0.86–0.89, cuEquivariance 1.2–1.4, PhysicsNeMo 1.9, STDE 2.0, torch AD 2.3, cuDNN 90 ms.
  - L-BFGS phase: every backend except cuDNN costs 6.3–9.0 ms per evaluation. That shared cost compresses the ratios at this size.
  - The larger H20 ratios are consistent with the FP64 gradient taking a larger share of each evaluation on H20; the H20 L-BFGS cost was not measured.
- **Memory and replay lane.** It reproduces the chunk trade-off at 16,384 points: chunk-512 allocates an 85% lower setup peak than GL8 (45.9 against 312.8 MiB) at 3.4× its replay time.
- **Scope.** No H100 ratio is pooled with H20.

Files:
- [audit-h20-h100-20261016.json](audit-h20-h100-20261016.json): the primary audit, with the batch path normalized to `<batch>/`.
- [h20-h100-supplement-20261016.json](h20-h100-supplement-20261016.json): the supplementary reader's output.
- [release-source-map-20261016.json](release-source-map-20261016.json).
