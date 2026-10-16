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

## Status

Pre-registered; the GPU batch is pending.
