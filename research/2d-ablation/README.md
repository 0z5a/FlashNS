# 2D representation/execution ablation (H100, pre-registered)

This study answers the external review's main question about the 2D result: how much of GL8's speed comes from the derivative **representation** (the 8-channel consumer state versus the complete 10-channel Q10 jet), and how much from the **execution** (native CUDA pointwise islands versus Inductor-generated kernels, compiled versus eager)? It also measures how both effects scale with interior points, width and depth.

[protocol-2d-ablation-20261016.json](protocol-2d-ablation-20261016.json) was committed before the GPU batch ran.

## Cells

Each cell computes the same FP64 Kovasznay full loss and all parameter gradients, replayed as a CUDA Graph.

| Cell | Backend | Execution | Compared with |
| --- | --- | --- | --- |
| A | `q10_compact` (complete Q10 jet) | Inductor | — |
| B | `gl8_compact_output` (GL8, explicit output-state VJP) | Inductor | A: representation under the same execution |
| C | `gl8_native` (same GL8 layout and VJP, native CUDA pointwise islands) | no compile | B: execution only |
| D | `q10_compact` | eager | — |
| E | `gl8_compact_output` | eager | D: representation without the compiler |
| F | `gl8_compact_autograd` (same GL8 state, generic parameter AD) | Inductor | B: the explicit transpose |

There is no native-kernel Q10 backend, so the total ratio C/A is decomposed as (B/A) × (C/B).

**Prior evidence, stated before the run.** An A100 study (2,048 points) found the compact GL8 state under Inductor *not* faster than compact Q10. The protocol therefore expects a representation effect near zero at that size, with most of the earlier solve-level gain coming from execution. The result is reported either way.

## Design

- **Sizes.**
  - Core: cells A–F at 2,048 and 16,384 interior points, 5 fresh processes each.
  - Point sweep: cells A, B, C and F at 4,096, 8,192, 32,768 and 65,536 points, 5 processes each.
  - Width/depth: cells A, B and C at hidden widths 32×32, 128×128, 64×64×64 and 64×64×64×64, at 2,048 and 16,384 points, 3 processes each.
  - Solves: cells A, B and C on the six formal seeds, as development data.
  - Boundary coordinates stay at 512 for every size.
- **Timing.** The unchanged `bench_gpu` method:
  - a check against the dense Q10 oracle (atol 2e-12, rtol 2e-11) before timing;
  - a 2 s warmup, then 7 × 100 CUDA Graph replays timed with CUDA events;
  - compile counters must not change during timing.
  - The cell statistic is the median over processes of each process's median.
- **Validity gates.**
  - A/A: cell A relabelled, which must agree within 3%.
  - Drift: four core tasks repeated at the end, within 5%.
  - Every process must pass the oracle check.
  - Four lanes pull from one pre-randomized queue, and every task's lane is recorded.
- **Effects.** Each effect is a time ratio with a percentile-bootstrap interval over processes; |ratio − 1| < 3% counts as no effect.
  - representation, compiled and eager: A/B and D/E;
  - execution: B/C;
  - total: A/C;
  - explicit transpose: F/B.
- **Profiles.** An untimed one-replay trace per core cell gives the kernel-time breakdown (GEMM, native islands, Inductor-generated, reduction, copy). It also records whether Inductor kept cuBLAS for the matrix products.

All numbers are H100 numbers; none is compared with or pooled with the H20 campaign.

## Files

- `make_ablation_plan.py` builds the frozen, pre-randomized [plan](plan-2d-ablation-20261016.json): 244 tasks in five tiers. The core decomposition comes first, so a deadline cut cannot remove it.
- `run_ablation.py` is the cluster-side driver, and `entry_ablation.sh` is the start command embedded in the batch script.
  - Admission checks the source and tool manifests, the plan hash, the GPU lock inodes and idle GPUs, and that the five data hashes and six formal initialization hashes match the frozen H20 inputs.
- `read_ablation.py` is the reader and registered analysis.
- `src/ablation_bench.py` is a width/depth wrapper around the unchanged `bench_gpu.py`.
  - It runs inside the pinned 2D source tree (commit `286d7e394c8b`), as do the timed backends.
  - At the original 2–64–64–3 topology it replaces nothing.
  - Its CPU checks (`src/test_ablation_shapes.py`) show the generic layer views are identical to the original at that topology, and that GL8 (explicit and autograd) and compact Q10 match the dense Q10 oracle at four other topologies.

## Status

Pre-registered; the GPU batch is pending.
