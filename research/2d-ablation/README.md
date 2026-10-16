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

## Results (attempt 2)

All 244 tasks completed on one four-GPU H100 node in 22 minutes.
- **Correctness.** The 226 timed processes pass the dense Q10 oracle, with a largest gradient error of 8.0e-15.
- **Drift.** The four drift repeats agree within 0.6%.
- **A/A gate.** It passes at 16,384 points (1.0002) but **fails at 2,048 points (1.057 against 3%)**, so every 2,048-point comparison failed a registered validity gate.
  - *Post hoc:* at 2,048 points the processes fall into two states about 6% apart. This happens in every compiled and graph cell, on all four GPUs, at a constant 1,980 MHz SM clock, with ≤ 0.5% spread inside each process. The cause was not found.

Full-gradient replay (ms) and ratios; above one favours the second cell:

| Points | A Q10 | B GL8 | C native | F autograd | A/B representation | B/C execution | A/C total | F/B transpose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.282 | 0.259 | 0.165 | 0.265 | 1.09 | 1.57 | 1.71 | 1.03 |
| 4,096 | 0.658 | 0.493 | 0.278 | 0.535 | 1.34 | 1.77 | 2.37 | 1.09 |
| 8,192 | 1.519 | 1.126 | 0.614 | 1.206 | 1.35 | 1.83 | 2.48 | 1.07 |
| 16,384 | 2.918 | 1.788 | 1.172 | 2.342 | 1.63 | 1.53 | 2.49 | 1.31 |
| 32,768 | 5.731 | 3.462 | 2.212 | 4.557 | 1.66 | 1.56 | 2.59 | 1.32 |
| 65,536 | 11.422 | 6.939 | 4.338 | 9.090 | 1.65 | 1.60 | 2.63 | 1.31 |

- **Execution.** Native pointwise kernels make the same GL8 computation 1.53–1.83× faster at every size.
  - B and C spend identical time in cuBLAS GEMMs: 571 and 573 µs per replay at 16,384 points.
  - Inductor emitted no GEMM templates, so the execution factor is pure pointwise work.
- **Representation.** It is small and unresolved at the original 2,048 points (1.09, inside the failed gate), 1.34–1.35 at 4k–8k, and 1.63–1.66 from 16k on.
  - Eager, GL8 is 1.27× (2k) and 1.91× (16k) faster than Q10.
  - Q10's GEMM time is 1.25× GL8's, the channel ratio 10/8, and its pointwise time is 1.84×.
- **Explicit transpose versus generic autograd** on the same GL8 state: no effect at 2k (1.03), 1.07–1.09 at 4k–8k, 1.31 from 16k on.
- **Size dependence.** The jump between 8k and 16k is a regime change; its cause was not profiled.
- **Topology.**
  - At 16k: representation is 1.61–1.66, execution 1.49–1.55 and total 2.39–2.56 for widths 32 and 128 and for 3 and 4 hidden layers.
  - At 2k: representation ranges from 0.92 (width 32, GL8 slower) to 1.63 (width 128). These cells have 3 processes each and share the 2k bimodality.
- **Solves (development, 2,048 points, six formal seeds, 18/18 accepted).**
  - The registered setup-plus-optimization geometric means are C/A 1.16 (0.91–1.40), B/A 0.98 and C/B 1.19.
  - Optimization alone gives 0.95, 0.96 and 0.99. The 1.16 comes from setup: Q10's Inductor compile takes 4.7–5.8 s against 0.3 s native.
  - Each evaluation spends 2.5–3.8 ms in optimizer work against under 0.3 ms of gradient replay. On H100 at this size the faster gradient therefore does not shorten a solve.

Files:
- [2d-ablation-result-20261016.json](2d-ablation-result-20261016.json): registered reader output.
- [2d-ablation-posthoc-20261016.json](2d-ablation-posthoc-20261016.json): clock/bimodality check, solve split, oracle maximum and profile microseconds, from `posthoc_ablation.py`; labelled post hoc.
- [release-source-map-20261016.json](release-source-map-20261016.json): executed copies of every published source.

Attempt 1 failed at start-up in all 244 tasks, before any timing ran: the source tree had been copied without its git metadata. Attempt 2 cloned the pinned commit and added a fail-fast guard; the protocol and plan were unchanged. Both archives are kept.
