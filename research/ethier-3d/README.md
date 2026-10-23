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

On 2026-10-14 all 18 planned solves (6 formal seeds × 3 arms) ran on two NVIDIA H100 GPUs (FP64, TF32 disabled, Torch 2.4.0a0, NumPy 1.26.4), and all 18 passed every gate in both the stop check and the independent final check. Times are setup plus optimization. The ratio is the geometric mean of six per-seed baseline/C17 ratios, not a ratio of medians.

| Arm | Accepted | Median s | Median evaluations | Paired ratio vs C17 | Range |
| --- | ---: | ---: | ---: | ---: | --- |
| FlashNS C17 | 6/6 | 55.7 | 6,676 | 1.000 | – |
| A24 reference | 6/6 | 210.1 | 7,220 | 3.602 | 3.353–4.284 |
| Naive Q35 | 6/6 | 386.6 | 6,675 | 6.957 | 6.750–7.436 |

- The arms need similar numbers of loss/gradient evaluations: acceptance falls between L-BFGS blocks 165 and 270. The gain therefore comes mainly from cheaper complete loss/gradient evaluations, not from fewer optimizer steps. Per evaluation (setup-plus-optimization time divided by evaluations, paired per seed), A24 and naive Q35 cost 3.537× (3.311–3.713) and 7.034× (6.757–7.300) as much as C17, which spends 8.0–8.7 ms per evaluation.
- The ratios measure the contracted state and the explicit parameter transpose together, because both baselines also use autograd for the parameter gradient; neither change is isolated.
- Pressure relative L2 is the binding gate. Its largest final values are 0.03966 (C17), 0.03972 (A24) and 0.039999 (naive Q35) against 0.04, while every other gate stays at or below 22% of its limit.
- The independent NumPy audit ([audit-v4-20261014.json](audit-v4-20261014.json)) verifies 14 source files and 91 raw files. It passes 1,512 comparisons over 2,227,032 numbers, with a maximum absolute discrepancy of 1.8e-14.
- [result-v4-20261014.json](result-v4-20261014.json) lists every case with its independent metrics, evaluation count and the L-BFGS block at acceptance.
- The complete 126-member archive (19,218,796 bytes, sha256 `1671ffc5…80dd41`) has verified copies on cluster scratch, persistent project storage and locally. It includes 8 macOS `._*` metadata files carried over from the upload; they are not in the source manifest and played no role in the run.

Scope:
- Both baselines are internal, and the problem is small (4,740 parameters, 512 interior points).
- C17 is a velocity–pressure contract. It is not the C36 curl-potential contract, and this result says nothing about C36 training.
- The two GPU lanes ran concurrently on one node whose other two GPUs carried another workload, so absolute times include that shared-node environment. Paired ratios compare arms of the same seed on the same GPU.

## External derivative baselines (pre-registered 2026-10-16)

The review asked for an external baseline on the three-dimensional C36 objective. C36 has no accepted solve, so this batch uses the accepted Ethier–Steinman benchmark instead; the C36 item stays open. [src/protocol-ext-20261016.json](src/protocol-ext-20261016.json) was committed before the batch ran. It keeps the v4 data, six formal seeds, initial parameter bytes, objective, optimizer program, stop checks and acceptance.

| Arm | Derivatives | Parameter gradient |
| --- | --- | --- |
| `c17_graph` (FlashNS) | C17 contracted state | explicit transpose |
| `torch_ad_graph` (external) | nested scalar-coordinate PyTorch autograd, as an ordinary PINN code and the 2D `torch_ad` adapter write it | autograd |
| `jet_collapsed_graph` (external) | Dangel et al.'s public `jet` (collapsed Taylor mode, pinned snapshot): spatial first derivatives and spatial Laplacian; a first-order jet gives d/dt; residual gradient by autograd, as in the 2D adapter | autograd |

- **Code.** [src/external.py](src/external.py) implements both external arms; neither uses FlashNS code.
- **Execution.** [src/solve_ext.py](src/solve_ext.py) is `solve.py` with this protocol, the arm table and `warn_only` determinism.
  - Every arm replays its complete loss/gradient as one CUDA Graph, as every v4 arm did; no arm is compiled.
  - All arms, C17 included, run in one batch on the Torch 2.12 runtime: one exclusive four-GPU node, the three arms of a seed on the same GPU, arm orders permuted over seeds.
  - [src/run_ext.py](src/run_ext.py) runs the batch once and stops starting cases 30 minutes before its deadline.
- **Analysis.** [src/audit_ext.py](src/audit_ext.py) is the v4 NumPy audit.
  - Its primary statistic is unchanged: baseline/C17 setup-plus-optimization geometric means over jointly accepted seeds.
  - It adds optimization-only and per-evaluation ratios.
- **Pre-submission checks.**
  - On CPU in the batch runtime, both external arms match independent scalar-coordinate AD within the v4 budget on 7-point and full data: at most 1.4e-4 of the budget.
  - The driver self-test runs all 18 cases as CPU smoke runs, every qualification passing at no more than 2.3e-4 of the budget. The audit then passes 1,512 comparisons, and the archive is verified.
  - The self-test caught one bug before submission: a stale `jet` cache keyed on a reused object id. It was removed.
- **Expectations** (not gates). Both external arms are expected to cost more per evaluation than C17; there is no numeric prediction. Pressure relative L2 was the binding gate in v4, so some external solves may fail acceptance. Ratios use jointly accepted pairs only.

### Result (2026-10-16)

The batch ran once on one four-GPU H100 node in about 8 minutes. All 18 solves exited normally and **all 18 are accepted** by both validators. The independent audit ([audit-ext-20261016.json](audit-ext-20261016.json)) passes 1,512 comparisons, with a maximum absolute discrepancy of 1.8e-14. The archive `575a586c…` (150 members) was re-read and verified.

| Arm | Accepted | Median s | Median evaluations | Paired ratio vs C17 (registered) | Optimization only | Per evaluation |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| FlashNS C17 | 6/6 | 47.2 | 6,728 | 1.000 | – | – |
| Collapsed jet (Dangel et al.) | 6/6 | 52.6 | 6,610 | **1.085** (0.98–1.19) | 1.082 | 1.096 |
| Nested torch AD | 6/6 | 89.0 | 6,709 | **1.901** (1.67–2.14) | 1.897 | 1.911 |

- **Binding gate.** Pressure relative L2 binds again: the largest values are 0.03994 (C17), 0.03997 (jet) and 0.03983 (torch AD) against 0.04. Every other gate stays at or below 22% of its limit.
- **Phase split** (post hoc, [ext-phase-split-20261016.json](ext-phase-split-20261016.json), `posthoc_ext_phase_split.py`).

  | Time per evaluation | C17 | Collapsed jet | Nested torch AD |
  | --- | ---: | ---: | ---: |
  | Adam phase (gradient, Adam step, stop checks) | 2.4–2.7 ms | 3.6–3.7 ms (1.37×) | 8.3–10.1 ms (3.62×) |
  | L-BFGS phase | 8.4–9.3 ms | 9.0–9.8 ms (1.07×) | 14.2–15.5 ms (1.70×) |

  - The L-BFGS phase adds about 6 ms per evaluation in every arm, as in 2D, and compresses the solve ratios.
  - Against a strong public Taylor-mode implementation, C17 shortens this small problem's solve by only about 8%, although its Adam-phase evaluation, which contains the gradient, is 1.37× faster.
  - One seed favours `jet`, and the protocol registered no gain rule, so 1.08× is a measured ratio, not a decided gain.
  - No arm was compiled, whereas the 2D comparison let each baseline use its faster qualified default or compiled variant. Compiled external arms and larger problems, where the gradient dominates, were not run.
  - In the 2D H100 retest, compiling made the collapsed-Taylor, ED and torch-AD replays 3.2×, 5.0× and 2.3× faster ([details](../2d-h100-retest/README.md)). A compiled `jet` could therefore match or beat C17 here; C17 itself is uncompiled explicit code.
- **Records.** [release-source-map-ext-20261016.json](release-source-map-ext-20261016.json): all 17 published files are byte-identical to their executed copies.

## Compiled external baselines (pre-registered 2026-10-17)

The uncompiled batch above left one question open. In the 2D H100 retest, compiling made the public packages' replays 2.3–5.0× faster, and the uncompiled collapsed jet was already within 8% of C17. [src/protocol-ext2-20261017.json](src/protocol-ext2-20261017.json), committed before this batch ran, asks whether compiling changes the comparison.

- **Arms.** Six arms on the six formal seeds, one fresh process each, 36 solves: C17, nested torch AD and collapsed jet, each in two variants.
  - *Graph*: the arm exactly as in the uncompiled batch.
  - *Compiled*: [`external.build_compiled`](src/external.py) traces the whole loss/gradient with `make_fx`, over a model that aliases the solver's parameters. It then compiles with `torch.compile(fullgraph=True, dynamic=False)` and captures one CUDA Graph.
  - Data, seeds, objective, optimizer, stop checks, qualification and acceptance are unchanged.
  - The full-data qualification replays each arm itself, compiled arms included, through three in-place parameter updates.
- **Warm-up and selection.** Each lane first runs every arm on development seed 642201 in warm-only mode: setup, capture, qualification and 200 timed replays, with no training.
  - Each method's *selected variant* is the one with the lower median warm-up replay time among qualified variants, fixed before any formal solve is read. This is the 2D rule.
- **Registered statistics.** The selected external variant over the selected C17 variant, as paired geometric means of setup-plus-optimization time (primary) and optimization-only time (secondary).
  - C17 is faster if the ratio exceeds 1.03 and at least 5 of the jointly accepted pairs favour it.
  - The external method is faster under the mirrored condition.
  - Otherwise there is no decided difference; with fewer than 5 pairs, the result is not decidable.
  - If formal compiled solves still show cold-cache setup, the reading leans on the optimization-only decision.
- **Supplementary.** Every arm against `c17_graph`, which also replicates the uncompiled batch, plus the compile factor per method and each solve's cache state.
- **Code and caches.**
  - [src/solve_ext2.py](src/solve_ext2.py) is `solve_ext.py` with the six-arm table, the compiled builder and warm-only mode.
  - [src/run_ext2.py](src/run_ext2.py) adds the warm-up stage and a cyclic arm order. It points every compiler and JIT cache (Inductor, Triton, XDG, TMPDIR, CUDA) to the batch root, and leaves them out of the archive.
  - [src/read_ext2.py](src/read_ext2.py) applies the registered rules after the unchanged [audit](src/audit_ext2.py).
- **Pre-submission checks.**
  - On CPU in the batch runtime, all three compiled paths match scalar-coordinate AD on 7 points within 2e-4 of the budget, before and after an in-place parameter update.
  - None of them grew the home-directory caches.
  - The driver self-test on the login node passed. It ran 24 warm-up and 36 formal CPU smoke cases, all of which exited 0. The audit passed 3,024 comparisons and the reader exited 0. The archive (313 members, caches excluded) was verified, and the home caches were unchanged.

### Result of the compiled batch (2026-10-17)

The batch ran once on one four-GPU H100 node in about 20 minutes. The audit passed 2,520 comparisons, with a maximum discrepancy of 1.8e-14. The registered reader exited 0. The archive `2d9140ed…` (293 members, compiler caches excluded) was verified.

- **The compiled C17 variant failed qualification in all ten attempts** (4 warm-ups, 6 formal solves). Its warm-up calls succeeded, but recompilation during CUDA Graph capture reached a generator that the compiler cannot trace (`Attempt to trace generator`). The cause was not found.
- **Selection** by warm-up replay: C17 graph (2.52 ms); collapsed jet compiled (0.85 ms, against 3.45 ms graph); nested AD compiled (2.39 ms, against 9.76 ms graph).
- **Acceptance.** All 30 solves of the five qualified arms are accepted.

| Selected external / selected C17 | Setup + optimization (primary) | Optimization only (secondary) |
| --- | --- | --- |
| Compiled collapsed jet / C17 graph | 1.05 (0.95–1.24), 3/6 favour C17: **no decided difference** | **0.81** (0.74–0.99), 0/6 favour C17: **external faster** |
| Compiled nested AD / C17 graph | **1.59** (1.50–1.75), 6/6: **C17 faster** | 1.01 (0.97–1.05): no decided difference |

- **Cache state.** Compiled setups in formal solves (jet 10.6–11.1 s, nested AD 25.8–26.8 s) are below their warm-up setups (22.7 s and 55.1 s). The registered cache caveat therefore does not apply.
- **Replication.** The graph-only arms reproduce the uncompiled batch: 1.11 for jet and 1.92 for nested AD, against 1.085 and 1.90.
- **Phase split** (post hoc, [ext2-phase-split-20261017.json](ext2-phase-split-20261017.json)).
  - In the Adam phase the compiled jet costs 0.39× graph-only C17's time per evaluation, about 2.6× faster. Compiled nested AD is level (1.01×).
  - In the L-BFGS phase the compiled jet costs 0.83× C17's time.
- **Reading.** On this benchmark, a public Taylor-mode package compiled by `torch.compile` evaluates the gradient faster than the C17 contract in eager PyTorch.
  - Unlike 2D GL8, whose native pointwise kernels the ablation credits with 1.53–1.83×, C17 has no execution layer.
  - Whether a compiled or native C17 would regain the lead is untested.
- **Records.**
  - [audit-ext2-20261017.json](audit-ext2-20261017.json) and [read-ext2-20261017.json](read-ext2-20261017.json).
  - [release-source-map-ext2-20261017.json](release-source-map-ext2-20261017.json): all 11 published files are byte-identical to their executed copies.
- **Rule fixed now for any repair attempt of the compiled C17 variant.** The paper's abstract follows the latest registered decision, and §7.4 reports every batch, including this one.

## Compiled batch, attempt 2 (pre-registered 2026-10-17)

[src/protocol-ext3-20261017.json](src/protocol-ext3-20261017.json) was committed before the batch ran. It is identical to attempt 1 in arms, data, seeds, warm-up, selection rule, statistics and decision rule. Only the compiled C17 builder changes, to the route that survived capture in a [one-GPU diagnostic](diagnostics/README.md):
- **Builder.** [src/external2.py](src/external2.py) compiles a function that takes the six parameter tensors as explicit arguments. The external methods still use `external.build_compiled` unchanged.
- **Gate.** Formal cases of an arm whose warm-up failed on every lane are not started.
- **Attempt 1 stays reported.** The abstract follows the latest registered decision, and §7.4 reports both attempts.
- **Code.** [src/solve_ext3.py](src/solve_ext3.py), [src/run_ext3.py](src/run_ext3.py), [src/audit_ext3.py](src/audit_ext3.py) and [src/read_ext3.py](src/read_ext3.py) differ from their attempt-1 versions only in file names, the builder and the gate.
- **Self-test** (login node, CPU smoke). 24 warm-up and 36 formal cases exit 0, the gate is empty, the audit and the reader pass, and the archive (314 members) is verified.

### Result of attempt 2 (2026-10-17)

The batch ran once on one four-GPU H100 node in about 20 minutes. The gate was empty.
- **Acceptance.** All 36 solves are accepted.
- **Audit.** It passed 3,024 comparisons, with a maximum discrepancy of 1.8e-14. The registered reader exited 0.
- **Archive.** `5b4833b8…` (316 members, compiler caches excluded) was verified.
- **Selection.** The repaired compiled C17 qualified on every lane, and every method selected its compiled variant.

| Method | Warm-up replay, graph / compiled | Setup s (compiled) | Adam ms per evaluation (compiled, post hoc) |
| --- | --- | --- | --- |
| C17 | 2.55 / **1.20** ms | 3.5–4.0 | 1.15–1.34 |
| Collapsed jet | 3.48 / **0.86** ms | 10.7–10.9 | 0.99–1.00 |
| Nested AD | 8.54 / **2.41** ms | 25.6–26.5 | 2.32–2.56 |

| Compiled external / compiled C17 | Setup + optimization (primary) | Optimization only (secondary) |
| --- | --- | --- |
| Collapsed jet | **1.13** (0.99–1.20), 5/6: **C17 faster** | **0.96** (0.85–1.01), 5/6 favour the jet: **external faster** |
| Nested AD | **1.68** (1.44–1.90), 6/6: **C17 faster** | **1.17** (0.96–1.31), 5/6: **C17 faster** |

- **The two jet decisions differ because of setup.** Compiled C17 takes 3.5–4.0 s, because its explicit-argument builder reuses the compile cache across seeds. The jet builder captures the parameters as constants and recompiles part of its graph in every solve, taking 10.7–10.9 s.
- **Per evaluation, the compiled jet is about 5% faster** (0.95), and its Adam-phase evaluation is about 1.3× faster.
- **Compile factor.** Compiling makes C17's replay 2.1× faster and its evaluation 1.21× faster than graph-only C17.
- **Reading.** With all methods compiled, the contracted state gives no per-evaluation advantage over collapsed Taylor mode on this small benchmark. C17's whole-solve lead over it rests on cheaper setup. C17 has no native execution layer; a native C17 is untested.
- **Paper.** Under the rule fixed before this attempt, the abstract and conclusion now follow this decision. §7.4 reports both attempts and the uncompiled batch.
- **Records.**
  - [audit-ext3-20261017.json](audit-ext3-20261017.json), [read-ext3-20261017.json](read-ext3-20261017.json) and [ext3-phase-split-20261017.json](ext3-phase-split-20261017.json).
  - [release-source-map-ext3-20261017.json](release-source-map-ext3-20261017.json): all 12 published files are byte-identical to their executed copies.
