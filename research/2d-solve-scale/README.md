# 2D solve-scale study (H100, pre-registered)

## Question

The [2D ablation](../2d-ablation/README.md) found a gap between gradient speed and solve speed at the frozen 2,048 interior points.
- Native GL8 (cell C) computes the full gradient 1.71× faster than compiled Q10 (cell A).
- A whole solve is not faster: optimization alone is 0.95×.
- Post hoc, the archived solves show why ([phase-split-2048-20261016.json](phase-split-2048-20261016.json), from `phase_split_2048.py`). The L-BFGS phase costs 6.3–7.2 ms per evaluation in every cell (A 6.38–6.61, B 6.37–6.59, C 6.32–7.21), against 0.17–0.28 ms of gradient replay. The Adam phase costs 0.25–0.39 ms per evaluation.

This study asks whether the gain returns once the gradient is a larger share of each evaluation, at 16,384 and 65,536 interior points.

## Design

[protocol-2d-solve-scale-20261016.json](protocol-2d-solve-scale-20261016.json) was committed before the batch ran.

- **Problem instances.** Each point count is a new problem instance with different collocation points. The network, optimizer program, validation sets and acceptance stay the same.
- **Arms.** Cells A (`q10_compact`, Inductor), B (`gl8_compact_output`, Inductor) and C (`gl8_native`) on the six formal initializations, one fresh process per solve. That makes 36 solves on one four-GPU H100 node.
- **Registered statistics.** Per size and pair (A/C, A/B, B/C), over jointly accepted seeds:
  - paired geometric means of setup plus optimization, optimization only, and optimization time per evaluation;
  - evaluation counts and acceptance per cell.
- **Decision.** A solve-level gain at a size requires an optimization-only geometric mean above 1.03 *and* at least 5 of 6 jointly accepted pairs favouring the faster cell. Fewer than 5 jointly accepted pairs is not decidable.
- **Predictions** (expectations, not gates): A/C optimization of about 1.4 at 16,384 and about 2.0 at 65,536 points.
- **Wrapper.** [`src/scale_solve.py`](src/scale_solve.py) runs `bench_gpu`'s unchanged solve path at another point count. Only the recorded `interior_count` field differs from the frozen `solve.py` template.
- **Admission.** The driver checks the pinned commit, the manifest, the six initialization hashes and the data hashes of both point counts.
- **Guards.** Fail-fast and deadline guards are carried over from the ablation driver.

## Results

The batch ran once on one four-GPU H100 node in 13 minutes.
- **Admission.** It verified the pinned commit, the 271-file manifest, the six initialization hashes and the data hashes of both sizes.
- **Completion.** All 36 solves completed and **all 36 pass acceptance**, with 5,918–8,977 evaluations.
- **Archive.** `2d-solve-scale-complete.tar.gz` (sha256 `430ee078…`, 118 members) was re-read and verified.

Registered statistics: paired geometric mean (range) of the first cell's time over the second's; "favour" counts jointly accepted pairs favouring the second cell.

| Points | Pair | Optimization only | Per evaluation | Setup + optimization | Favour | Registered decision |
| ---: | --- | --- | --- | --- | ---: | --- |
| 16,384 | A/C | **1.48** (1.31–1.80) | 1.43 (1.34–1.57) | 1.21 (0.62–2.14) | 6/6 | gain |
| 16,384 | A/B | 1.26 (1.12–1.43) | 1.25 (1.21–1.32) | 1.12 (0.53–1.73) | 6/6 | gain |
| 16,384 | B/C | 1.18 (1.02–1.28) | 1.14 (1.06–1.20) | 1.08 (0.36–2.70) | 6/6 | gain |
| 65,536 | A/C | **1.88** (1.75–2.05) | 1.95 (1.90–2.01) | 2.12 (1.80–2.49) | 6/6 | gain |
| 65,536 | A/B | 1.41 (1.31–1.55) | 1.46 (1.42–1.50) | 1.41 (1.26–1.60) | 6/6 | gain |
| 65,536 | B/C | 1.34 (1.24–1.46) | 1.34 (1.31–1.37) | 1.50 (1.43–1.58) | 6/6 | gain |

- **Predictions.** The protocol expected A/C optimization of about 1.4 at 16,384 and about 2.0 at 65,536 points; the observed values are 1.48 and 1.88. The gap between optimization-only and per-evaluation ratios is seed-to-seed variation in evaluation counts.
- **Phase cost (post hoc, [solve-scale-posthoc-20261016.json](solve-scale-posthoc-20261016.json)).** Each value is the time per evaluation minus the 2D ablation's registered replay median for the same cell and size.
  - Adam phase: 0.05–0.12 ms more than replay, so those replay medians carry over to these solves.
  - L-BFGS phase: **5.9–7.1 ms** more than replay, at 2,048, 16,384 and 65,536 points and in every cell. This cost lies outside the gradient, does not depend on the point count, and was not profiled further.
  - Relative to gradient replay, it is 22–43× at 2,048 points and 0.5–1.5× at 65,536 points.
  - So on H100 the solve-level gain is a matter of problem size: absent at 2,048 points, 1.48× and 1.88× at 16,384 and 65,536.
- **Setup depends on cache state (post hoc).** Each lane keeps its own Inductor cache, so setup depends on queue order.
  - A: the first solve of a cell and size on a lane took 22–23 s; later solves took 4.7–5.9 s.
  - B: 12–14 s first, 3.3–4.3 s later. Two B solves started together at the batch start took 47 s.
  - C: two native solves started at the batch start took 65 s and the rest 0.3–1.4 s; the cause was not investigated.
  - The same check on the ablation shows every 2,048-point Q10 solve ran after earlier tasks of its cell on its lane. Its 4.7–5.8 s setups are therefore warm-cache values, and the 2,048-point setup comparison is warm against warm.
  - Warm Q10 setup (4.7–5.9 s) is still slower than native (0.3–1.4 s); that is a real difference between the cells.
  - At 16,384 and 65,536 points, queue order decided which seeds paid a cold compile. This widens the per-seed setup-plus-optimization ranges and lifts the 65,536-point C/A value (2.12) above optimization alone (1.88). The registered decision uses optimization alone.
- **Scope.** H100 only; nothing is pooled with the H20 campaign. The H20 gain at 2,048 points (1.48 in optimization alone, with a compiled Q10 gradient of 3.27 ms) is consistent with this picture, but the H20 L-BFGS cost was not measured and the problem instances differ. This is one problem and one optimizer implementation.

## Files

- [solve-scale-result-20261016.json](solve-scale-result-20261016.json): registered reader output (`read_solve_scale.py`).
- [phase-split-2048-20261016.json](phase-split-2048-20261016.json): post-hoc Adam/L-BFGS split of the 18 archived 2,048-point ablation solves (`phase_split_2048.py`).
- [solve-scale-posthoc-20261016.json](solve-scale-posthoc-20261016.json): post-hoc phase cost against replay and the setup cache-state check (`posthoc_solve_scale.py`).
- [release-source-map-20261016.json](release-source-map-20261016.json): byte identity of every published file with its executed copy, plus the archive digest.
- `make_solve_scale_plan.py`, `run_solve_scale.py`, `entry_solve_scale.sh`, `src/scale_solve.py`: plan builder, cluster-side driver, batch start command, solve wrapper.
