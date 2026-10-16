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

## Status

Pre-registered (commit `170412b`); the GPU batch was submitted after this commit and is pending.
