> Historical implementation notes. See [current validation](../../docs/validation.md) for the completed GPU and full-solver results.

# SM120 coordinate parameter contraction: G1

2026-09-12. Independent candidate based on the frozen first-fused-ready source.
All experimental hooks remain disabled by default. G0/G1 comparison enables
only `coordinate_wgrad`; both arms use the original forward GEMMs, complete
native stable activation VJP, torch tail, objective, and optimizer rules.

## Change and contract

`NativeCUDA.coordinate_wgrad(layout, coordinates, derivative)` consumes complete
materialized `bar_Z[rows,C]` and returns `dW[C,dimension]`, `db[C]`. Physical
Cartesian Q10/Q20 input jets have coordinate values at slot 0 and unit seeds at
slot `dimension-axis`. There is no factorial, extra normalization or averaging.

The two CUDA kernels reduce eight consecutive points per channel into
`partials[ceil(points/8),C,dimension+1]`, then merge tiles in increasing order.
Empty segments still write zero parameter gradients. Every structural-zero
product is retained so a NaN/Inf in an omitted input-jet slot has the appropriate
axis-specific effect on dW; db consumes only slot 0. This does not reproduce
arbitrary GEMM rounding, overflow or signed-zero order. FP64 and `--fmad=false`
are retained. There are no floating-point atomics or changes to the full VJP.

`PackedPINN(..., coordinate_wgrad=...)` takes its own fixed coordinate snapshot,
independently of the forward hooks. It dispatches only the layer-0 parameter
contraction, including a single affine output network and empty split segments.
The current objective remains 2-D NS; Q20 validation is operator-level only.

G1 continues to materialize bar_Z. A future G2 may feed the same reduction helper
from a full VJP in registers; G1 results cannot establish G2 correctness or its
potential traffic savings.

## Executed evidence

- CPU regression: **145 passed, zero skipped**, 12.38 seconds; CUDA devices hidden.
- The actual production C++ helper is compiled with GCC 13.3, no FP contraction.
  It passes 20 Q10/Q20 dense-BLAS comparisons, 96 individual NaN/Inf slot probes,
  output/partial canaries, 24 layout/empty/network variants with three actual
  Adam updates, and three independent nested-AD full-gradient comparisons.
- SM120 compile-only: nvcc 13.0.88, 3.515 seconds, `--fmad=false`.
  Partial kernel registers: Q10 52, Q20 70. Final merge: 40 for both dimensions.
  All four report zero stack and spill bytes. This is not runtime occupancy.
- GPU correctness, Graph replay, four Sanitizers, performance and solver
  convergence have not been executed for this candidate.

The evidence directory is `coordinate-wgrad-build-v1`; compile-only source
hashes include `coordinate_wgrad.cuh`. GPU builds also hash the new header.

## Allocated-window execution

After the resource owner assigns a physical SM120 GPU UUID and window:

```bash
FLASHNS_PYTHON=python \
bash scripts/run_sm120_coordinate_wgrad.sh NEW_OUTPUT GPU_UUID
```

The runner acquires the allocation owner's shared host lock plus the FlashNS
UUID lock, rejects existing GPU processes, builds fresh, runs original v8
preflight, four bounded Sanitizers, full candidate preflight, then screening.
Respect the assigned time window; the runner's stage timeouts are ceilings,
not a requested reservation duration.

Full preflight covers offset inputs/outputs/scratch, NaN-prefill coverage, Q10/Q20
empty/full/value/mixed layouts, nondefault stream, graph x/bar_Z updates, stable
saturated a1, and actual 2048+512-point full gradients/three Adam steps against
nested AD plus NumPy Adam. Screening compares complete compact loss/parameter
gradient Graph replay using A/A and G0/G1 APPA/PAAP blocks, both event and wall
timing, fixed parameter storage and source/binary/UUID/preflight hashes.
Kernel-only preflight cannot authorize timing. Optimizer/convergence are not
timed or established by that screening.
