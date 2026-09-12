> Historical implementation notes. See [current validation](../../docs/validation.md) for the completed GPU and full-solver results.

# First activation VJP + coordinate parameter contraction: G2

2026-09-12. Independent candidate based on frozen coordinate-wgrad-ready (G1).
The implementation is opt-in and leaves earlier frozen candidates untouched.

## Implemented boundary

`NativeCUDA.coordinate_activation_wgrad(layout,x,H,a1,bar_H)` evaluates the
existing complete stable Q10/Q20 VJP inside each eight-point tile, passes all
bar_Z components to G1's unchanged coordinate contraction helper, and writes
the same partials. Both paths use the same final merge kernel. G2 removes the
first hidden layer's global bar_Z output and one activation-kernel launch.
The upstream dgrad GEMM and its complete bar_H output remain unchanged.

`PackedPINN(..., first_activation_wgrad=...)` invokes the fused terminal at the
reverse transition from layer 1 to layer 0, accumulates dW0/db0 in the original
segment order, and ends that segment's backward pass. It owns a fixed input
coordinate snapshot independently of the other hooks. A single affine network
uses its ordinary parameter-contraction path; it has no activation to fuse.
For a one-hidden-layer network, this terminal hook takes precedence over the
tail VJP hook at their shared activation boundary. The G1/G2 screening keeps
tail mode `torch` in both arms and enables no forward specialization.

There is no low-order-only VJP or H-only derivative reconstruction. Complete
high-order contributions and structural-zero NaN/Inf propagation are retained.
Stable a1 remains a required checkpoint. Only parameter first VJP is supported;
coordinate gradients, HVP and double backward are outside this private path.

## Executed checks

- **204 CPU tests passed, zero skipped**, 15.74 seconds, CUDA devices hidden.
- New host tests compile the actual production G2 header and full stable VJP.
  Sixteen Q10/Q20 shape/channel cases compare G2 against an explicitly
  materialized G1 control with identical finite bits, then against independent
  tensor VJP + dense coordinate GEMM at atol=2e-12, rtol=2e-11.
- 192 hidden/seed NaN/Inf slot probes, six auxiliary probes, two all-finite-input
  internal-overflow probes, and twelve saturated-center cases pass. Centers
  ±20/±100/±350 retain positive a1 and positive bias-gradient tails.
- Thirty-six layout/empty/depth variants compare complete gradients and three
  real Adam updates. Three layouts additionally compare with nested AD.
- sm_120 compile-only passes with nvcc 13.0.88, FP64, fmad=false, 3.846 seconds.
  G2 partial-kernel registers: Q10 72, Q20 122; zero stack/spill bytes. G1's
  coordinate contraction header is byte-identical to the frozen G1 version.
- No G2 GPU execution, Sanitizer, runtime occupancy, performance or solver
  convergence result has been produced.

CPU finite-bit agreement is limited to the exercised host implementation and
does not establish GPU finite-bit agreement. The GPU preflight explicitly
checks that condition before allowing screening.

## GPU materials

`first_vjp_wgrad_preflight.py` checks G1/G2 raw kernels, canaries and full writes,
nondefault stream, mutable Graph inputs, full VJP nonfinite/overflow behavior,
stable tails, and all-layout 2048+512-point full gradients plus three Adam
updates against nested AD and NumPy Adam. Q20 coverage is operator-level; the
full NS objective is still 2-D Q10.

`first_vjp_wgrad_screening.py` compares complete compact loss/parameter-gradient
Graph replay with shared read-only parameters, separate Graph pools, A/A and
G1/G2 balanced APPA/PAAP blocks, event/wall measurements and drift checks. The
full preflight source/binary/UUID must match. Kernel-only checks cannot authorize
timing. Screening does not time the optimizer or establish convergence.

After explicit physical GPU UUID/window allocation, use the finite runner:

```bash
FLASHNS_PYTHON=python \
bash scripts/run_sm120_first_vjp_wgrad.sh NEW_OUTPUT GPU_UUID
```

The runner acquires the shared host execution lock and UUID lock, builds fresh,
runs original v8 preflight, four bounded Sanitizers, full G2 preflight and then
paired screening. Respect the allocation window; timeout ceilings are not a
reservation request. Preserve the original tail candidate's queue priority.

At Ni=2048, Nb=512, C=64, Q10, the removed bar_Z allocation is 10,747,904 logical
bytes, with 21,495,808 logical write/read bytes across that boundary. The same
491,520-byte partial buffer is used by G1 and G2. These counts do not predict
actual DRAM traffic, peak memory or speed. G2 uses more registers per thread;
its eight-point serial work and the final small reduction may outweigh the
removed materialization. Only GPU evidence can resolve that tradeoff.
