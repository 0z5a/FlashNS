> Historical implementation notes. See [current validation](../../docs/validation.md) for the completed GPU and full-solver results.

# Isolated first-layer fusion candidate

This source tree was copied from `coordinate-ready/flashns`. It is an independent,
opt-in experiment; the earlier tail and coordinate candidates remain frozen.
The hypothesis is that avoiding the first hidden layer's materialized affine
tensor `Z` and one activation launch can reduce a complete gradient step. No GPU
bottleneck, speedup, or solver result has been established for this candidate.

The matched control is `NativeCUDA.coordinate_affine` followed by the original
native stable activation. Both arms call the same scalar coefficient helper in
`coordinate_jet.cuh`, preserving its axis accumulation order and IEEE propagation
of non-finite inputs through zero coefficients. Both require FP64, full Q10/Q20,
no fast math, and `--fmad=false`. The candidate calls the unchanged stable full-jet
activation, or its scalar value-only formula, while `Z` is private to one thread.
Both arms materialize `H` and the stable `a1` checkpoint for the original VJP.

The optional Python interfaces are:

```python
hidden, auxiliary = native.coordinate_affine_activation(layout, coordinates, weight, bias)
step = PackedPINN(
    ...,
    coordinate_affine=native.coordinate_affine,
    first_affine_activation=native.coordinate_affine_activation,
)
```

The hook applies only to layer zero when a hidden activation exists. A single
linear layer still uses the selected coordinate-affine path and has no activation.
The packed coordinate input for handwritten wgrad, every wgrad/bias reduction,
later affine layers, and the tail dispatch remain unchanged. The default hook is
`None`. Coordinates are still copied at setup so later caller mutation cannot
change the fixed problem.

## Validation entrances

The following HOST check uses only the Python standard library and a C++17
compiler. It compiles the actual shared coefficient/activation helper and tests
an independent closed-form jet and its parameter VJP through fourth-order tanh
derivatives, plus matched non-finite propagation:

```bash
python3 experiments/pinn_v8/validate_first_fused_host.py --output NEW_HOST_DIRECTORY
```

CPU dispatch/gradient/three-Adam-update and linear-output checks use the existing
PyTorch/pytest environment:

```bash
PYTHONPATH=src python -m pytest -q tests/test_first_fused.py tests/test_coordinate_affine.py tests/test_tail_dispatch.py
```

Only after the target GPU has been allocated, use the ordinary target-device
`build.py`, followed by `first_fused_preflight.py --build BUILD --output NEW.json`.
It checks Q10/Q20, empty and mixed segments, channel/block tails, offset inputs and
both outputs, non-default streams, non-finite propagation, stable saturated tails,
Graph input/parameter updates, and the complete problem's gradients/three Adam
updates against nested AD and NumPy. `--kernel-only` is bounded Sanitizer coverage
and is not a full correctness report for timing. The report identifies the source,
binary, GPU UUID, and `contract.candidate=coordinate_affine_activation`.

The separate `first_fused_screening.py` and `run_sm120_first_fused.sh` orchestrate
matched measurement only after full GPU correctness. They do not turn HOST or
compile-only success into a GPU claim.

## Evidence boundary

The local HOST shared-core check passed 32 finite cases with 256 basis seeds and
36 non-finite cases. Its maximum error/tolerance ratio was 4.321e-5; materialized
and fused HOST arithmetic agreed bitwise. These observations do not validate CUDA
launching, synchronization, GPU libdevice, register behavior, or performance.
CPU integration, SM120 compilation, Sanitizers, GPU correctness, and performance
must each be reported from their own execution evidence.
