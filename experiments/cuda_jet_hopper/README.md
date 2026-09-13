# Controlled FP64 Hopper experiment

See the [results and reproduction commands](../../docs/cuda-hopper-results.md).

`build.py` emits a predeclared 48-configuration grid. `validate.py --exhaustive` checks the local and full-gradient contracts. `check_sanitizers.py` checks the same binary hashes with all four Compute Sanitizer modes and records the hardware-counter permission probe and SASS. `run_benchmark.py` selects a TMA configuration and independently retests its matched synchronous/cp.async controls.

The `hopper.py` binding exposes U/F and explicit fallbacks. It is an experimental backend; the measured largest 3D case does not justify replacing the previous fused backend for every shape.
