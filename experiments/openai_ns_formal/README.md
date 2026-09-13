# Official formal replay and finite local numerical adapter

See [scope, source pins, actual checker results and numerical validation](../../docs/openai-ns-formal-results.md).

`bootstrap.py`, `setup_comparator_tools.py`, `run_phase.py` and `unix_guard.py` support the unprivileged CPU build and the original Comparator challenges. Exact executed versions are retained separately from current source files. No upstream Lean source or challenge configuration was patched.

`projected_amplitude.py` implements a finite local complex amplitude RHS/pressure, an independent high-precision saddle-system reference and trajectory checks. `numerical_run.py` compares FP64 NumPy, tiled NumPy, eager GPU and compiled GPU, recording both resident and transfer-inclusive timing. It does not numerically implement the global construction.

Example in the recorded CUDA environment, using a fresh output filename:

```bash
NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/openai_ns_formal/numerical_run.py --output experiments/openai_ns_formal/artifacts/new-numerical.json --repeats 15
```

Full validation runs by default. `--validation-artifact` can reuse a passed numerical validation only when the operator source hash is unchanged; the exact previous validation source hashes and report hash are retained.
