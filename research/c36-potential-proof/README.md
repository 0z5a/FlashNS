# C36 potential gradient proof

This research snapshot compares the complete 50,692-parameter C36 gradient with independent native automatic differentiation. It uses two seeds and 17 fixed interior points from the full physical source cache, float64, and the unchanged `atol=1e-11`, `rtol=1e-9` budget.

Run from this directory with NumPy and PyTorch installed. First produce the CPU reference:

```sh
PYTHONPATH=src python -m box376.verify_potential_gradient --inputs data --output outputs/subset
```

On a machine with a CUDA-capable PyTorch build, expose one GPU and qualify the same model and data:

```sh
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.verify_potential_gpu --inputs data \
  --cpu-proof outputs/subset --output outputs/gpu
```

The GPU check runs geometry preprocessing, the explicit C36 forward and parameter pullback, and an independent native coordinate-AD reference on CUDA. TF32 is disabled. Both GPU implementations are also compared with the frozen CPU arrays. All 50,692 gradient entries are compared for each seed; the fixed inputs and initial parameters must match exactly. The complete proof has 22 comparisons and preserves 20 raw arrays.

Recompute those comparisons from the saved arrays using a separate NumPy-only reader:

```sh
PYTHONPATH=src python -m box376.audit_potential_gpu \
  --gpu-proof outputs/gpu --cpu-proof outputs/subset --output outputs/gpu-audit.json
```

Use new output paths for each run. Runtime environment activation must happen before setting `PYTHONPATH`. The scripts preserve raw arrays and failure traces. Stage timings include first-call overhead and are diagnostic only. This subset proof does not establish the complete 4096-point objective, training convergence, physical acceptance, or speed. CPU results are in [result.json](result.json). The H100 run in [gpu-result.json](gpu-result.json) passed all 22 comparisons and the separate NumPy audit of all 20 saved arrays. The largest error was 0.000433 of the unchanged tolerance budget; peak PyTorch allocated GPU memory was 3.403 GiB. Both seeds cover all 50,692 parameters (101,384 gradient entries in total). The final `proof.json` is authoritative; `progress.json` is a phase snapshot.
