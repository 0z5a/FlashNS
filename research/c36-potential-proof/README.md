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

The spatial gate uses explicit multiplication of its three axis factors. This
keeps high-order derivatives finite at near-zero cutoffs: source points 1965
and 2489 previously produced 15 nonfinite geometry derivatives. Two CPU
regression tests check finite derivatives and exact zeros in the flat time region.
The old 17-point CPU arrays are rechecked with the fixed implementation.

Geometry preprocessing uses batches of 256 to bound the derivative graph memory.
The expanded CUDA check covers all 4096 interior points with two candidate partitions
(chunk sizes 128 and 64), and independently compares 131 fixed interior points,
including the two cutoff regression points, with native coordinate automatic
differentiation (chunks 66 and 65). Both seeds retain
all 50,692 parameters and the original float64 error budget. Raw per-chunk losses
and complete gradients are saved. NumPy float64 sums the saved CUDA chunk
gradients for the partition and native-total comparisons; the independent reader
recomputes those sums.
Partition consistency uses the same candidate implementation and does not provide
an independent reference for the remaining interior points. Initial, boundary,
and gauge loss families, optimizer updates, and physical acceptance are outside
this check. The reported losses are weighted interior numerators with frozen
physical scales, before family normalization.

```sh
PYTHONPATH=src python -m unittest box376.test_potential_geometry
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.verify_potential_scale_gpu --inputs data --cpu-proof outputs/subset --output outputs/gpu-scale
PYTHONPATH=src python -m box376.audit_potential_scale_gpu \
  --inputs data --cpu-proof outputs/subset --proof outputs/gpu-scale --output outputs/gpu-scale-audit.json
```

The H100 run in [gpu-scale-result.json](gpu-scale-result.json) passed all 26 comparisons and the independent NumPy audit of 39 saved arrays. The largest error was 0.00164087 of the unchanged tolerance budget; peak allocated GPU memory was 20.484 GiB. Science and audit exited normally; their process IDs and GPU contexts were confirmed absent. Other workloads occupied the GPU at completion, so the device was not globally idle and no exclusive-GPU timing claim is made. The earlier 17-point CPU/CUDA result files retain their original evidence from commit `e7d1e1c733f8ef54b8a451a5b54eb5e800c90676`; the new scale result records the fixed source hashes.
