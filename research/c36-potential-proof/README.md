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

`prepare_geometry(points, eta=1/32, *, chunk_size=256)` now computes the affine
spatial feature derivatives directly and bounds native gate differentiation by
`chunk_size`. It preallocates the returned C36 arrays, preserves point order,
dtype and device, and returns detached tensors. Exactly flat time regions have
zero gate states and require no gate derivative graph. Empty Nx4 inputs are
supported; malformed coordinates, invalid chunk sizes and nonfinite/nonpositive
cutoff widths are rejected. Total output storage still grows with the point
count; the chunk bound applies to the derivative graph. Five CPU tests cover
coordinate-AD agreement, flat cutoffs, uneven chunks, noncontiguous input views,
input preservation, empty inputs and invalid arguments.

The proof in [gpu-geometry-result.json](gpu-geometry-result.json) checks all 4096
geometry rows against the frozen native geometry from commit `7679cb6`, performs
fresh native coordinate differentiation at 131 points, rechecks the original 17
CPU anchors, and tests a different geometry partition. Both complete parameter
gradients and the full interior states/losses are recomputed in CUDA and compared
with the previous 128-point-partition arrays. Native parameter gradients are not
rerun in this check. Use the original reference packet identified by
`reference_provenance.raw_sha256`; regenerating the scale packet using the new
geometry implementation does not reproduce that independent full reference.

```sh
PYTHONPATH=src python -m unittest box376.test_potential_geometry
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.verify_geometry_preparation_gpu --inputs data \
  --cpu-proof outputs/subset --scale-proof outputs/gpu-scale-7679cb6 \
  --output outputs/gpu-geometry
PYTHONPATH=src python -m box376.audit_geometry_preparation_gpu --inputs data \
  --cpu-proof outputs/subset --scale-proof outputs/gpu-scale-7679cb6 \
  --proof outputs/gpu-geometry --output outputs/gpu-geometry-audit.json
```

All 18 comparisons passed; the separate NumPy reader verified all 16 saved arrays. The maximum error was 0.0010071 of the original tolerance budget. Geometry-only peak PyTorch allocation was 0.618 GiB and full-proof peak was 1.766 GiB. The observed 4096-point setup call took 23.940 seconds; this is a diagnostic measurement, not a paired speed comparison. The scientific worker and reader exited normally and released their GPU context. Historical result files preserve the exact source hashes used in their runs.

`PreparedPotentialNormalizedObjective(data, calibration, geometry_chunk=256)`
prepares the C36 geometry and physical normalization for all four loss families.
It clones and detaches source tensors and copies the source metadata. Later
changes to the caller's coordinates, weights, forcing, targets, denominators or
calibration key cannot invalidate the prepared geometry. Calling the object with
`(parameters, chunk)` returns a detached normalized loss, the complete flat
parameter gradient, and four raw weighted family numerators. The
`contributions(parameters, chunk)` iterator exposes each family index, its raw
chunk numerator and its normalized gradient contribution. Global family
denominators and coefficients are applied to each contribution; a subset retains
the original denominators when it represents part of the canonical objective.
This interface does not install an autograd backward or perform optimizer steps.

The CUDA objective check evaluates the canonical 4,999 points (4,096 interior,
512 initial, 384 boundary and 7 gauge points) with candidate chunks of 128 and 64.
It compares fresh native parameter gradients at 131 interior points and all 903
value-family points, with the unchanged physical source and normalization.
The remaining interior points have candidate partition consistency checks.
The compact potential gate vanishes at canonical initial, boundary and gauge
points. A separate diagnostic challenge therefore assigns 17 active interior
coordinates to each family, with nonzero targets and different nonunit global
weight sums. Those coordinates are synthetic value-loss probes, not new physical
initial or boundary conditions; the canonical cache is unchanged. The challenge
checks nonzero gradients in every family against native AD and verifies that
mutating caller data leaves the prepared objective unchanged. Both seeds retain
all 50,692 parameters, float64 and the original derivative comparison budget.

```sh
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.verify_potential_objective_gpu --inputs data --output outputs/gpu-objective
PYTHONPATH=src python -m box376.audit_potential_objective_gpu --inputs data \
  --proof outputs/gpu-objective --output outputs/gpu-objective-audit.json
```

The NumPy reader reconstructs normalized losses, gradients, family numerators
and family gradients from saved chunk arrays, verifies the original input cache,
and repeats all native and partition comparisons. Timings are diagnostic setup
and evaluation calls. This proof does not establish a full independent native
reference for all 4,096 interior points, optimizer behavior, training convergence,
physical acceptance or comparative performance.

The H100 run in [gpu-objective-result.json](gpu-objective-result.json) passed all 52 comparisons and five CPU geometry regression tests. The independent NumPy reader verified all 140 saved arrays. The maximum error was 2.07227e-05 of the original tolerance budget and peak PyTorch allocation was 10.797 GiB. Science, audit and the Slurm step exited normally, and the worker processes and their GPU contexts were confirmed absent. The raw archive was fully read and hash-verified on MN5 and VPS.
