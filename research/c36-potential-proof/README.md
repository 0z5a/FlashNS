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

## Full native objective qualification

Use `--native-all` to include every canonical interior point in the fresh native
reference. `--native-interior-chunk` controls native coordinate AD memory use;
the successful H100 qualification uses chunks of 256. The default subset mode
and the historical result above keep their original scope.

```sh
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.verify_potential_objective_gpu --inputs data \
  --native-all --native-interior-chunk 256 --output outputs/gpu-objective-full-native
PYTHONPATH=src python -m box376.audit_potential_objective_gpu --inputs data \
  --proof outputs/gpu-objective-full-native --output outputs/gpu-objective-full-native-audit.json
```

The 2026-10-12 H100 run in
[gpu-objective-full-native-result.json](gpu-objective-full-native-result.json)
passes all 52 comparisons for two seeds and all 50,692 parameters per seed.
Fresh native AD now covers all 4,999 canonical points: 4,096 interior, 512
initial, 384 boundary and seven gauge points. Candidate partitions 128 and 64,
nonzero value-family diagnostic probes and caller-data mutation checks also pass.
The separate NumPy reader verifies all 140 saved arrays and reconstructs every
chunk gradient and normalized family contribution. The physical cache, model,
normalization and `atol=1e-11, rtol=1e-9` remain unchanged. Five CPU geometry
regression tests pass. The maximum error is 2.07227e-05 of the original
tolerance budget; peak PyTorch allocation/reservation is 37.761/37.867 GiB.
Science, audit and the Slurm step exit naturally with code zero. Successful
and failed-start archives are fully read and hash-verified on two existing MN5
allocations. Optimizer updates, training convergence, physical acceptance and
comparative performance remain unqualified.

## Frozen timing, training and physical acceptance protocol

[protocol-20261012.json](protocol-20261012.json) fixes the next experiment before
its GPU measurements. It adopts the existing 4,999-point source cache and fixed
normalization without changing the cache's historical development metadata.
Both original seeds start from fresh C36 curl-potential parameters. Direct
velocity-model checkpoints are incompatible with this representation.

The timing runner measures complete objective and gradient evaluations with
CUDA synchronization. Each baseline has five alternating AB/BA pairs per seed;
candidate and Taylor-AD windows last at least 60 seconds. Geometry setup is
reported separately. Native coordinate AD and parameter AD over C36 Taylor
states are distinct baselines: the latter shares the forward algebra and is
not an independent coordinate derivative oracle. No complete-solver speedup
is inferred from objective timings.

Training uses 4,000 Adam updates and 20 LBFGS calls per seed, with frozen
hyperparameters and a finite closure cap. Compute-only CUDA graph replay must
match eager and parameter-AD gradients at changed parameters before use.
Initial, scheduled and final complete gradients are checked; all LBFGS
candidates are retained. Adam checkpoints and optimizer state are saved every
100 updates. A finished budget is separate from the declared loss and gradient
convergence test.

```sh
PYTHONPATH=src python -m unittest box376.test_potential_ad_objective
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.potential_run --inputs data --protocol protocol-20261012.json \
  --mode timing --baseline taylor_parameter_ad --output outputs/timing-taylor
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.potential_run --inputs data --protocol protocol-20261012.json \
  --mode timing --baseline native_coordinate_ad --output outputs/timing-native
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.potential_run --inputs data --protocol protocol-20261012.json \
  --mode training --output outputs/training
PYTHONPATH=src python -m box376.audit_potential_run --inputs data \
  --protocol protocol-20261012.json --proof outputs/training --output outputs/training/audit.json
```

Physical acceptance observes the trained potential through native coordinate
AD, with a separate NumPy residual consumer. It includes independent stratified
and uniform-volume holdouts, seven time slices, initial/boundary values and
27 particles integrated with the observed network velocity. Two DOP853
refinements check trajectory integration. The original velocity, momentum,
divergence, pressure and endpoint limits remain unchanged. Time-slice and region
diagnostics are reported separately from the original combined gate. No exact
velocity is inserted into the model or trajectory right-hand side.

```sh
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.potential_physical --inputs data --protocol protocol-20261012.json \
  --checkpoint outputs/training/seed361901-final.pt --output outputs/physical-seed361901
PYTHONPATH=src python -m box376.audit_potential_run --inputs data \
  --protocol protocol-20261012.json --proof outputs/physical-seed361901 \
  --training outputs/training --output outputs/physical-seed361901/audit.json
```

Repeat physical acceptance for seed 361902. The NumPy-only reader reconstructs
saved gradient comparisons, family-normalized losses, paired statistics,
physical equations and acceptance decisions. It checks observed saved arrays;
it does not independently replay every Adam update or reintegrate particle
paths. GPU results for this protocol are pending; no speed, training convergence
or physical acceptance is claimed by these new entry points alone.

The prepared objective now constructs its affine transform and output-scale
constants on the device during setup. Its default eager equations are unchanged;
a CPU regression checks bitwise equality of the default and cached forward
states and complete parameter pullbacks. This avoids host-to-device constant
creation during CUDA graph capture. A failed capture is preserved and terminates
its process: PyTorch 2.4 can leave the allocator in capture mode, so an eager
fallback must use a fresh process and `--eager` with a new output directory.
The executed training packet's complete gradient and changed-parameter replay
checks pass on H100; optimization and physical acceptance are still in progress.

[protocol-graph-timing-20261012.json](protocol-graph-timing-20261012.json)
adds a separate timing scope in which both the explicit computation and the
C36 parameter-AD baseline use qualified CUDA graph replay. Parameter AD also
constructs its model constants during setup. Both captures are checked at the
original parameters and after a parameter change, with the same full objective,
FP64 precision and original derivative tolerance. Capture costs are reported
separately. Native coordinate AD remains an eager baseline. The eager protocol
and its raw measurements are preserved, including slowdowns.

```sh
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.potential_run --inputs data \
  --protocol protocol-graph-timing-20261012.json --mode timing --graphs \
  --baseline taylor_parameter_ad --output outputs/timing-matched-graphs
PYTHONPATH=src python -m box376.audit_potential_run --inputs data \
  --protocol protocol-graph-timing-20261012.json --proof outputs/timing-matched-graphs \
  --output outputs/timing-matched-graphs/audit.json
```

Use `--train-engine taylor_ad` to run the parameter-AD training baseline with
the identical fresh seeds, objective, Adam/LBFGS recipe and finite budgets.
Its scheduled gradient checks use the explicit implementation as the reference;
saved arrays identify the actual and reference engines. Run independent physical
acceptance for both baseline checkpoints as for the explicit checkpoints.
Complete-solver speed or time to accepted accuracy remains unset until both
implementations pass the same physical gates.

## Packed symmetric gathers

`torch_c36.unpack` now expands symmetric derivative channels with four cached
index gathers. This preserves the original forward values and transpose while
removing the chain of in-place `CopySlices` operations. The two CPU regression
tests compare the previous expansion's values and complete activation/product
input gradients, including noncontiguous and empty inputs.

[packed-gather-result-20261012.json](packed-gather-result-20261012.json) records
48 passing H100 comparisons across both original seeds, all 4,999 canonical
points and all 50,692 parameters. The preserved pre-optimization full native-AD
reference is identified by SHA-256; fresh native coordinate AD is not rerun by
this qualification. Both candidate chunk sizes, parameter AD, nonzero value
families, and original/changed-parameter graph replay are checked. The unchanged
`atol=1e-11, rtol=1e-9` budget passes a separate NumPy reader of all 96 arrays.
The raw arrays, source packet and exit receipts have verified copies on durable
MN5 storage and in the local workspace. The reported three-sample timings are
diagnostic; the full paired eager and matched-graph experiments are running.

```sh
PYTHONPATH=src python -m unittest test_c36_unpack
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m box376.verify_packed_gather --inputs data \
  --protocol protocol-20261012.json \
  --reference preserved-full-native/raw-proof.npz --output outputs/packed-gather
PYTHONPATH=src python -m box376.audit_packed_gather \
  --proof outputs/packed-gather --output outputs/packed-gather-audit.json
```

The earlier complete eager experiment at commit `73baa3b` is preserved in
[paired-eager-result-20261012.json](paired-eager-result-20261012.json): explicit
2.55670 seconds versus parameter AD 2.35870 seconds, paired speed ratio 0.92256
(95% bootstrap interval 0.92122–0.92419). This is an observed slowdown. Comparing
different source revisions' timings is historical evidence, not a paired
before/after optimization experiment. A separate native-coordinate-AD timing
experiment and identical-budget training/physical acceptance remain in progress.
