# C36 potential gradient proof

The mathematical source of this three-dimensional problem is OpenAI Math376,
*Incompressible Box Transport and Finite Computation*, pinned at commit
`adc7f1241b42e322a6451854ab7e4b4c146bf78a`. The [original geometry source](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Incompressible-Box-Transport-and-Finite-Computation-September-27-2026/build/geometry.tex)
constructs smooth, compactly supported incompressible motion for prescribed
volume-preserving affine box schedules. This experiment instantiates one box
in the unit cube, `t in [0,1]`, `nu=0.01`, `eta=1/32`, and zero reference
pressure. With `y=x-c(t)` and `L=D_dot D_inverse`, the potential is
`A=chi * (c_dot cross y / 2 + (L y) cross y / 3)`, the exact velocity is
`U=curl A`, and the fixed forcing is `f=U_t+(U dot grad)U-nu Laplacian(U)`.
The implementation includes the cutoff contribution `grad(chi) cross B`
and the spatial derivative of the fixed force. The network learns its own
potential and pressure. This is a constructed forced-flow instance with stated
conditions. Its numerical results do not establish a general solution formula
for arbitrary Navier--Stokes data or a regularity theorem. The original
mathematical-source hash, implementation hash and frozen-data linkage are
recorded in [math376-provenance.json](math376-provenance.json).

Two properties of this instance matter for the diagnostics. The motion is
confined to `t in (1/4, 3/4)`; at every other time `U`, `f` and their
derivatives vanish identically. During the motion the box plateau
(`chi = 1`) carries the affine velocity `U = c_dot + L y` with zero
Laplacian, and because `L` is diagonal its force `f = c_ddot + (L_dot + L^2) y`
is a gradient; the rotational part of the forcing lies in the cutoff layers.
The CPU check `PYTHONPATH=src python -m box376.check_source_structure --output
source-structure-check-20261018.json` confirms both to rounding on the frozen
source. Its output is [source-structure-check-20261018.json](source-structure-check-20261018.json).
It also records that only 0.89% of uniform space-time points carry a nonzero
exact velocity. It reruns no training or observation.

The post-hoc tool `tools/c36_pressure_structure.py` decomposes the stratified
corrected-pressure error of saved observations. It reproduces every recorded
regional and global value exactly. For the four v5 finals
([c36-pressure-structure-20261018.json](c36-pressure-structure-20261018.json)):
- **Plateau.** Within the motion window the error is 4.4–6.6× the limit, 27–70% of it a per-time offset.
- **Cutoff layers.** They carry 1.2–3.4× the limit. On their own, with the plateau error set to zero, they would leave 1.5–1.9× the limit, so plateau-only weighting or sampling cannot pass.
- **A wider cutoff.** At `eta = 1/16` the layer force RMS is four to five times smaller and the force-gradient RMS nine to eleven times smaller.

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

The H100 run in [gpu-objective-result.json](gpu-objective-result.json) passed all 52 comparisons and five CPU geometry regression tests. The independent NumPy reader verified all 140 saved arrays. The maximum error was 2.07227e-05 of the original tolerance budget and peak PyTorch allocation was 10.797 GiB. Science, audit and the Slurm step exited normally, and the worker processes and their GPU contexts were confirmed absent. The raw archive was fully read and hash-verified on the experiment system and an independent host.

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
and failed-start archives are fully read and hash-verified on two existing the experiment system
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
paths. The completed H100 training and physical-acceptance results for this
protocol are reported below: both engines finish the frozen budget for both
seeds and all four attempts fail the convergence and physical gates, so no
solve speed or time to accepted accuracy is claimed.

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
the experiment system storage and in the local workspace. The reported three-sample timings are
diagnostic. The complete paired timings at source commit `ae51ca6` are recorded
in [paired-packed-gather-result-20261012.json](paired-packed-gather-result-20261012.json).
Each mode has ten alternating pairs, two original seeds and windows of at least
60 seconds; setup and capture are excluded and reported separately.

| Execution of both engines | Explicit seconds/evaluation | Parameter AD seconds/evaluation | Paired baseline/explicit ratio (95% bootstrap interval) |
| --- | ---: | ---: | --- |
| Eager | 1.73958 | 1.69253 | 0.97293 (0.97047–0.97592) |
| CUDA graph replay | 0.48423 | 0.46955 | 0.96968 (0.96932–0.97008) |

The explicit implementation remains 2.78% slower in eager and 3.13% slower in
matched graph replay. Both GPU runs and NumPy readers exit normally; all 36
gradient comparisons and paired statistics also pass local NumPy rereading of
the copied raw arrays. Both timing archives have verified the experiment system/local copies.
The earlier native-coordinate-AD experiment is now complete at source `73baa3b`.
[paired-native-coordinate-result-20261013.json](paired-native-coordinate-result-20261013.json)
preserves all ten alternating pairs: explicit 2.77727 seconds versus native
3,484.28352 seconds, ratio 1,254.84 (95% bootstrap interval 1,211.34–1,291.26).
The original native-specific window rule uses one complete native evaluation;
candidate windows last at least 60 seconds. This measures the original recursive
native derivative implementation; the later cache lifetime repair is outside
that experiment. All gradient/statistic audits pass, and the complete archive
has 57 verified members on the experiment system and locally. The original controller's exit 1
retains the earlier pre-update training failure; timing science, reader and
timing driver each exit 0. These measurements remain separate from the newer
packed-gather/Taylor-AD comparison and complete-solver accuracy.

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
before/after optimization experiment. The separate native-coordinate-AD timing
experiment and original finite-budget training are complete.

The native derivative oracle now builds cached derivative prefixes iteratively.
The recursive closure previously retained itself, its cache and native graphs
until cyclic garbage collection. Prefix order, native differentiation and
parameter graphs are preserved. Two CPU tests verify bitwise C17/C36 states and
parameter gradients, and immediate closure release with cyclic collection
disabled. [native-oracle-cache-result-20261012.json](native-oracle-cache-result-20261012.json)
also verifies bitwise native states and all 50,692 parameter gradients at two
points from a completed trained checkpoint; a separate NumPy reader rechecks
the saved arrays. This does not qualify a new GPU speed ratio.

Physical observation also accepts `--device cpu --observer-chunk 128`. CUDA and
128-point chunks remain the defaults. This permits CPU observation of a verified
GPU-trained checkpoint with the same holdouts, native derivatives, NumPy
equations, particle integration and acceptance gates. CPU observation timings
are separate from GPU performance measurements.

[training-finite-budget-result-20261012.json](training-finite-budget-result-20261012.json)
records the complete original explicit budgets at commit `7281fae`: 4,000 Adam
updates and 20 LBFGS calls per seed, with 444/438 closures. Both fail convergence:
losses are 5.40207e-4 and 5.16776e-4; gradient infinity norms are 7.50714e-4 and
1.28521e-3. The limits remain 1e-4 and 1e-3. All 48 scheduled saved-gradient
comparisons pass the NumPy audit. The complete 1.54 GB training archive, including
all candidates/checkpoints, has been member-verified on durable the experiment system storage.
Both final checkpoints also have verified local copies.

Native-AD CPU value observations cover both complete 4,096-point holdouts for
each seed. Stratified/uniform velocity relative errors are 2.053%/4.374% and
1.881%/3.461%, above the original 1% limit; corrected pressure also exceeds its
original limit. Separate NumPy readers recheck the saved value metrics. These
value observations exclude PDE derivatives and trajectories. GPU physical
observation exhausted memory with the old recursive derivative cache; its failure
and source are retained. The cache lifetime repair passed H100 qualification of
the native state and all 50,692 parameter gradients at the trained CPU anchors.
Sixteen native observations with cyclic collection disabled retained exactly
67,514,880 allocated bytes after each observation.

[physical-finite-budget-result-20261013.json](physical-finite-budget-result-20261013.json)
records both complete GPU physical observations, all original holdouts and both
DOP853 trajectory refinements. The original physical gates fail for both seeds.
Velocity relative errors remain 2.053%/4.374% and 1.881%/3.461%; stratified
momentum RMS is 0.01422/0.01386 against 0.01. Maximum particle endpoint errors
are 0.02317/0.02980 against 0.001. Refinement changes are below 7.3e-11 against
1e-7, while initial/boundary velocity and pressure gauge errors are zero.
Both remote and local NumPy readers pass all 117 comparisons per seed. The
17.7 MB archive has 126 member-verified files on the experiment system and locally. Readers check
the analytic source independently at the original tolerance and use the sealed
force bytes for cancellation-sensitive residuals. They do not reintegrate the
ODE or replay optimizer trajectories.

The identical-budget parameter-AD baseline also completed both 4,000-Adam /
20-LBFGS budgets. Its final losses are 5.38698e-4 / 5.15322e-4 and gradient
infinity norms 5.90443e-4 / 2.06306e-3; both seeds fail convergence. All saved
gradient comparisons pass. Its complete 1.55 GB training archive is verified on
the experiment system. Physical observation completed from its original verified checkpoints after
the same old-cache OOM, with both original trajectory refinements and all holdouts.
Both seeds fail the original physical gates. Velocity relative errors are
2.033%/4.313% and 1.850%/3.415%; endpoint errors 0.02276/0.02998. Remote and local
NumPy readers each pass 117 checks per seed. The 15.3 MB archive has 123 verified
members on the experiment system and locally. Both training engines remain unaccepted at the
original finite budget. No complete-solver speed or time to accepted accuracy
is established.

## Opt-in packed activation candidate

`torch_c36_packed.activate_packed` computes the same tanh identities in symmetric
storage: ten third-derivative and six Hessian entries rather than dense 27/9
entries. The default activation remains unchanged. Select the candidate with
`PreparedPotentialNormalizedObjective(data, calibration, activation=activate_packed)`;
the chosen activation is also retained for the parameter transpose.

[packed-activation-cpu-result-20261013.json](packed-activation-cpu-result-20261013.json)
checks both initial seeds, changed parameters and both original trained final
checkpoints. All 50,692 gradient entries pass against the previous explicit
implementation and parameter AD on small canonical subsets, with two partitions
and the original tolerance. A separate NumPy reader passes 72 comparisons.
CPU tests also cover all derivative-state transposes, noncontiguous and empty
inputs, and zero/saturated activation values. A local single-activation/transpose
diagnostic took 0.01510 seconds versus 0.02924 seconds for dense storage. This
is a CPU kernel diagnostic; full 4,999-point GPU qualification and formal paired
performance have not run, and no new GPU acceleration claim is made.

The CPU candidate now also passes all 4,999 canonical points and complete
50,692-parameter gradients against preserved full native-GPU anchors for both
seeds, with 128/64 partitions and changed parameters. A separate NumPy reader
passes 36 comparisons of 72 arrays; maximum error is 9.18e-6 of the unchanged
tolerance budget. Original initial checkpoint bytes are used directly because
regeneration under another PyTorch build differed by one last bit. Native
coordinate AD and CUDA graphs are not rerun in this CPU check. Single CPU calls
took 3.89/3.73 seconds for packed storage, 6.36/6.21 for dense storage and
5.14/5.13 for parameter AD. These are diagnostics, not formal GPU paired results.

`torch_c36_packed_vjp.activate_packed_vjp` adds an opt-in analytic transpose for
the nonlinear state. It reverses packed contractions and differentiates the
actual stable coefficient program, including exp-based coefficients, tanh
rounding and the absolute-value origin. Saturated inputs with cotangents scaled
by 1e8 pass the original comparison budget. Select it with
`activation=activate_packed, activation_vjp=activate_packed_vjp`; default
nonlinear pullbacks continue using local state autograd. The parameter-AD
baseline also accepts `activation=activate_packed` so equivalent forward
optimizations can be compared on both engines.

[analytic-transpose-cpu-result-20261013.json](analytic-transpose-cpu-result-20261013.json)
records 42 passing full canonical comparisons for the analytic candidate,
packed/compiled parameter AD and changed parameters, plus independent NumPy
audits of all 84 arrays. Compilation is opt-in through `torch.compile` on the
selected forward and transpose. These CPU results use PyTorch 2.14.1; CUDA and
the cluster's PyTorch 2.4 runtime remain unqualified for these new variants.

[paired-analytic-vs-compiled-ad-cpu-result-20261013.json](paired-analytic-vs-compiled-ad-cpu-result-20261013.json)
records ten alternating CPU pairs, both original initial checkpoints, all
4,999 points and 50,692 gradients. Both engines share compiled packed forward;
the candidate uses the analytic transpose and the baseline uses parameter AD.
Every window has at least three evaluations and 60 seconds. Setup/compilation
and warmup are excluded and recorded separately. The ratio is 1.26509 with
95% paired bootstrap interval 1.20315–1.32790. NumPy independently rechecks all
24 initial/final gradient arrays and the complete pairing/statistics. This is
a CPU objective/gradient measurement; GPU and accepted-solver speedups are
not established.

```sh
PYTHONPATH=src python -m unittest test_c36_packed box376.test_potential_audit
PYTHONPATH=src python -m box376.verify_packed_activation_cpu --inputs data \
  --protocol protocol-20261012.json --training original-training \
  --output outputs/packed-activation-cpu
PYTHONPATH=src python -m box376.audit_packed_activation_cpu \
  --proof outputs/packed-activation-cpu --output outputs/packed-activation-cpu-audit.json
PYTHONPATH=src python -m box376.verify_packed_activation_full_cpu --inputs data \
  --protocol protocol-20261012.json --training original-training \
  --gather-proof preserved-packed-gather/qualification --activation-mode analytic \
  --output outputs/analytic-full-cpu
PYTHONPATH=src python -m box376.audit_packed_activation_full_cpu \
  --proof outputs/analytic-full-cpu --output outputs/analytic-full-cpu-audit.json
```

## Opt-in analytic activation on H100

`box376.packed_analytic_gpu_run` freezes the full canonical objective, original stored parameter bytes, unchanged error budget and two seeds. Before measuring ten alternating pairs (each arm at least 60 seconds and three evaluations), it checks original/changed parameters in direct calls and CUDA Graph replay against the preserved full native reference and parameter AD. Both measured arms share the same packed forward; the candidate uses the manual analytic activation transpose.

Use `--activation-mode eager` for the explicit packed forward plus CUDA Graphs, or `--activation-mode compiled` to compile both forward arms and the candidate activation transpose. Compiled mode requires a compatible Triton installation; compiler failures are preserved and do not silently select eager mode. `box376.audit_packed_analytic_gpu` is a separate NumPy-only reader for saved gradients, native anchors, window durations and paired bootstrap statistics. These entry points do not run training or establish complete-solver acceleration. The completed [eager packed-forward graph result](paired-analytic-vs-packed-ad-gpu-result-20261013.json) passes 78 numerical checks and both remote/local NumPy audits of 156 arrays and 20 timing windows. Its ten-pair ratio is 1.044582×, with 95% interval [1.044130, 1.045051]; candidate/control means are 0.328305/0.342941 seconds. The complete 106-member archive is verified on the experiment system and locally. This matched-forward graph result is separate from the older dense-activation timings and from compiled GPU performance.

The alpha Torch 2.4 runtime passes legacy tuple targets to Triton. The opt-in `--legacy-triton-target` adapter translates only that target representation to Triton 3.0's device target, checking architecture and warp size against the actual CUDA device. Source, dtype and compiler options stay unchanged; both arms use the same per-process adapter. The [small GPU compiler fixture](compiled-activation-GPU-smoke-result-20261013.json) passes forward and activation-transpose checks at saturated and near-zero values with large cotangents.

The subsequent [complete compiled H100 experiment](paired-analytic-vs-compiled-ad-gpu-result-20261013.json) passes all 78 numerical comparisons and independent remote/local NumPy readers of 156 arrays and 20 timing windows. Both arms compile the same packed forward; the candidate also compiles its analytic activation transpose. Ten alternating pairs give **1.323962×** (95% paired interval [1.323087, 1.324762]), with candidate/control window means 0.164484/0.217771 seconds. All 114 members of the complete 161,322,508-byte archive are verified on the experiment system and locally. Science, readers and the actual Slurm step exit zero, owned children are joined and the original GPU lock is released. [Earlier compiler preflight failures](compiler-preflight-negative-results-20261013.json) retain their original failed status and fully verified cluster archives. These timings establish objective/gradient throughput, with no additional training or physical acceptance claim.

## Exact zero-gate constant folding

`box376.fold_zero_gate.ZeroGateFoldedObjective` is an opt-in wrapper around either prepared Taylor-state engine. It evaluates the network only where at least one of the 36 gate channels is nonzero, using exact equality without a threshold. For every omitted network row, it retains the parameter-independent forcing, forcing-gradient or target loss at the original weight. Every family keeps its original full weight sum, normalization and coefficient. The original prepared engine remains usable; the default path is unchanged. Native-coordinate AD cannot select this wrapper.

The [CPU qualification](zero-gate-folding-cpu-result-20261013.json) passes 12 full-objective/numerator/all-parameter-gradient comparisons against the preserved native H100 anchors, covering both original stored initializations. The NumPy-only reader verifies all 24 objective arrays and 15 folding arrays. Of the 4,999 canonical rows, 2,933 have active jets; the other 2,066 are represented by their constant loss. Four regression cases cover nonzero forcing/targets, changed parameters, empty active families, and a zero gate value with a nonzero derivative as small as `1e-30`. The same regressions also pass in the actual experiment runtime before GPU admission. The complete CPU archive is fully verified locally. This is the numerical CPU qualification; the GPU measurement follows below.

To qualify and measure both equally folded compiled-forward arms, add `--fold-zero-gates` to `box376.packed_analytic_gpu_run`. Pass `--inputs` to the NumPy reader so it can verify the original family weight sums and physical normalization:

```bash
PYTHONPATH=src python -m box376.packed_analytic_gpu_run \
  --inputs data --protocol protocol-graph-timing-20261012.json \
  --reference preserved-full-native-reference.npz \
  --activation-mode compiled --fold-zero-gates --output outputs/folded-GPU
PYTHONPATH=src python -m box376.audit_packed_analytic_gpu \
  --inputs data --protocol protocol-graph-timing-20261012.json \
  --reference preserved-full-native-reference.npz \
  --proof outputs/folded-GPU --output outputs/folded-GPU-audit.json
```

On the legacy legacy compiler runtime, select `--legacy-triton-target` explicitly for the GPU worker as described above. Use a new output directory for every attempt. Both arms use folding and the same packed forward; the complete objective, stored parameter bytes and original numerical/timing protocol are preserved.

### Folded H100 timing (2026-10-16, pre-registered)

[protocol-zero-gate-gpu-20261016.json](protocol-zero-gate-gpu-20261016.json) was committed before the batch ran.
- **Design.** One exclusive H100 node; only one GPU is used, for timing isolation. The unchanged runner and graph-timing protocol run twice, one after the other, on the same GPU:
  - first, the folded comparison: analytic candidate against parameter-AD control, both folded and compiled;
  - then, an unfolded repeat.
- **Driver.** `box376.run_zero_gate_gpu` runs the batch once, with an embedded start command. The reader is `box376.read_zero_gate_gpu`.
- **Inputs.** The source is a git archive of the pre-registration commit. The data and the frozen native reference are the unchanged 2026-10-13 inputs. The batch prepared on 2026-10-13 was cancelled while pending and never ran.

| Run | Analytic s | Parameter AD s | Ratio | 95% interval |
| --- | ---: | ---: | ---: | --- |
| Unfolded repeat | 0.166597 | 0.220791 | 1.325303 | 1.324919–1.325678 |
| Folded (both arms) | **0.099377** | 0.139132 | **1.400047** | 1.399911–1.400178 |

- **Checks.**
  - Both runs pass all 78 full objective/numerator/gradient comparisons; the largest error is 3.0e-6 of the original budget.
  - The NumPy reader checks 20 windows.
  - The folding reader confirms the 2,933 active rows and that every original row is accounted for. The folded rows' constant loss is zero for this instance, and normalization is unchanged.
- **Registered classification.** The folded interval's lower bound exceeds 1.03, so the outcome is **analytic advantage under folding**.
- **Replication.** The unfolded repeat gives 1.325, against 1.324 on 2026-10-13.
- **Folding factor (descriptive).** Folding shortens an evaluation by 1.68× for the analytic candidate and 1.59× for parameter AD; cost proportional to rows would give 1.70 (1.74 counting 128-row chunks).
  - Parameter AD gains less, which suggests it carries more cost that does not scale with rows.
  - A folded analytic evaluation takes 99.4 ms, against 220.8 ms for unfolded parameter AD.
- **Scope.** Objective/gradient throughput only. No training, solver or physical claim is made. The gain depends on this instance's gate geometry.
- **Rerunning.** The public driver reads the node-local cache root from `NODE_LOCAL_SCRATCH` and the cache prefix from `USER`; set both before starting it.
- **Records.**
  - [zero-gate-folding-gpu-result-20261016.json](zero-gate-folding-gpu-result-20261016.json): reader output.
  - [release-source-map-zero-gate-20261016.json](release-source-map-zero-gate-20261016.json): 12 executed files byte-identical to the public files. The archive is `90ab9782…`, 133 members, verified.

## Development run with a larger L-BFGS budget (v2)

[protocol-v2-development-20261014.json](protocol-v2-development-20261014.json) changes only the L-BFGS budget, from 20 to 800 calls (closure cap 520 to 20,800). Data, model, normalization, Adam schedule, L-BFGS settings, seeds, training limits and every physical gate stay as in v1. Seeds 361901 and 361902 were observed in v1, so this is a development reachability study, not a formal time-to-accuracy comparison.

Each seed trains with both engines on one four-GPU H100 node, one lane per GPU: the compiled packed activation with its analytic transpose (`analytic`), and parameter AD over the same compiled packed forward (`parameter_ad`). Both replay CUDA Graphs. `box376.run_c36_v2` runs the batch once. Each lane runs a compiler smoke, the training (`box376.potential_train_v2`) and physical observations of its checkpoints within a time budget and the allocation deadline. Then `box376.read_c36_v2` rechecks every saved comparison, hash and event count from the raw files. Every complete loss/gradient check passes at 28 check points per run, with a maximum error of 1.16e-5 of the original budget.

| Engine | Seed | First call meeting both training limits | Seconds to it | Final loss | Final physical outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| analytic | 361901 | 92 | 1,014 | 1.84e-6 | fails stratified pressure (0.0025) and endpoint (0.00107) |
| analytic | 361902 | 86 | 975 | 1.35e-6 | fails stratified pressure (0.0025) only |
| parameter_ad | 361901 | 90 | 1,319 | 1.63e-6 | partial: stratified pressure 0.0021 fails, uniform passes |
| parameter_ad | 361902 | 85 | 1,276 | 2.34e-6 | partial: stratified pressure 0.0033 fails, uniform passes |

- **Training gate.** All four runs pass it after about 1,860–2,040 L-BFGS evaluations. On these seeds, the analytic engine reaches every loss threshold from 1e-2 to 1e-5 1.30–1.33× sooner than parameter AD.
- **Physical observation.** One complete observation takes about 99 minutes, so this batch observed only the final checkpoints. The batch's own time guard stopped the parameter-AD observations after both holdouts, so that the reader and archive could finish within the allocation. [The later observation batches](#all-plane-observations-along-the-training-trajectory) reproduce both analytic rows exactly, complete both partial rows and re-read every observation with the independent reader.
- **Comparison with v1.**
  - Stratified velocity error falls from 0.018–0.021 to 0.0011–0.0015.
  - Uniform velocity error falls from 0.034–0.044 to 0.0020–0.0025.
  - The analytic trajectory endpoint error falls from 0.023–0.030 to 0.00097 and 0.00107.
  - The uniform field gate passes in every run, but only 63 of its 4,096 points lie inside the box support.
  - The initial and boundary values pass in both complete observations.
- **Remaining failure.** Stratified corrected pressure fails in every run. It is largest in the plateau region inside the box, not in the cutoff layers.
- **Outcome.** The registered conditions are (a) "Loss stays above 1e-4", (b) "Loss passes 1e-4 but physical errors stop falling" and (c) "Physical gates pass". The reader's mechanical classification is (b). This batch could not evaluate the trend condition of (b), because it observed only final checkpoints; every reported physical error fell 7–31× per seed relative to v1. The [later observations of every checkpoint](#all-plane-observations-along-the-training-trajectory) show the errors still falling at the end of the budget, so (b) does not apply either, and no registered branch applies cleanly. Budget alone reaches the training gate, but within 800 calls it does not reach the stratified pressure gate.
- **Timing caveat.** Both arms' training times include the same scheduled dense-AD checks and checkpoint writes, which bias the 1.30–1.33× toward one.

Files:
- [c36-v2-development-result-20261014.json](c36-v2-development-result-20261014.json): public summary.
- [c36-v2-development-reader-20261014.json](c36-v2-development-reader-20261014.json): independent reader output.
- Complete archive: 295 members, sha256 `266aa8d2…b3b45c6e8`. It has verified copies on cluster scratch and persistent project storage.

## All-plane observations along the training trajectory

Four more run-once observation batches (one four-GPU H100 node each) observe every saved checkpoint of the v2 runs and 26 checkpoints of the v3 continuation runs below.
- **Coverage.** The v2 runs saved 52 checkpoints, from the untrained parameters through Adam updates 1,000–4,000 and L-BFGS calls 20–800; they hold 46 distinct parameter vectors. In total there are 72 observations of 72 distinct parameter vectors.
- **Planes.** Each observation uses the unchanged observer (`box376.potential_physical`) on every data plane: the stratified and uniform holdouts, seven time slices, the initial and boundary sets, and both trajectory refinements.
- **What is gated.** Physical acceptance gates the two holdouts, the initial and boundary values, and the finer trajectory refinement. The time slices are diagnostic planes with the same field limits.
- **Independent reader.** `box376.audit_potential_run` re-reads every observation: 117 saved-array checks each, 8,424 in total, all PASS.
- **Reproduction.** Re-observing the two analytic v2 finals reproduces every metric of their earlier observation exactly.

**Observer chunk.** `box376.observer_calibration` measures chunks on 512 points, and `box376.run_c36_observe` picks one by a rule fixed before the measurement: the fastest chunk whose states agree with chunk 128 to 1e-10 of their maximum magnitude, within a peak-memory limit.
- Chunk 512 is 3.9× faster than chunk 128. It agrees to at most 1.8e-11 absolute (7.9e-12 relative) and peaks at 56.6 GB. Chunks 1,024 and 2,048 run out of device memory.
- The first batch's 50 GB limit, set before this measurement, kept chunk 128 (about 100 minutes per observation). Its deadline guard stopped it after 8 of 46 planned observations.
- Later batches use a corrected 60 GB limit and chunk 512 (median 31 minutes). Chunk equivalence rests on the calibration points; no full checkpoint was observed at both chunks.

**Results.**
- **Acceptance.** No observation passes physical acceptance; 171 of 360 acceptance tests pass.
  - Initial and boundary values vanish identically, because the gate vanishes there.
  - The uniform holdout passes from L-BFGS call 200 on (loss ≤ 2.8e-5).
  - The trajectory endpoint passes only at three final checkpoints.
  - The stratified holdout fails at every checkpoint. At the finals, only its corrected pressure fails, at 1.9–3.3× the limit.
- **Diagnostic time slices.** The four slices at rest (t = 0, 1/4, 3/4, 1), where the exact velocity and force vanish identically, always pass. The three inside the motion window always fail, by at least 2.2× on pressure.
- **Trend.** Along every v2 run the physical errors fall with the loss.
  - Rank correlation between log loss and log error is 0.95–0.96 for stratified pressure ([c36-trend-correlations-20261015.json](c36-trend-correlations-20261015.json), from [tools/c36_rank_correlations.py](tools/c36_rank_correlations.py)) and 0.99–1.00 for velocity.
  - The L-BFGS-phase log–log slopes are 0.31–0.41 for pressure and 0.45–0.50 for velocity.
  - From call 400 to call 800, pressure still falls by 0.53–0.83× and velocity by 0.48–0.61×.
  - So the errors have not stopped falling; they fall more slowly than the loss. Extrapolating the pressure slopes would put the gate at a loss 6–47× below the final values; that extrapolation is untested.
- **Trajectory sensitivity.** The two engines differ only by rounding, yet they end at stratified pressures of 2.49e-3 and 2.08e-3 (seed 361901) and 2.49e-3 and 3.28e-3 (seed 361902).

Files:
- [c36-observations-20261015.json](c36-observations-20261015.json): every observation with all planes, acceptance tests, reader result and trend statistics. It was generated on the cluster by [tools/c36_publish_observations.py](tools/c36_publish_observations.py); [tools/c36_trend_figure.py](tools/c36_trend_figure.py) draws the paper's Figure 4 from it.
- Batches: four observation batches (the last runs two frozen drivers in sequence). The complete archives have 191, 581, 371, 131 and 161 members. They are verified on cluster scratch and on persistent project storage, and every owned GPU lock was released.

## Temporal-smoothness continuation (v3)

[protocol-v3-smoothness-development-20261015.json](protocol-v3-smoothness-development-20261015.json) keeps the v2 budget (4,000 Adam updates, 800 L-BFGS calls) but starts on a smoother problem.

**Smoothness family.** `box376.schedule_family` replaces the motion schedule s(t) = σ(2t − 1/2) by s_w(t) = σ(t/w − 1/(2w) + 1/2). The box then moves for t in (1/2 − w/2, 1/2 + w/2), with the same displacement and dilation.
- The k-th time derivatives scale as w⁻ᵏ, and w = 1/2 is the original problem bit for bit.
- The model gate contains the schedule speed, so the geometry features, the gate and the forcing all change with w. The value targets stay zero, which is checked.
- [`box376.test_schedule_family`](src/box376/test_schedule_family.py) has six CPU tests. They include agreement of both engines with native coordinate AD at w = 3/4; they run in CI.

**Schedules.** Each schedule is paired with the v2 analytic run of the same seed, the w = 1/2 baseline.
- **C1:** 1,000 Adam updates at w = 1, then 1,000 at w = 3/4, then 2,000 Adam updates and 800 L-BFGS calls at w = 1/2.
- **C2:** 2,000 Adam updates at w = 1, then 2,000 Adam updates and 100 L-BFGS calls at w = 3/4, then 700 L-BFGS calls at w = 1/2.

**Stage checks.** `box376.potential_train_v3` builds a fresh compiled engine and CUDA Graph for every stage. Each stage is checked against dense parameter AD at the entry parameters, after a parameter change, and again after garbage collection and allocator churn.

**Attempt 1 failed in all four lanes** at the Adam-100 dense-AD check.
- The graph-capture helper keeps the graph and its outputs but not the engine that owns the prepared data. The trainer held that engine only in a stage-local variable, so replays read released memory.
- The repair keeps the engine with its replay, adds the post-churn check and a non-negative-loss guard, and gives the CPU self-test a weak-reference replay. A negative control without retention fails the self-test.
- Attempt 2 changes only the trainer. Both attempts are archived: [c36-v3-smoothness-batches-20261015.json](c36-v3-smoothness-batches-20261015.json).

**Training (attempt 2).** All four runs complete, pass all 32 complete-gradient checks per run (maximum 1.2e-5 of the original budget) and reach the training gate.

Evaluations to the first original-problem loss at or below each threshold (seconds in parentheses):

| Seed | Schedule | 1e-3 | 3e-4 | 1e-4 | 3e-5 | 1e-5 | Final loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 361901 | baseline | 4,042 (677) | 5,144 (864) | 6,039 (1,014) | 8,077 (1,359) | 12,129 (2,045) | 1.84e-6 |
| 361901 | C1 | 2,577 (564) | 4,538 (889) | 5,128 (989) | 6,704 (1,254) | 9,975 (1,826) | 1.11e-6 |
| 361901 | C2 | 6,239 (1,190) | 6,311 (1,202) | 6,571 (1,246) | 8,010 (1,488) | 12,192 (2,195) | 1.25e-6 |
| 361902 | baseline | 4,030 (666) | 5,048 (834) | 5,891 (975) | 7,600 (1,262) | 10,955 (1,821) | 1.35e-6 |
| 361902 | C1 | 2,764 (593) | 4,576 (896) | 5,200 (1,003) | 7,012 (1,309) | 9,903 (1,800) | 1.41e-6 |
| 361902 | C2 | 6,237 (1,186) | 6,306 (1,197) | 6,585 (1,244) | 8,262 (1,529) | 11,622 (2,118) | 1.32e-6 |

- **C1** reaches every registered threshold in 8–36% fewer evaluations on both seeds. It is not always faster in seconds: its two stage rebuilds (about 65 s each) precede every threshold.
- **C2** is later at every threshold in seconds on both seeds, and later in nine of ten seed–threshold pairs in evaluations.

**Physics on the original problem (stratified holdout and trajectory endpoint, ×1e-3).**

| Seed | Schedule | Pressure at gate | Velocity at gate | Endpoint at gate | Final pressure | Final plateau pressure | Final endpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 361901 | baseline | 9.73 | 9.02 | 11.1 | 2.49 | 4.65 | 1.07 |
| 361901 | C1 | 13.1 | 8.27 | 8.29 | 2.09 | 3.11 | 1.12 |
| 361901 | C2 | 5.95 | 6.00 | 5.01 | 1.93 | 3.14 | 1.01 |
| 361902 | baseline | 9.41 | 7.69 | 10.2 | 2.49 | 4.17 | 0.97 |
| 361902 | C1 | 11.1 | 8.09 | 10.8 | 2.77 | 4.35 | 1.12 |
| 361902 | C2 | 6.48 | 5.97 | 4.07 | 2.74 | 4.39 | 0.89 |

- **At the training gate** (loss ≤ 1e-4 and ‖g‖∞ ≤ 1e-3, matched by loss; reached at L-BFGS calls 86–92, 51–55 and 117–118), C2 is clearly closer to the physical solution: 0.61–0.69× the baseline's pressure and 0.40–0.45× its endpoint error.
- **At the final checkpoints** both schedules are lower than the baseline on seed 361901 and higher on seed 361902. These are 10–22% changes, comparable to the 16–32% difference between the two engines.
- **Every final checkpoint fails stratified pressure.** Only the trajectory endpoint test differs between the schedules.
- **Transfer.** The w = 3/4 parameters, observed on the original problem, have stratified velocity error 0.049–0.059. The exact w = 3/4 and w = 1/2 fields differ by 0.81 in that norm ([c36-schedule-field-distance-20261015.json](c36-schedule-field-distance-20261015.json), from `box376.schedule_field_distance`). The gate carries the schedule speed and the potential is learned in moving-box features, so most of the width dependence lies outside the learned function.

**Registered rule.** The protocol's clauses, quoted:
- helps: "both schedules or one schedule reach every threshold sooner and the final physical errors (stratified pressure in particular) are lower than the baseline on both seeds";
- no effect: "differences within the seed-to-seed spread of the baseline";
- hurts: "thresholds reached later or final errors higher on both seeds".

**Reading.**
- **The continuation does not help:** neither schedule meets the helps clause.
- **C2 meets the hurts clause through its thresholds:** later at all five in seconds on both seeds, and at nine of ten in evaluations.
- **C1 fits no branch cleanly.**
  - It is not helpful: its final pressure is higher on seed 361902, and it is not faster in seconds at every threshold.
  - It is not harmful: its evaluation counts are all lower.
  - The no-effect band is degenerate, because both baseline seeds end at the same pressure to three digits.
  - The rule also did not say whether thresholds count evaluations or seconds.
- **Bottom line.** Within this budget a smoother early problem moves the trajectory, but neither schedule changes which acceptance test fails.

Files:
- [c36-v3-smoothness-result-20261015.json](c36-v3-smoothness-result-20261015.json): reader output. Its state label is inherited from the v2 reader and is kept unchanged.
- [c36-v3-baseline-curves-20261015.json](c36-v3-baseline-curves-20261015.json): the v3 reader's per-run function applied unchanged to the v2 analytic runs.
- [release-source-map-20261015.json](release-source-map-20261015.json): every executed packet manifest. All published sources are byte-identical to the executed copies, except the attempt-1 trainer.
- Complete training archive: sha256 `949dbee4…91b4077ac`, 292 members, verified on cluster scratch and persistent project storage.

## Combination study (v4, pre-registered; attempt cancelled, superseded by v5)

[protocol-v4-combination-development-20261015.json](protocol-v4-combination-development-20261015.json) tests the combination of the C2 continuation with a long L-BFGS budget.

**Design.** Both the C2 runs and the paired v2 analytic baseline runs resume from their final checkpoints. Their prior budgets were equal: 4,000 Adam updates and 800 L-BFGS calls. Each run then gets 4,000 more L-BFGS calls on the original problem, with a fresh L-BFGS history.
- `box376.potential_train_v4` verifies each start checkpoint's file and parameter hashes and refuses a mismatch.
- It checks the original-problem engine against the preserved native reference.
- At the resume point it checks dense parameter AD at entry, after a parameter change, and after allocator churn.
- A continuity check requires the start loss to equal the source run's final loss.
- Counters restart at zero, so the all-planes observer indexes the run's own event log.

**Questions.**
- Primary: does the extra budget reach physical acceptance?
- Secondary: does the C2 start reach it sooner? This comparison is underpowered (two seeds, and rounding-equivalent engines already differ by up to 32%), so differences below 32% are not interpreted.

**Decision.** [`box376.decide_c36_v4`](src/box376/decide_c36_v4.py) implements the registered rule mechanically; [six synthetic tests](src/box376/test_decide_c36_v4.py) cover its outcomes and run in CI. It reads the observations at 800, 1,600, 2,400 and 3,200 extra calls and at the final checkpoint. The six mutually exclusive outcomes are:
- O1: both arms pass, with sub-result C2 sooner, baseline sooner, or same/mixed;
- O2: only C2 passes;
- O3: only the baseline passes;
- O4: partial;
- O5: no run passes, errors still falling;
- O6: no run passes, errors stopped falling (final pressure ≥ 0.95× its value at 3,200 calls on both seeds).

If any arm passes on both seeds, a formal fresh-seed protocol follows.

**Status: the attempt was cancelled after 13.5 minutes, before any physical observation.** The cancellation was based on optimizer progress only. The partial outputs are archived (157 members, sha256 `6e1b6143…`).

All four lanes passed every qualification check:
- native reference ≤ 2.3e-6 of the original budget;
- dense AD at the resume point, including after allocator churn, ≤ 6.0e-6;
- the check at L-BFGS call 100 ≤ 6.8e-6;
- the start loss reproduced the source final loss bit for bit or within 4.2e-11 of the budget.

**Why it was cancelled: the design is flawed.** A fresh L-BFGS history started at a loss of about 1e-6 falls into an overshoot/backtrack cycle.
- In each iteration the trial step t = 1 raises the loss (to 1.3–1.9e-6). The strong-Wolfe search then accepts a tiny step.
- Over the first 100 calls the loss fell by only 0.2–0.7%. The source runs fell by about 32% over their last 100 calls.
- Fresh histories started at higher loss behave normally: the C2 runs' w = 1/2 stage started at loss 1.6e-2, and the v2 runs' L-BFGS started at loss 1.1e-3.
- The resumed study therefore measured the restart, not the extra budget, and is not continued.

**What the source runs showed.** In the last 50 calls of the source runs:
- the v2 baseline runs still fall by 3.4–3.7% per 10 calls;
- the continuation runs have stalled at 1.1–1.4e-6, falling 0.02–0.46% per 10 calls with 24–25 evaluations per call.

## Budget study (v5, pre-registered; outcome E5)

[protocol-v5-budget-development-20261015.json](protocol-v5-budget-development-20261015.json) replaces the cancelled v4 design.

**Design.** It is the v2 development recipe with the L-BFGS budget extended from 800 to 4,800 calls, in one continuous run per (seed, engine). That avoids any restart of the L-BFGS history.
- Both engines (compiled analytic transpose and parameter AD over the same compiled forward) train both seeds on one four-GPU H100 node.
- The trainer and driver are the unchanged v2 sources (`box376.potential_train_v2`, `box376.run_c36_v2`); only [`entry_c36_v5.sh`](src/box376/entry_c36_v5.sh) and the protocol are new.
- The smoothness arm is dropped: its source runs had stalled at 1.1–1.4e-6 by call 800, while the baseline was still descending.

**Primary question.** Does the extended budget reach physical acceptance?

**Decision.** [`box376.decide_c36_v5`](src/box376/decide_c36_v5.py) applies the registered rule mechanically; [five synthetic tests](src/box376/test_decide_c36_v5.py) run in CI. It uses the separate all-planes observations at 1,600, 2,400, 3,200 and 4,000 total calls and at the final checkpoint, plus call 800 if its parameters differ from the v2 final. The five mutually exclusive outcomes are:
- E1: all four runs pass;
- E2: exactly one engine passes on both seeds;
- E3: partial;
- E4: no run passes, errors still falling;
- E5: no run passes, errors stopped falling (final pressure ≥ 0.95× its value at 4,000 calls on both seeds of an engine).

E1 or E2 leads to a formal fresh-seed time-to-accuracy protocol.

**Reproduction diagnostic.** As a diagnostic only, never a gate, the parameters at call 800 are compared with the v2 runs.

**Training.** All four runs completed the 4,800 calls within the time guard, taking 20,418–26,652 s against a 27,000 s limit. All 60 complete-gradient checks pass, at most 1.1e-5 of the original budget.
- **The loss stalls soon after call 800.** From call 800 to 4,800 it falls only 3–12% (for example 1.830e-6 → 1.607e-6). After call 1,200 it falls about 0.2–0.3% per 400 calls.
- **No run reaches 1e-6.** The threshold 3e-6 is reached after 16,077–18,924 evaluations.
- **Reproduction diagnostic.** The runs are not bit-identical to v2: the first difference is in the last digit of the loss at Adam updates 3–8, a reduction-order effect. Call 800 was therefore observed anew.
- **Engine timing (development only).** Parameter AD needed 1.28–1.36× the seconds of the analytic engine to each threshold from 1e-3 to 3e-6.

**Physics.** A separate observation batch covered 24 checkpoints (calls 800, 1,600, 2,400, 3,200, 4,000 and 4,800 for each run) on all planes. The independent reader passes all 2,808 checks.

| Engine | Seed | Loss ×1e6 at 800 → 4,800 | Stratified pressure ×1e-3 at 800 → 4,800 | Plateau pressure ×1e-3 at 4,800 | Endpoint ×1e-3 at 4,800 | Failing tests at 4,800 |
| --- | ---: | --- | --- | ---: | ---: | --- |
| analytic | 361901 | 1.830 → 1.607 | 2.40 → 2.36 | 4.29 | 0.90 | stratified |
| analytic | 361902 | 1.383 → 1.338 | 2.57 → 2.52 | 4.52 | 1.00 | stratified |
| parameter AD | 361901 | 1.749 → 1.628 | 2.26 → 2.17 | 3.96 | 1.02 | stratified, trajectory |
| parameter AD | 361902 | 1.499 → 1.432 | 3.03 → 2.97 | 5.89 | 0.86 | stratified |

- **The errors stall with the loss.** Stratified pressure changes by only 1.5–3.8% and ends at 2.2–3.0× its limit.
- **Other tests.** The uniform holdout passes at every checkpoint. The trajectory endpoint stays within 0.86–1.02× its limit and passes at 12 of the 24 checkpoints.
- **Acceptance.** No checkpoint passes physical acceptance.

**Registered outcome: E5.** [`box376.decide_c36_v5`](src/box376/decide_c36_v5.py) finds that no run passes and that both engines have stopped falling: the final pressure is 0.998× its value at 4,000 calls. No formal claim follows.
- At 800 calls the errors were still falling, but with this optimizer they stop shortly afterwards, together with the loss. The gradient ∞-norm at the end is 1.2–4.6e-6.
- So the stratified pressure gap is not a budget limit of this optimizer at this scale.
- In the terms of the v2 rule this is branch (b): reaching the gate needs an explicitly changed problem definition (sampling, weights or capacity) in a new protocol.

The in-batch reader is the unchanged v2 reader, so [its output](c36-v5-budget-result-20261015.json) carries the v2 state name and v2 branch letter. The registered v5 outcome is the one in [c36-v5-decision-20261015.json](c36-v5-decision-20261015.json).

Files:
- [c36-v5-observations-20261015.json](c36-v5-observations-20261015.json): all 24 observations with all planes and reader results.
- [c36-v5-training-curves-20261015.json](c36-v5-training-curves-20261015.json): loss curves with the v5 thresholds.
- Archives: v5 training (446 members) and the observation batch (431 members) are both verified on cluster scratch and persistent project storage.


## Wider-cutoff instance (v6, pre-registered; outcome F3)

[protocol-v6-eta16-development-20261018.json](protocol-v6-eta16-development-20261018.json) studies a different single-box instance. The cutoff width doubles from 1/32 to 1/16; every other element of the v2 recipe is kept:
- the Math376 construction and box motion;
- the sampler seeds and point counts;
- the model and its features;
- the scale rule and physical limits;
- the Adam schedule, the L-BFGS settings and the 800-call budget;
- the stored initial parameter bytes, both engines and the observer.

**Motivation.** The [pressure post-hoc](c36-pressure-structure-20261018.json) shows that at 1/32 the cutoff layers alone keep the stratified pressure 1.5–1.9× over its limit, so plateau-only weighting cannot pass. At 1/16 the layer force RMS is 4–5× smaller and the force-gradient RMS 9–11× smaller. This is a new, easier instance, not a repair of the 1/32 problem. Its decision does not compare with v1–v5.

**Same limits, different absolute tolerances.** The scales follow the unchanged rule from the new calibration:
- velocity RMS 0.303 (1/32: 0.377);
- momentum 38.1 (164);
- momentum gradient 5,252 (50,451).

Pressure is normalized by the squared velocity RMS, so the absolute pressure tolerance becomes 9.2e-5, or 0.65× the 1/32 value of 1.42e-4. The wider support also widens the model's feature range inside it from at most 4 to at most 7 half-widths.

**Inputs and qualification (CPU, before submission).**
- **New generators.** [`box376.calibrate_source_eta`](src/box376/calibrate_source_eta.py) and [`box376.prepare_training_data_eta`](src/box376/prepare_training_data_eta.py) differ from their originals only in the cutoff width argument.
- **Originals now in the repository.** [`calibrate_source.py`](src/box376/calibrate_source.py) and its mpmath [`oracle.py`](src/box376/oracle.py) are added byte-identical. The frozen calibration was generated by them but they were never published before.
- **Reproduction at 1/32.** The new generators reproduce the frozen calibration and training source to at most 2.8e-14 of each array maximum ([check](v6-generator-reproduction-20261018.json)). Sampled weights, regions, ids and targets are identical.
- **New inputs.** They are in [data-eta16/](data-eta16/). Ten saved calibration points pass the 80/160-digit mpmath checks, at most 2.3e-4 of the budget.
- **Geometry** ([`box376.check_geometry_eta`](src/box376/check_geometry_eta.py); [1/16](v6-geometry-eta16-20261018.json), [1/32](v6-geometry-eta32-20261018.json)).
  - Prepared feature and gate states agree with fresh native coordinate AD on 88 region-stratified points, at most 2.7e-6 of the budget.
  - The gate is exactly zero at every initial, boundary and gauge point.
  - The support stays within [0.1875, 0.75].

**Trainer and observer.**
- **Trainer.** [`box376.potential_train_v6`](src/box376/potential_train_v6.py) is the v2 trainer except for its reference checks.
  - The preserved full native reference belongs to the 1/32 data, so it supplies only the stored initial parameters.
  - At the original parameters the eager engine is compared with fresh native coordinate AD of the complete objective on a subset: 131 interior points plus every value point, with canonical denominators. At 1/16 the subset covers 16 plateau, 25 face, 26 edge, 25 corner and 10 outside points in the motion window, plus 29 at resting times.
  - The eager, compiled and graph engines are compared with dense parameter AD on the complete data.
  - A full native reference would cost about one H100 hour per seed.
- **Observer.** [`box376.potential_physical_eta`](src/box376/potential_physical_eta.py) runs the unchanged observer with its two cutoff-dependent calls bound to 1/16. It uses chunk 512, observes the final checkpoint first, and then gate-first and call 200 if time allows.
- **Driver.** [`box376.run_c36_v6`](src/box376/run_c36_v6.py) is the v2 driver with these modules and the decision step.

**Decision** ([`box376.decide_c36_v6`](src/box376/decide_c36_v6.py), [synthetic tests](src/box376/test_decide_c36_v6.py)). It applies within this instance only:
- **F1:** some run passes physical acceptance at an observed checkpoint. This leads to a formal fresh-seed time-to-accuracy protocol on this instance.
- **F2:** no pass, but some run's complete final observation exceeds no acceptance limit by more than 1.5×.
- **F0 (incomplete):** no pass and no near miss, and some run lacks a complete final observation, for example after a failed or killed training or an observation that did not run. There is no scientific reading; the cause is found and repaired before a new attempt.
- **F3:** no pass and no near miss, with complete final observations of all four runs.

**Execution.** One ACC node for 3 h runs four concurrent single-GPU lanes (two seeds × two engines), the smallest count that runs all four at once. The start command is embedded in the batch script. The work root is on project storage, with every cache inside the batch directory.

**Results (paper Section 6.11, Table 22; registered outcome F3).** The batch ran once on one four-GPU H100 node, from the git archive of `7671c3f`; the [release map](release-source-map-v6-20261018.json) has the hashes. Start-up qualification passes in every lane: subset native 2.0–2.8e-7, dense AD, compiled and graph at most 9.8e-7 of the budget. All 116 saved checks are within 1.7e-5.

| Engine | Seed | Gate (L-BFGS call) | Final loss ×1e6 | Stratified pressure / limit | Plateau | Layers alone | Stratified velocity ×1e-3 | Endpoint ×1e-3 | Failing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| analytic | 361901 | 149 | 4.227 | 2.89 | 4.84 | 2.20 | 1.81 | 0.80 | stratified |
| analytic | 361902 | 110 | 2.767 | 2.16 | 3.37 | 1.73 | 1.46 | 0.80 | stratified |
| parameter AD | 361901 | 152 | 3.490 | 2.58 | 4.10 | 2.04 | 1.59 | 0.62 | stratified |
| parameter AD | 361902 | 111 | 3.002 | 2.04 | 3.02 | 1.67 | 1.56 | 1.09 | stratified, trajectory |

- **Final checkpoints.** Stratified pressure (2.0–2.9×) is the only failing test in three runs; the fourth also misses the endpoint by 9%. Velocity, momentum, divergence, the uniform holdout and the initial and boundary values pass. The first training-gate checkpoints fail more widely (pressure 9.1–11.5×, endpoint 4.8–9.0×).
- **Registered outcome F3** ([decision](c36-v6-decision-20261018.json)): no pass and no near miss (best final 2.04× its worst limit), all four finals complete. No formal claim follows.
- **Interpretation.** Doubling the cutoff lowered the layer forcing 4–5× but not the normalized pressure error, which stays where it was at 1/32 after 800 calls (2.3–3.0×). The layers alone still leave 1.7–2.2× ([decomposition](c36-v6-pressure-structure-20261018.json)), so the remaining error is not set by the layer forcing alone.
- **Engine timing (development only).** Parameter AD needed 1.25–1.34× the analytic engine's seconds to each loss threshold, 1.26–1.27× per evaluation.
- **What drives the pressure error (CPU post-hoc, paper Section 6.11).** The exact pressure vanishes, so `grad p = R − dI` pointwise: `R` is the observed momentum residual and `dI` the inertia error of the learned velocity. [`tools/c36_pressure_attribution.py`](tools/c36_pressure_attribution.py) decomposes the saved states of the v5 and v6 finals ([v5](c36-pressure-attribution-v5-20261019.json), [v6](c36-pressure-attribution-v6-20261019.json)). It reproduces the observer's regional values and checks the identity and its divergence to 4e-12.
  - **Not forced by the velocity error.** The cosine between `grad p` and `−dI` is −0.05 to 0.07 in every region, at the noise floor of 0.02–0.03.
  - **A small share of the residual.** `dI` dominates the residual at 3–15× `grad p`. The pressure therefore carries 4–11% of the residual energy in the plateau and 0.5–1.3% in the layers, and the plateau holds 0.6–2.5% of the stratified momentum energy.
  - **Poisson balance.** In the plateau the velocity term is 0.3–0.7× `Δp`. In the layers both divergences are large and nearly cancel, so the balance is not interpretable there.
  - **Reading.** The pressure error is an independent component that the loss weights at about one percent; a more accurate velocity would not by itself remove it.

Files:
- [c36-v6-observations-20261018.json](c36-v6-observations-20261018.json): training and all observations, from [`tools/c36_v6_publish.py`](tools/c36_v6_publish.py).
- [c36-v6-development-result-20261018.json](c36-v6-development-result-20261018.json): the unchanged in-batch reader, byte-identical to the executed output; it carries the development state name.
- Archive: 396 members, verified on cluster project storage.

## Optimizer study (v7, pre-registered): dense self-scaled Broyden

[protocol-v7-ssbroyden-development-20261019.json](protocol-v7-ssbroyden-development-20261019.json) keeps the original instance (`eta = 1/32`) and the v2 recipe and replaces only the L-BFGS phase.

**Motivation.** The [pressure attribution](c36-pressure-attribution-v5-20261019.json) shows that the remaining pressure error is an independent component weighted at about one percent of the residual energy. L-BFGS stalls before resolving it, near a loss of 1.3–1.6e-6 (v5). Self-scaled quasi-Newton methods reach much lower PINN losses than L-BFGS (Urbán, Stefanou and Pons, J. Comput. Phys. 523, 113656, 2025; Kiyani et al., arXiv:2501.16371). Those studies use networks of 1–3 thousand parameters; this one has 50,692.

**Optimizer** ([`box376.ssbroyden`](src/box376/ssbroyden.py)).
- **Update.** The self-scaled Broyden update with a dense FP64 inverse Hessian on the GPU (20.6 GB), following Eqs. (10) and (13)–(23) of Urbán et al. They agree with Kiyani et al., Eq. (15); both were taken verbatim from the two papers and checked against each other.
- **Settings.** H0 = I. The strong-Wolfe line search uses c1 = 1e-4 and c2 = 0.9, and its tolerance on parameter change is 0, as in the L-BFGS runs.
- **Cost per iteration.** One dense matrix-vector product: `H y = H g_{k+1} + d_k`, and the next `H g` is carried in O(n).
- **Safeguards.** The update is skipped when there is no curvature. H is reset on a non-descent direction or on a line search without decrease. Two consecutive resets end the phase.
- **Tests** ([`test_ssbroyden`](src/box376/test_ssbroyden.py), CPU, both runtimes):
  - with τ = φ = 1 the update equals the textbook inverse BFGS update;
  - τ, θ and φ match a separately written implementation of Kiyani et al.'s definitions in both θ branches;
  - the secant equation holds within its rounding amplification;
  - positive definiteness is preserved, and the carried product matches an explicit one;
  - all three modes converge on a convex quadratic and on extended Rosenbrock.

**Arms.** There are four lanes on one node, all with the compiled analytic engine:
- **fresh:** the unchanged 4,000 Adam updates, then SSBroyden, on both seeds;
- **continue:** SSBroyden from the two v5 analytic final checkpoints. Their file and parameter hashes are in the protocol, and the start loss must reproduce the source final loss within the original budget.

[`box376.potential_train_v7`](src/box376/potential_train_v7.py) is the v2 trainer with these changes. Complete-gradient checks against dense parameter AD, and the carried-product check, run every 1,000 iterations. The training guard is 4,000 s.

**Amendment before any run (2026-10-19).** The allocation behind the regular QoS is exhausted. At the user's instruction the batch uses the 2 h interactive QoS on one node. So the training guard falls from 6,600 to 4,000 s, and the deadline reserve from 900 to 400 s; v6's reader, decision and archive took about 25 s. That leaves the optimizer about 60% of the planned budget (roughly 11,000–13,000 iterations), which is registered as a further caveat.

**Decision** ([`box376.decide_c36_v7`](src/box376/decide_c36_v7.py), [tests](src/box376/test_decide_c36_v7.py)). It uses the v6 outcomes in order:
- F1: a pass;
- F2: a near miss within 1.5×;
- F0: incomplete, with no scientific reading;
- F3: fail with all four finals complete.

There are three secondary readouts, which do not change the outcome:
- for the fresh runs, the first evaluation at each loss threshold against the v5 analytic L-BFGS run of the same seed;
- for the continue runs, the final loss over the start loss;
- post hoc, the pressure attribution of the finals.

**Caveats (registered).**
- A dense quasi-Newton update adds at most rank two per iteration, so about 20,000 iterations see fewer directions than there are parameters. A failure within 3 h does not exclude SSBroyden at a longer budget; seconds per iteration are recorded.
- A failed continuation does not by itself reject the optimizer, because restarting the curvature at a loss near 1e-6 can measure the restart (v4).

**Self-tests.** The full trainer does not fit the login node, so the integration was tested as a component ([result](v7-component-selftest-20261019.json)). The optimizer drives a 200-parameter slice of the real eager C36 objective through the trainer's own callbacks:
- the parameters equal the optimizer's point after every iteration;
- the loss falls over six iterations;
- the carried product agrees to 4e-16;
- the start-checkpoint hash logic accepts the v5 bytes and rejects a modified copy.

A short driver self-test with one fresh and one continue lane ([record](v7-selftest-20261019.json)) imports every GPU-path module under Torch 2.4. The continue lane receives the packet's start checkpoint, and its start loss reproduces the v5 final loss to 1e-8 of the budget on CPU. The fresh lane reaches the optimizer phase. The login node then ends both CPU trainers. Even so, the driver's reader, decision (F0, as registered for incomplete lanes) and archive complete.


## GH200 fresh-arm continuation (pre-registered 2026-10-21)

[protocol-v7-gh200-fresh-development-20261021.json](protocol-v7-gh200-fresh-development-20261021.json) registers an independent two-run development batch after the MN5 route was retired. The exact v5 continuation checkpoints are unavailable. Their original two tasks remain unexecuted here; this batch does not complete the earlier four-task protocol or pool results with its unknown execution state.

- **Runs:** fresh starts for the previously observed seeds 361901 and 361902, one allocated GH200 per seed. The stored initial parameter bytes, eta=1/32 data, 4,000 Adam updates, SSBroyden algorithm, **4,000-second training cap**, objective and all numerical/physical gates are unchanged.
- **Allocation:** one node, two GPUs, three hours (at most six allocated GPU-hours). The additional wall time covers setup and physical observation on the new platform; it does not increase training time.
- **Runtime:** ARM Python 3.13.5, Torch 2.14.1+cu130, Triton 3.8.0, NumPy 2.5.4 and SciPy 1.18.1. The [driver adapter](src/box376/run_c36_v7_gh200.py) disables the old Torch target-tuple shim through the frozen execution field. Scientific child commands and the reader are inherited from v7.
- **Qualification:** [independent archive audit](gh200-platform-qualification-20261021.json): seven optimizer/decision tests pass; 32 raw-array comparisons pass the unchanged numerical budget (maximum 2.25e-6 of that budget), including the complete 50,692-parameter gradient, in-place parameter changes and CUDA Graph replay. The full 20.56 GB inverse Hessian allocates and passes identity-product/reset checks. The 149-member archive was verified independently. This establishes the tested platform paths only, not training convergence, physical acceptance or a speedup.
- **Decision:** the existing ordered F1/F2/F0/F3 rules apply only to these **two** registered fresh runs. F3 requires both complete final observations; missing observations remain F0. Results are development evidence because both seeds were seen before. The unavailable original continuation arms remain a separate unresolved historical scope.
- **Lifecycle:** immutable source/data/reference hashes, single execution, bounded cluster-side workload, complete success/failure archives and resource release without an online desktop.
