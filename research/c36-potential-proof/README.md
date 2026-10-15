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

The [CPU qualification](zero-gate-folding-cpu-result-20261013.json) passes 12 full-objective/numerator/all-parameter-gradient comparisons against the preserved native H100 anchors, covering both original stored initializations. The NumPy-only reader verifies all 24 objective arrays and 15 folding arrays. Of the 4,999 canonical rows, 2,933 have active jets; the other 2,066 are represented by their constant loss. Four regression cases cover nonzero forcing/targets, changed parameters, empty active families, and a zero gate value with a nonzero derivative as small as `1e-30`. The same regressions also pass in the actual experiment runtime before GPU admission. The complete CPU archive is fully verified locally. This is numerical CPU qualification; the folded GPU comparison is pending and no new timing ratio is claimed.

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
- **Diagnostic time slices.** The four slices at rest (t = 0, 1/4, 3/4, 1) always pass. The three inside the motion window always fail, by at least 2.2× on pressure.
- **Trend.** Along every v2 run the physical errors fall with the loss.
  - Rank correlation between log loss and log error is 0.94–0.96 for stratified pressure and 0.99–1.00 for velocity.
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

**Physics (stratified holdout, ×1e-3).**

| Seed | Schedule | Pressure at gate | Velocity at gate | Endpoint at gate | Final pressure | Final plateau pressure | Final endpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 361901 | baseline | 9.73 | 9.02 | 11.1 | 2.49 | 4.65 | 1.07 |
| 361901 | C1 | 13.1 | 8.27 | 8.29 | 2.09 | 3.11 | 1.12 |
| 361901 | C2 | 5.95 | 6.00 | 5.01 | 1.93 | 3.14 | 1.01 |
| 361902 | baseline | 9.41 | 7.69 | 10.2 | 2.49 | 4.17 | 0.97 |
| 361902 | C1 | 11.1 | 8.09 | 10.8 | 2.77 | 4.35 | 1.12 |
| 361902 | C2 | 6.48 | 5.97 | 4.07 | 2.74 | 4.39 | 0.89 |

- **At the training gate** (matched by loss; reached at L-BFGS calls 86–92, 51–55 and 117–118), C2 is clearly closer to the physical solution: 0.61–0.69× the baseline's pressure and 0.40–0.45× its endpoint error.
- **At the final checkpoints** both schedules are lower than the baseline on seed 361901 and higher on seed 361902. These are 10–22% changes, comparable to the 16–32% difference between the two engines.
- **Every final checkpoint fails stratified pressure.** Only the trajectory endpoint test differs between the schedules.
- **Transfer.** The w = 3/4 parameters, observed on the original problem, have stratified velocity error 0.049–0.059. The exact w = 3/4 and w = 1/2 fields differ by 0.81 in that norm ([c36-schedule-field-distance-20261015.json](c36-schedule-field-distance-20261015.json), from `box376.schedule_field_distance`). The gate carries the schedule speed and the potential is learned in moving-box features, so most of the width dependence lies outside the learned function.

**Registered rule.**
- **C2 is harmful**, by the threshold clause on both seeds.
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

