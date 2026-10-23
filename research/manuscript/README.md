# FlashNS research manuscript

[PDF](flashns.pdf) and [editable standalone LaTeX](flashns.tex), updated October 23, 2026. Author and the original accepted 2D H20 experiment are retained from the October 6 release. Equations, tables, bibliography, the method figure and both original plot datasets are embedded in the source; no external TeX inputs are needed. The PDF is built directly with pdfLaTeX (three passes) and needs only standard TeX Live packages: TikZ, booktabs, longtable, hyperref and Latin Modern.

The October 14 revision restructures the paper after an external review without changing any measured value, gate, seed or evidence file: a shorter abstract, an explicit contribution list, a terminology table, a related-work section with five additional verified references (Forward Laplacian, Taylor-mode AD in JAX, Griewank and Walther, NSFnets, Neural Conservation Laws), a method figure, a paragraph on what the 2D solve times measure, a diagnosis of the failed 3D acceptance, and an appendix that now holds the execution and provenance records. Section 7 adds the first accepted three-dimensional solve comparison. On the transient Ethier–Steinman benchmark (protocol v4, six formal seeds, H100 FP64), all 18 solves pass all eight gates. C17 is 3.602× faster than the A24 reference and 6.957× faster than the naive Q35 ablation (paired geometric means); the [result](../ethier-3d/result-v4-20261014.json), [audit](../ethier-3d/audit-v4-20261014.json) and frozen protocol are pinned in the evidence index. Two further clarifications come from the raw protocol rather than new runs: the 2D closure counts include the 4,000 Adam evaluations (every accepted 2D solve stops during L-BFGS after 2,208–5,259 L-BFGS evaluations), and the relative L2 error of v is the binding 2D acceptance criterion for all nine methods.

The new C36 section reports complete 4,999-point / 50,692-parameter checks, paired H100 and CPU objective-gradient comparisons, all four frozen-budget training failures, and complete physical acceptance failures. Older H100 packed-gather rows remain about 3% slower than their dense-activation parameter-AD control. The subsequent packed-activation/analytic-transpose CUDA Graph experiments give 1.044582× with eager activation and **1.323962× with compiled activation** (95% paired interval 1.323087–1.324762). Both arms use the same packed forward and compilation choice; the compiled candidate/control means are 164.484/217.771 ms. Both GPU experiments pass 78 numerical comparisons and independent remote/local readers of 156 arrays and 20 timing windows. The separate one-thread CPU result is 1.265×. The older 1,255× native comparison uses a different source and its original recursive cache. No accepted 3D time-to-accuracy improvement is established.

The October 15 revision adds Sections 6.8 and 6.9, which come from new run-once H100 batches:
- **Section 6.8: physical errors along the training trajectory.** Every saved C36 development checkpoint and 26 continuation checkpoints are observed on all data planes (72 observations), and each is re-read by the independent reader (8,424 checks).
  - The physical errors fall with the loss and are still falling at the end of the budget, so the trend condition of the registered branch (b) does not hold.
  - Stratified corrected pressure inside the moving box stays 1.9–3.3× above its limit.
- **Section 6.9: temporal-smoothness continuation.** It widens the box-motion schedule (w = 1, then 3/4, then the original 1/2) under the same budget.
  - One schedule meets the training gate with 0.6–0.7× the baseline's stratified pressure error. The other reaches the loss thresholds in 8–36% fewer evaluations.
  - Neither changes which acceptance test fails. Under the registered rule, the continuation does not help: the first schedule is harmful and the second fits no branch cleanly.
- **Earlier text.** Section 6.7, the discussion and the conclusion were updated to match, and Figure 4 is generated from the published observation record.
- **Section 5.6: representation and execution on H100 (October 16).** A pre-registered ablation (244 fresh-process measurements) separates the GL8 state from its execution.
  - Native pointwise execution is 1.53–1.83× at every size.
  - The state alone is unresolved at 2,048 points (the A/A gate failed there) and 1.63–1.66× from 16,384 points on.
  - The explicit transpose beats generic autograd only from 16,384 points on (1.31×).
  - At the original size the solve-level gain does not carry over to H100. The L-BFGS phase costs 6.3–7.2 ms per evaluation against under 0.3 ms of gradient replay, so optimization alone is 0.95×.
  - The contributions, discussion and conclusion are updated accordingly.
- **Section 5.6: solves at larger point counts (October 16).** A second pre-registered H100 study runs 36 complete solves (cells A, B and C at 16,384 and 65,536 points, six formal seeds, all accepted).
  - The registered rule finds a solve-level gain for every pair at both sizes.
  - Native GL8 over compiled Q10 is 1.48× (16,384) and 1.88× (65,536) in optimization alone; the protocol expected about 1.4× and 2.0×.
  - Post hoc, each L-BFGS evaluation costs 5.9–7.1 ms beyond the gradient at every size. The solve-level gain is therefore a matter of problem size, not of the device.
  - The 2,048-point Q10 setups (4.7–5.8 s) were warm-cache values, so that setup comparison is warm against warm; warm Q10 setup is still slower than native. At larger sizes, queue order decided which seeds paid a cold compile, which widens the setup-plus-optimization ranges.
  - Table 10 is new, and the replay-versus-solve discussion and the conclusion are rewritten.
- **Section 6.3: zero-gate folding on H100 (October 16).** A pre-registered batch measures exact zero-gate constant folding in both arms on one H100.
  - Both runs pass all 78 numerical comparisons.
  - With folding, the analytic transpose is **1.400×** faster than parameter AD (interval 1.3999–1.4002); the registered classification is an analytic advantage.
  - An unfolded repeat gives 1.325× (1.324× before).
  - Folding itself shortens an evaluation by 1.68× (analytic) and 1.59× (parameter AD), against 1.70 for cost proportional to rows.
  - Table 13 (the C36 timing table) gains two rows, and the abstract, the limitation and the conclusion are updated; the limitation no longer says the folding is qualified on CPU only.
- **Section 6.17: paired C36 cohort against torch.func (October 23).** The registered second paired cohort (five new seeds, both arms on one GH200 per seed, verified loss threshold 1e-8).
  - All ten arms reach the threshold and pass physical acceptance; all five audits pass.
  - Analytic faster in all five pairs: geometric mean 3.92 (3.87–3.96), inside the predicted 3.83 (3.6–4.1); evaluation counts equal, so the gain is per evaluation (4.0–4.1× Adam, 3.7–3.8× SSBroyden).
  - A post-hoc development profile shows the analytic graph replay 3.17× faster at chunk 2,048 than at the protocol's chunk 128, so the registered ratio is a lower bound at the analytic engine's best chunk.
  - Table 29 is new; the Ethier tables are now 30–33. The abstract, discussion and conclusion state the solve-level result.
- **Section 5.9: optimizer execution on GH200 (October 23).** A pre-registered GH200 study replaces only the two-loop recursion of `torch.optim.LBFGS` by a device-side version (Triton kernels, CUDA Graph replay) and runs 36 native-GL8 FP64 solves plus four drift repeats.
  - All 36 stock/light pairs are bitwise identical in final parameters, evaluation and iteration counts and acceptance.
  - Optimization is 2.21× (1.96–2.38) shorter at 2,048 points, 1.82× at 16,384 and 1.48× at 65,536, all six pairs favourable at every size; the L-BFGS phase falls from 6.6–8.5 to 2.9–4.9 ms per evaluation.
  - The registered drift rule marks the timing NOT_VALID: one stock repeat ran 5.7% faster the second time (the other three within 5%, all four identical). The decision is reported with that label.
  - Platform note: CPU `torch.randn` differs between x86-64 and aarch64, so the GH200 initializations are pinned separately; two earlier attempts were refused or failed before any scored task.
  - Table 13 is new; later tables move up by one. Two development measurements (a single-program two-loop, a four-GPU symmetric-buffer split) are stated as such.
- **Section 5.8: the H20 campaign on H100 (October 16).** The pre-registered, unchanged H20 campaign was run on one H100 node.
  - All 78 solves pass, with a CPU scalar-AD recheck.
  - GL8 still beats every baseline, but the ratios compress: Q10 1.66 → 1.22, collapsed Taylor 1.73 → 1.37, ED/torch AD/STDE 2.31–3.37 → 1.29–1.90.
  - The supplementary phase split shows why: a fixed L-BFGS cost of 6.3–9.0 ms per evaluation.
  - The chunk memory trade-off reproduces: an 85% lower setup peak at 16,384 points.
  - Table 12 is new, and the abstract and conclusion now state the H100 ratios beside the H20 ones.
- **Section 5.7: reduced precision on H100 (October 16).** A pre-registered study runs compiled Q10 and native GL8 in FP64, FP32 and FP32 with TF32 matrix products. The 2D source needed a 28-line patch that relaxes only its dtype checks.
  - FP32 suffices at these 1–2% gates: gradient error is 1.4–2.7e-7, and all 24 FP32 solves pass the unchanged FP64 acceptance. Every solve stops at the gates, so tighter accuracy was not tested.
  - TF32 does not: gradient error is 1.5–2.2e-4, and 0 of 24 solves pass, because L-BFGS stalls at about one iteration per block.
  - FP32 gives no solve gain at 2,048 points. At 65,536 points it shortens optimization by 1.53× (Q10) and 1.36× (GL8).
  - Native GL8 keeps its advantage in FP32: 1.67× over Q10 at 65,536 points.
  - Table 11 is new; the scope paragraph and the conclusion are updated.
- **Section 7.4: external derivative baselines in 3D (October 16).** On Ethier–Steinman, because C36 has no accepted solve, a pre-registered batch adds nested PyTorch autograd and the public collapsed Taylor-mode `jet` package. C17 reruns in the same batch.
  - All 18 solves pass.
  - C17's paired ratios are 1.90× against nested autograd and 1.08× (0.98–1.19) against collapsed `jet`. No arm was compiled, and no gain rule was registered.
  - Post hoc, the Adam-phase gap to `jet` is 1.37×; the fixed ~6 ms L-BFGS cost per evaluation compresses it.
  - Table 24 is new; the abstract, the 3D discussion, the limitations and the conclusion are updated. The C36 external-baseline item stays open.
- **Section 7.4: compiled variants (October 17).** A second pre-registered batch runs each method graph-only and compiled. Each method's variant is selected by its warm-up replay time.
  - C17's compiled variant failed CUDA Graph capture in all ten attempts, so C17 runs graph-only.
  - Against compiled nested AD, C17 is 1.59× faster (registered: C17 faster).
  - Against compiled collapsed `jet`, there is no decided difference in setup plus optimization (1.05). In optimization alone the jet is faster: it takes 0.81× C17's time.
  - The compiled jet's gradient phase is about 2.6× faster than eager C17.
  - The abstract, the 3D discussion, the limitations and the conclusion now state this; Table 25 is new.
- **Section 7.4: compiled C17, second attempt (October 17).** A one-GPU diagnostic found a compile route for C17 that survives CUDA Graph capture. A second pre-registered attempt then ran with every method compiled.
  - All 36 solves are accepted.
  - C17 is 1.68× faster than nested AD and 1.13× faster than collapsed `jet` in setup plus optimization; both are registered "C17 faster".
  - In optimization alone the jet is faster: C17 takes 1/0.96 ≈ 1.05× as long (registered "external faster"). C17's lead over it rests on cheaper setup, 3.5–4.0 s against 10.7–10.9 s.
  - Table 26 is new, and the abstract and conclusion follow this latest decision.
- **Section 6.10: a larger L-BFGS budget (also October 15–16).**
  - A resumed design (v4) is reported as cancelled before any physical observation, because a fresh L-BFGS history at loss ~1e-6 stalls.
  - Protocol v5 runs the development recipe with 4,800 L-BFGS calls. The loss and the physical errors stall soon after call 800, and stratified pressure ends at 2.2–3.0× its limit.
  - The registered outcome is E5, so no formal claim follows.
- **Sections 6.1, 6.8 and 6.10: what the source does and where the pressure error lies (October 18, CPU only).**
  - A check of the frozen source shows two properties. Outside the motion window the exact velocity and force vanish, so the four resting time slices that always pass carry no flow. Inside the moving box the force is a gradient.
  - A post-hoc decomposition of the v5 finals puts the plateau pressure at 4.4–6.6× its limit. The cutoff layers alone would still leave 1.5–1.9× its limit, so plateau-only weighting cannot pass.
- **Section 6.11: a wider cutoff (protocol v6, October 18).** A pre-registered batch runs the development recipe on a different instance with the cutoff width doubled to 1/16. That instance's layer forcing is 4–5× smaller.
  - All four runs reach the training gate after 110–152 L-BFGS calls.
  - At the final checkpoints stratified pressure stays at 2.0–2.9× its limit, about where it was at 1/32. The layers alone would still leave 1.7–2.2×.
  - The registered outcome is F3 (fail), so no formal claim follows. Table 22 is new, and the later tables move up by one.
- **Section 6.12: a dense quasi-Newton optimizer (protocol v7, October 21).** Only the L-BFGS phase is replaced, by self-scaled Broyden with a dense FP64 inverse Hessian (20.6 GB); everything else is the original eta = 1/32 recipe. The registered four-lane H100 batch could not be completed, so its two continuation arms were never run. A separate registration ran the two fresh runs on GH200.
  - Both runs train for the 4,000 s guard and reach losses of 1.7–2.9e-13.
  - Both final observations pass every acceptance test. Stratified pressure is about 1e-3 of its limit, and it is still the tightest test. The registered outcome is F1. These are development runs on seeds seen before.
  - The registered secondary readout counts evaluations to each loss threshold, against v5: 0.26–0.30 of v5's count at 3e-6. v5 never reaches 1e-6. This is a count, not a time ratio.
- **Section 6.13: a prospective five-seed time-to-accuracy cohort (October 21).** Five new seeds run as independent one-GPU jobs. Each is inspected first after 1,000 active training seconds.
  - Four of the five pass physical acceptance at that first inspection (stratified pressure 0.21–0.27 of its limit). Their candidate TTAs are 5,228–5,581 s, about 3,100–3,200 s of which is the observation itself. The fifth run was cancelled during its observation.
  - All four frozen CPU audits fail, at a relative tolerance on the spatial residual gradient. The registered outcome is therefore not qualified: 0/5, with no median.
  - A post-hoc diagnosis shows that the failing differences are at most 4 ulp of terms up to 1.4e5 that cancel to about 1e-2. All of them are within the standard rounding bound. The cause is the auditor's tolerance, not the physics. A qualified cohort needs a new registration.
  - Tables 23–24 are new, and the Ethier–Steinman tables move up by two (now 25–28). The abstract, the Ethier introduction, the 3D discussion and the conclusion are updated.
- **Section 6.14: a second, qualified cohort (October 21).** It was pre-registered in `1fd2bc6` with new seeds, a new holdout and an auditor that bounds the cancelled arrays by the a-priori rounding bound. The auditor was first validated on the v1 archives.
  - All five runs pass physical acceptance at the first inspection, and all five audits pass. The registered outcome is a qualified cohort, 5/5.
  - The median time to verified accuracy is 5,074 s (range 4,825–6,019 s) on one GH200, mostly spent in the physical observation. There is no baseline and no speedup claim.
  - Table 25 is new, and the Ethier–Steinman tables are now 26–29. The abstract, the contributions, the discussion and the conclusion now report this cohort.
- **Section 6.15: paired engines to verified accuracy (October 22).** A registered cohort (`aaea603`) runs the analytic and parameter-AD engines on five new seeds, both arms on one GH200 per seed. Each arm stops at a training loss of 1e-8 and is verified physically.
  - All ten arms reach the threshold and pass physical acceptance, and all audits pass. The registered outcome is analytic faster: the geometric mean ratio is 1.257 (range 1.249–1.269), inside the predicted 1.18–1.28.
  - Evaluation counts match (0.99–1.01), so the gain is per evaluation.
  - Table 26 is new, and the Ethier–Steinman tables are now 27–30. The abstract, contributions, discussion, conclusion and appendix are updated (48 pages).
- **Section 6.16: external derivative backends (October 22).** Nested PyTorch autograd, `torch.func` and a collapsed-`jet` adapter produce the same C36 state as the analytic engine. Per evaluation:
  - `torch.func` equals the analytic engine in eager mode (0.99) and is 3.72× slower as a CUDA Graph replay. The C36 advantage over it is therefore execution, not representation.
  - The jet adapter and nested autograd are 38× and 251× slower in eager mode.
  - The registered run did not complete: a driver bug dropped the analytic graph, and the job ran out of host memory while tracing the jet adapter. Table 27 reports a post-hoc rerun. The Ethier–Steinman tables are now 28–31 (49 pages).

[Evidence index](evidence-index.json) pins the source reports and their SHA256 values. [C36 reproduction instructions](../c36-potential-proof/README.md) describe the independent readers and protocols. The complete historical H20 artifacts remain preserved separately.

The three-dimensional problem definition explicitly attributes its mathematical
source to OpenAI Math376's *Incompressible Box Transport and Finite Computation*.
The paper gives the compact-curl construction and residual-defined fixed force,
identifies the single-box instance's scope, and cites the pinned original
geometry source. [The source-to-data mapping](../c36-potential-proof/math376-provenance.json)
records the matching mathematical commit and physical-source SHA256.

[中文思路摘要](flashns-summary.zh-CN.md) explains the method, validated H20 solve results and current H100 C36 scope. A new 2D H100 solve comparison remains pending and is not included as a result in this release.
