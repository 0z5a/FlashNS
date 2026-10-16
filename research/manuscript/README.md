# FlashNS research manuscript

[PDF](flashns.pdf) and [editable standalone LaTeX](flashns.tex), updated October 15, 2026. Author and the original accepted 2D H20 experiment are retained from the October 6 release. Equations, tables, bibliography, the method figure and both original plot datasets are embedded in the source; no external TeX inputs are needed. The PDF is built directly with pdfLaTeX (three passes) and needs only standard TeX Live packages: TikZ, booktabs, longtable, hyperref and Latin Modern.

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
  - Table 23 is new; the abstract, the 3D discussion, the limitations and the conclusion are updated. The C36 external-baseline item stays open.
- **Section 6.10: a larger L-BFGS budget (also October 15–16).**
  - A resumed design (v4) is reported as cancelled before any physical observation, because a fresh L-BFGS history at loss ~1e-6 stalls.
  - Protocol v5 runs the development recipe with 4,800 L-BFGS calls. The loss and the physical errors stall soon after call 800, and stratified pressure ends at 2.2–3.0× its limit.
  - The registered outcome is E5, so no formal claim follows.

[Evidence index](evidence-index.json) pins the source reports and their SHA256 values. [C36 reproduction instructions](../c36-potential-proof/README.md) describe the independent readers and protocols. The complete historical H20 artifacts remain preserved separately.

The three-dimensional problem definition explicitly attributes its mathematical
source to OpenAI Math376's *Incompressible Box Transport and Finite Computation*.
The paper gives the compact-curl construction and residual-defined fixed force,
identifies the single-box instance's scope, and cites the pinned original
geometry source. [The source-to-data mapping](../c36-potential-proof/math376-provenance.json)
records the matching mathematical commit and physical-source SHA256.

[中文思路摘要](flashns-summary.zh-CN.md) explains the method, validated H20 solve results and current H100 C36 scope. A new 2D H100 solve comparison remains pending and is not included as a result in this release.
