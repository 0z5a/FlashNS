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
  - The solve-level gain does not carry over to H100 at the original size, because optimizer overhead dominates. The contributions, discussion and conclusion are updated accordingly.
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
