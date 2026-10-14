# FlashNS research manuscript

[PDF](flashns.pdf) and [editable standalone LaTeX](flashns.tex), updated October 14, 2026. Author and the original accepted 2D H20 experiment are retained from the October 6 release. Equations, tables, bibliography, the method figure and both original plot datasets are embedded in the source; no external TeX inputs are needed. The PDF is built directly with pdfLaTeX (three passes) and needs only standard TeX Live packages: TikZ, booktabs, longtable, hyperref and Latin Modern.

The October 14 revision restructures the paper after an external review without changing any measured value, gate, seed or evidence file: a shorter abstract, an explicit contribution list, a terminology table, a related-work section with five additional verified references (Forward Laplacian, Taylor-mode AD in JAX, Griewank and Walther, NSFnets, Neural Conservation Laws), a method figure, a paragraph on what the 2D solve times measure, a diagnosis of the failed 3D acceptance, and an appendix that now holds the execution and provenance records. Two clarifications come from the raw protocol rather than new runs: the 2D closure counts include the 4,000 Adam evaluations (every accepted 2D solve stops during L-BFGS after 2,208–5,259 L-BFGS evaluations), and the relative L2 error of v is the binding 2D acceptance criterion for all nine methods.

The new C36 section reports complete 4,999-point / 50,692-parameter checks, paired H100 and CPU objective-gradient comparisons, all four frozen-budget training failures, and complete physical acceptance failures. Older H100 packed-gather rows remain about 3% slower than their dense-activation parameter-AD control. The subsequent packed-activation/analytic-transpose CUDA Graph experiments give 1.044582× with eager activation and **1.323962× with compiled activation** (95% paired interval 1.323087–1.324762). Both arms use the same packed forward and compilation choice; the compiled candidate/control means are 164.484/217.771 ms. Both GPU experiments pass 78 numerical comparisons and independent remote/local readers of 156 arrays and 20 timing windows. The separate one-thread CPU result is 1.265×. The older 1,255× native comparison uses a different source and its original recursive cache. No accepted 3D time-to-accuracy improvement is established.

[Evidence index](evidence-index.json) pins the source reports and their SHA256 values. [C36 reproduction instructions](../c36-potential-proof/README.md) describe the independent readers and protocols. The complete historical H20 artifacts remain preserved separately.

The three-dimensional problem definition explicitly attributes its mathematical
source to OpenAI Math376's *Incompressible Box Transport and Finite Computation*.
The paper gives the compact-curl construction and residual-defined fixed force,
identifies the single-box instance's scope, and cites the pinned original
geometry source. [The source-to-data mapping](../c36-potential-proof/math376-provenance.json)
records the matching mathematical commit and physical-source SHA256.

[中文思路摘要](flashns-summary.zh-CN.md) explains the method, validated H20 solve results and current H100 C36 scope. A new 2D H100 solve comparison remains pending and is not included as a result in this release.
