# FlashNS research manuscript

[PDF](flashns.pdf) and [editable standalone LaTeX](flashns.tex), updated October 13, 2026. Author and the original accepted 2D H20 experiment are retained from the October 6 release. Equations, tables, bibliography and both original plot datasets are embedded in the source; no external TeX inputs are needed.

The new C36 section reports complete 4,999-point / 50,692-parameter checks, paired H100 and CPU objective-gradient comparisons, all four frozen-budget training failures, and complete physical acceptance failures. The 1.265× compiled result is CPU-only. Older H100 packed-gather rows remain about 3% slower than their dense-activation parameter-AD control; the subsequent eager packed-activation/analytic-transpose CUDA Graph experiment gives 1.044582× against a matched packed-forward parameter-AD control. The older 1,255× native comparison uses a different source and its original recursive cache. No accepted 3D time-to-accuracy improvement is established.

[Evidence index](evidence-index.json) pins the source reports and their SHA256 values. [C36 reproduction instructions](../c36-potential-proof/README.md) describe the independent readers and protocols. The complete historical H20 artifacts remain preserved separately.
