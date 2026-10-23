# Paired C36 engine cohort: analytic vs parameter AD (registered)

This cohort is registered before any of its runs. The machine-readable specification is [protocol-formal-pair-20261021.json](protocol-formal-pair-20261021.json). It answers the question the qualified TTA cohort ([FORMAL-TTA-V2-20261021.md](FORMAL-TTA-V2-20261021.md)) leaves open: does the analytic engine reach verified accuracy sooner than the parameter-AD control?

**Design.** Five previously unused seeds (1344954027, 164034124, 2110545488, 911819800, 1176150139; new holdout 1944134319) come from the same SHA-256 rule with the label `flashns-c36-formal-pair-20261021`.
- **Arms.** Each seed runs twice, once per engine, one after the other on the same GH200 in one job. The order alternates across seeds, analytic first for seeds 1, 3 and 5.
- **Recipe.** Everything else is the TTA cohort's: data, network, Adam4000 and dense SSBroyden, checks, observer and auditor revision 2. The analytic arm also runs the native subset check; that check covers the analytic eager path and does not depend on the engine.
- **Stopping.** Each arm stops at the first accepted SSBroyden iterate with a loss of at most 1e-8, or at a 2,500 s training guard. The threshold is informed by development data: the nine audited passing first inspections lie at 6.7e-9 to 1.7e-8. The threshold checkpoint is then observed once; an arm that misses the threshold is not observed.

**Primary measurement and decision.** The primary measurement is each arm's active training seconds at the threshold iterate, recorded before any check of that iterate. The ratio is parameter AD over analytic, per seed.
- **Jointly accepted pair.** Both arms reach the threshold, both threshold checkpoints pass physical acceptance, and both independent audits pass.
- **Outcomes** ([`decide_c36_pair`](src/box376/decide_c36_pair.py), with tests):
  - **Analytic faster:** at least 4 jointly accepted pairs, a geometric mean ratio above 1.03, and at least 4 pairs above 1.
  - **Parameter AD faster:** the mirror case.
  - **No decided difference:** otherwise, with at least 4 pairs.
  - **Insufficient pairs:** fewer than 4.
- **Secondary measurements.** Seconds per evaluation in each phase, evaluations to the threshold, and setup.

**Prediction (registered).** The ratio should be about 1.23 (1.18–1.28). The [GH200 qualification](pair-gh200-qualification-20261021.json) on development seed 361901 measured 1.25 per Adam evaluation and 1.21 per SSBroyden evaluation. Both engines passed every check there. With the 20.6 GB inverse Hessian resident, both peak at 20.9 GB.

**Execution.** Each seed is one independent single-GPU job with a 5 h ceiling and an expected 3 h. The job starts its frozen driver itself, archives every outcome and exits. Both arms on one card means no card waits idle.
