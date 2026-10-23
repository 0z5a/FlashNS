# Paired C36 cohort: analytic engine vs torch.func (registered)

Registered before any of its runs. The specification is [protocol-formal-pair-func-20261022.json](protocol-formal-pair-func-20261022.json). It extends the external throughput comparison ([EXTERNAL-THROUGHPUT-20261022.md](EXTERNAL-THROUGHPUT-20261022.md)) to whole solves. In that comparison torch.func matched the analytic engine in eager mode and was 3.7 times slower as a CUDA Graph replay, and a full solve with it fitted one job.

**Design.** Five previously unused seeds (232943677, 1952762959, 1212253177, 190009641, 101379141; new holdout 567563932).
- **Arms.** Each seed runs the analytic engine and the torch.func backend one after the other on the same GH200, in alternating order. Both arms train on a CUDA Graph replay. The torch.func arm captures its eager evaluation at chunk 1024; make_fx-traced versions held about 92 GB in the graph pool and left no room for the inverse Hessian.
- **Unchanged from the earlier paired cohort.** Everything else is as in [FORMAL-PAIR-20261021.md](FORMAL-PAIR-20261021.md):
  - the 1e-8 loss threshold, with each threshold checkpoint observed once;
  - the independent audit, here [`audit_c36_pair_func`](src/box376/audit_c36_pair_func.py), which accounts for the torch.func arm having no compiled stage;
  - the decision rule ([`decide_c36_pair_func`](src/box376/decide_c36_pair_func.py)): margin 1.03, at least 4 jointly accepted pairs, at least 4 agreeing.
- **Budget.** The training guard is 6,000 s and the job ceiling 7 h.

**Prediction (registered).** The ratio of torch.func to analytic active training seconds to the threshold should be about 3.83 (3.6–4.1). It rests on the [GH200 qualification](pair-func-gh200-qualification-20261022.json) on development seed 361901: 3.98 per Adam evaluation, 3.65 per SSBroyden evaluation, and 90.8 GB free before the inverse Hessian.

## Results (2026-10-23)

All five jobs ran once on one GH200 each (11,716–14,316 s wall, 17.76 allocated GPU-hours in total) and completed; every seed's archive (247–249 members) passed the independent CPU audit (`PASS_INDEPENDENT_ARCHIVE_READBACK`, 166 frozen files and the protocol/manifest hashes verified). Both arms of every seed reached the loss threshold 1e-8, and all ten threshold checkpoints pass physical acceptance (worst test: stratified corrected pressure at 0.21–0.26 of its limit; stratified velocity relative L2 7.7–8.6e-5; trajectory endpoint 0.03–0.06 of its limit).

| Seed | Analytic s | Analytic SSBroyden eval. | Analytic S press. | torch.func s | torch.func eval. | torch.func S press. | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 232943677 | 981 | 2,956 | 0.257 | 3,878 | 2,924 | 0.230 | 3.953 |
| 1952762959 | 1,006 | 3,007 | 0.235 | 3,919 | 2,988 | 0.236 | 3.895 |
| 1212253177 | 1,010 | 2,986 | 0.231 | 3,912 | 2,982 | 0.250 | 3.873 |
| 190009641 | 996 | 2,998 | 0.217 | 3,921 | 3,000 | 0.217 | 3.936 |
| 101379141 | 958 | 2,774 | 0.229 | 3,797 | 2,772 | 0.213 | 3.962 |

- **Registered decision** ([c36-pair-func-cohort-20261022.json](c36-pair-func-cohort-20261022.json), `decide_c36_pair_func`): **ANALYTIC_FASTER**, 5/5 jointly accepted pairs, geometric mean ratio **3.924** (range 3.873–3.962), all five ratios above 1; inside the registered prediction 3.83 (3.6–4.1).
- **Where the ratio comes from.** The two arms need nearly the same number of SSBroyden evaluations (torch.func/analytic 0.99–1.00) and 4,000 Adam updates each, so the gain is per evaluation: torch.func takes 4.0–4.1 times as long per Adam evaluation (0.553–0.554 s against 0.135–0.138 s) and 3.7–3.8 times per SSBroyden evaluation (0.566–0.567 against 0.147–0.152 s, which includes the shared dense SSBroyden update). Both arms peak at 20.9 GB with the inverse Hessian resident. Setup: analytic 196–229 s (eager and dense AD 85–103, compile and warm-up 106–120, capture 4–5), torch.func 94–117 s (no compiled stage; capture 9–11).
- **Start-up checks.** Every arm agrees with dense parameter AD at the initialization and after a parameter change within 3.2e-6 of the original budget (analytic graph 5.1e-7–1.8e-6; torch.func graph 8.1e-7–2.3e-6), and the analytic arm's native-subset check within 3.1e-7–8.8e-7.
- **Post hoc, not registered.** The analytic arm runs at the protocol's `explicit_chunk` 128. A development profile on seed 361901 ([c36-engine-profile-20261022.json](c36-engine-profile-20261022.json), `tools/c36_engine_profile.py`) found the analytic graph replay 3.17 times faster at chunk 2048 (0.1317 → 0.0416 s per evaluation; 60,702 → 6,837 kernel launches), while the torch.func arm already runs near its best chunk (0.515–0.624 s at 512–2048). The registered ratio therefore understates the analytic engine's advantage at its best chunk; the protocol fixed chunk 128 before this profile existed, and the ratio is reported as registered.
- **Scope.** As in the earlier paired cohort, the physical observation (2,661–4,048 s per arm, the same native-AD observer in both arms) is excluded from the ratio.
