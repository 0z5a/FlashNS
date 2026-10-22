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
