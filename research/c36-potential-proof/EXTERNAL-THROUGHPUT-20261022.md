# C36 external derivative backends: registered throughput comparison

Registered before its run. The specification is [protocol-external-throughput-20261022.json](protocol-external-throughput-20261022.json); the driver is [`tools/c36_external_throughput.py`](tools/c36_external_throughput.py).

**Question.** How fast is the complete C36 objective, with all 50,692 parameter gradients, when the coordinate derivatives come from public automatic-differentiation tools instead of the C36 contracted state?

**Methods.** All three external backends ([`box376.external_c36`](src/box376/external_c36.py)) produce the same 36-channel network state as the analytic engine. Gate, curl, consumer and loss are the same C36 code in every method, and every external parameter gradient is autograd.
- **Nested AD.** Nested scalar-coordinate PyTorch autograd over the batch, as an ordinary PINN code writes it.
- **torch.func.** `jacfwd` nested to third order, plus the Hessian of the spatial Laplacian, vectorized with `vmap`.
- **Collapsed jet.** Dangel et al.'s `jet` at the commit pinned in the protocol. Collapsed Taylor mode gives the value, first derivatives and spatial Laplacian, and a first-order jet gives `dt`. The remaining derivatives use autograd, as in the Ethier and 2D adapters.

**One function for all three backends.** The prepared geometry features are affine in space, and the state needs one time derivative. A per-point polynomial built from the prepared geometry state therefore has the exact derivatives the state needs. Every backend differentiates that same function.
- [CPU tests](src/box376/test_external_c36.py) check each backend against the analytic engine on six training points: state within 7.2e-4 of the original budget, complete parameter gradient within 0.09.
- [Fixtures](src/box376/test_external_compile.py) check that the traced routes reproduce the eager result after an in-place parameter change.

**Variants and selection.**
- **Candidates.** Analytic: eager, compiled, graph. torch.func and jet: eager, traced (`make_fx`, the Ethier pattern), traced captured in a CUDA Graph. Nested AD: eager only.
- **Selection.** Each candidate is checked against dense parameter AD and timed on a development initialization. Each method's fastest passing candidate is selected. Failures are recorded and dropped.
- **Excluded in development.** Two compiled routes are excluded, with reasons recorded in the protocol: `torch.compile` of the trace fails at call time, and the functional `grad_and_value` route hits an internal PyTorch assertion.

**Measurement.** Timing uses the two stored initializations. Two ratios are reported per method:
- **Matched eager:** eager against eager. It separates representation from execution mode.
- **Best against best:** selected variant against selected variant.

**Registered predictions.** The matched-eager ratio should be about 1.0 for torch.func, 8.6 for jet and 91 for nested AD. Best against best is not predicted.

**Feasibility rule.** For each method, the projected solve takes 7,000 evaluations at the selected variant's speed, plus 7,800 s for observations and setup.
- If that fits one 5 h job, a paired solve can be proposed as a separate registration, and only with the user's go-ahead.
- Otherwise only the projection is reported.

**Caveats (registered).**
- In the jet adapter only the first two orders use Taylor mode. The third derivatives, the Hessian of the Laplacian and the time-mixed terms are nested autograd, and they dominate its cost.
- Nested AD needs about 96 GB at chunk 512.
