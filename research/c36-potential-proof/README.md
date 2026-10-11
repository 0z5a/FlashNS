# C36 potential gradient proof

This research snapshot compares the complete 50,692-parameter C36 gradient with independent native automatic differentiation. It uses two seeds and 17 fixed interior points from the full physical source cache, float64, and the unchanged `atol=1e-11`, `rtol=1e-9` budget.

Run from this directory with NumPy and PyTorch installed:

```sh
PYTHONPATH=src python -m box376.verify_potential_gradient --inputs data --output outputs/subset
```

The script preserves raw arrays and failure traces. This subset proof does not establish the complete 4096-point objective, GPU qualification, training convergence, physical acceptance, or performance. Results are recorded separately.
