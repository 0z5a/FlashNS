# Complete Kovasznay PINN solves

See the [fixed problem, acceptance criteria, results and reproduction commands](../../docs/pinn-solver-results.md).

All eight backends use the same full physical loss and initialization. `preflight.py` verifies loss, every parameter gradient and actual common Adam updates. `run_suite.py` freezes the pilot-tested protocol and runs 24 fresh-process solves in a fixed randomized order. `extend_suite.py` preserves the original capped outcomes and offers one uniform larger iteration cap, restarting only previously nonconverged cases.

Every solve records setup and total wall time, all stop checks, an additional independent scalar-AD acceptance, a parameter checkpoint, input hashes and immutable executed-source versions. No acceptance threshold was relaxed after timing.
