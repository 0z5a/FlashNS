# H100 NVL follow-up evidence

Three separate completed workloads are described in:

- [Hopper resources and copy-pipeline control](../../docs/cuda-hopper-results.md)
- [Official CPU build, Comparator replay and finite local amplitude adapter](../../docs/openai-ns-formal-results.md)
- [Full PINN solves with fixed acceptance criteria](../../docs/pinn-solver-results.md)

`collect_remote.py` preserves new experiment artifacts, executed sources, native baseline binaries, failed attempts and logs. It excludes installed runtime wheels, compiler caches and the full CUTLASS checkout; their versions or hashes remain in the measurement records. The official Lean source checkout and PDF are separately pinned under `sources/openai_ns_formal`.

After retrieving the evidence:

```bash
.venv/bin/python3.12 experiments/hopper_followup/audit.py
uv run --no-project --python .venv/bin/python3.12 --with matplotlib==3.10.8 python experiments/hopper_followup/summarize.py
```

The audit verifies source and own-binary references, solver protocols, every checkpoint, paired initialization/data, the overlapping capped/extended trajectories, and the final independent acceptance criteria. The figures are standalone PNG/PDF/SVG artifacts, with raw curves retained in each solve's JSON. The original 24-run fixed-budget suite remains separate from the seven additional full reruns under a uniformly larger iteration cap.

Bitwise trajectory replay is recorded separately from numerical acceptance. Six fresh extension runs match their overlapping original metric histories exactly; PhysicsNeMo shows a tiny difference at Adam step 100 and subsequently a different L-BFGS trajectory despite identical source, data and initialization. All final runs satisfy the common independent acceptance.

For the finite amplitude performance retest, use `NUMPY_MADVISE_HUGEPAGE=0` before importing NumPy. The default-hugepage runs and the diagnostic record remain archived. This is a per-process allocation hint; the experiment did not alter the host kernel or hugepage settings.
