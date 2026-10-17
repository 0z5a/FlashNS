"""Build the frozen, pre-randomized task plan of the 2D representation/execution ablation.

    python3 make_ablation_plan.py PROTOCOL.json OUT_PLAN.json

Tiers run in order (core decomposition first, so a deadline cut never removes it); the order
inside each tier is shuffled with the protocol's fixed seed. Lanes pull from one shared queue,
so the lane of every task is recorded rather than assigned. Standard library only.
"""
import hashlib
import json
import random
import sys
from pathlib import Path


def task(cell, cells, interior, hidden, replicate, *, kind="replay", seed, profile=False, label=None):
    backend, eager = cells[cell]["backend"], cells[cell]["eager"]
    label = label or cell
    name = f"{label}-n{interior}-h{'x'.join(map(str, hidden))}-r{replicate}" if kind == "replay" else f"{label}-solve-s{seed}"
    return {"id": name, "kind": kind, "cell": cell, "label": label, "backend": backend, "eager": eager,
            "interior": interior, "hidden": list(hidden), "replicate": replicate, "seed": seed, "profile": profile}


def main():
    protocol_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    protocol = json.loads(protocol_path.read_text())
    cells = protocol["cells"]
    rng = random.Random(protocol["randomization_seed"])
    base = tuple(protocol["original_hidden"])
    seed = protocol["timing_seed"]
    tiers = []
    core = []
    for interior in protocol["core_points"]:
        for cell in protocol["core_cells"]:
            for r in range(protocol["replicates"]["core"]):
                core.append(task(cell, cells, interior, base, r, seed=seed, profile=(r == 0)))
        for r in range(protocol["replicates"]["core"]):
            core.append(task(protocol["aa_cell"], cells, interior, base, r, seed=seed, label=protocol["aa_cell"] + "_AA"))
    tiers.append(core)
    sweep = [task(cell, cells, interior, base, r, seed=seed)
             for interior in protocol["sweep_points"] for cell in protocol["sweep_cells"]
             for r in range(protocol["replicates"]["sweep"])]
    tiers.append(sweep)
    shapes = [task(cell, cells, interior, tuple(hidden), r, seed=seed)
              for hidden in protocol["shape_hidden"] for interior in protocol["shape_points"]
              for cell in protocol["shape_cells"] for r in range(protocol["replicates"]["shape"])]
    tiers.append(shapes)
    solves = [task(cell, cells, protocol["original_points"], base, 0, kind="solve", seed=s)
              for cell in protocol["solve_cells"] for s in protocol["solve_seeds"]]
    tiers.append(solves)
    tasks = []
    for tier, members in enumerate(tiers):
        rng.shuffle(members)
        for t in members:
            t["tier"] = tier
        tasks.extend(members)
    # Endpoint drift: repeat the first core task of each lane-sized block at the very end.
    for t in [dict(x) for x in tasks[:protocol["drift_repeats"]]]:
        t.update(id=t["id"] + "-drift", label=t["label"] + "_DRIFT", tier=len(tiers), profile=False, drift_of=t["id"])
        tasks.append(t)
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    plan = {"version": protocol["version"] + "-plan", "protocol": protocol_path.name,
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "tiers": ["core decomposition (2k, 16k) with A/A label control and profiles", "point sweep",
                      "width/depth", "development solves", "endpoint drift repeats"],
            "counts": {str(i): sum(t["tier"] == i for t in tasks) for i in range(len(tiers) + 1)},
            "tasks": tasks}
    out.write_text(json.dumps(plan, indent=1) + "\n")
    print(len(tasks), "tasks", plan["counts"])


if __name__ == "__main__":
    main()
