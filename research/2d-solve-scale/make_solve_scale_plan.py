"""Build the frozen task plan of the 2D H100 solve-scale study.

    python3 make_solve_scale_plan.py PROTOCOL.json OUT_PLAN.json

One task per (cell, seed, interior count); sizes run in protocol order (16,384 first), shuffled within
a size with the protocol's fixed seed. Standard library only.
"""
import hashlib
import json
import random
import sys
from pathlib import Path


def main():
    protocol_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    protocol = json.loads(protocol_path.read_text())
    rng = random.Random(protocol["randomization_seed"])
    tasks = []
    for tier, n in enumerate(protocol["interior_counts"]):
        members = [{"id": f"{cell}-n{n}-s{seed}", "cell": cell, "backend": spec["backend"], "interior": n, "seed": seed, "tier": tier}
                   for cell, spec in protocol["cells"].items() for seed in protocol["seeds"]]
        rng.shuffle(members)
        tasks.extend(members)
    plan = {"version": protocol["version"] + "-plan", "protocol": protocol_path.name,
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "counts": {str(n): sum(t["interior"] == n for t in tasks) for n in protocol["interior_counts"]}, "tasks": tasks}
    out.write_text(json.dumps(plan, indent=1) + "\n")
    print(len(tasks), "tasks", plan["counts"])


if __name__ == "__main__":
    main()
