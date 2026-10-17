"""Build the frozen task plan of the 2D precision study.

    python3 make_precision_plan.py PROTOCOL.json OUT_PLAN.json

Tier 0: solves and replays at 2,048 points; tier 1: at 65,536 points; tier 2: replays at 16,384
points. Tasks are shuffled within a tier with the protocol's fixed seed. Standard library only.
"""
import hashlib
import json
import random
import sys
from pathlib import Path


def main():
    protocol_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    p = json.loads(protocol_path.read_text())
    rng = random.Random(p["randomization_seed"])
    tiers = {0: [], 1: [], 2: []}
    for cell, spec in p["cells"].items():
        for precision in p["precisions"]:
            for n in p["solves"]["interior_counts"]:
                for seed in p["solves"]["seeds"]:
                    tiers[0 if n == 2048 else 1].append({"id": f"solve-{cell}-{precision}-n{n}-s{seed}", "kind": "solve", "cell": cell,
                                                        "backend": spec["backend"], "precision": precision, "interior": n, "seed": seed})
            for n in p["replay"]["interior_counts"]:
                for r in range(p["replay"]["replicates"]):
                    tiers[{2048: 0, 65536: 1, 16384: 2}[n]].append({"id": f"replay-{cell}-{precision}-n{n}-r{r}", "kind": "replay", "cell": cell,
                                                                   "backend": spec["backend"], "precision": precision, "interior": n,
                                                                   "seed": p["replay"]["seed"], "replicate": r})
    tasks = []
    for tier in sorted(tiers):
        members = tiers[tier]
        rng.shuffle(members)
        tasks.extend(dict(t, tier=tier) for t in members)
    plan = {"version": p["version"] + "-plan", "protocol": protocol_path.name,
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "counts": {k: sum(t["kind"] == k for t in tasks) for k in ("solve", "replay")},
            "tiers": {str(k): len(v) for k, v in tiers.items()}, "tasks": tasks}
    out.write_text(json.dumps(plan, indent=1) + "\n")
    print(len(tasks), "tasks", plan["counts"], plan["tiers"])


if __name__ == "__main__":
    main()
