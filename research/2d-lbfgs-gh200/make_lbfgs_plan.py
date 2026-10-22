"""Build the frozen, pre-randomized task plan of the 2D L-BFGS GH200 study.

    python make_lbfgs_plan.py --protocol protocol-2d-lbfgs-gh200-20261023.json --output plan-2d-lbfgs-gh200-20261023.json

Tier 0: the 36 solves (2 cells x 3 sizes x 6 seeds) in one shuffled order (protocol randomization
seed). Tier 1: the first four tier-0 tasks again as drift repeats. Every task records its cell,
size, seed and the command line the driver runs.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def build(protocol):
    tasks = [{"id": f"{cell}-n{n}-s{seed}", "tier": 0, "cell": cell, "interior": n, "seed": seed}
             for cell in protocol["cells"] for n in protocol["interior_counts"] for seed in protocol["seeds"]]
    random.Random(protocol["randomization_seed"]).shuffle(tasks)
    drift = [dict(t, id="drift-" + t["id"], tier=1, repeat_of=t["id"]) for t in tasks[:4]]
    for t in tasks + drift:
        t["argv"] = ["light_solve.py", "--optimizer", t["cell"], "--backend", protocol["backend"],
                     "--interior-count", str(t["interior"]), "--seed", str(t["seed"])]
    return tasks + drift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    plan = {"protocol": args.protocol.name, "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            "version": protocol["version"], "tasks": build(protocol)}
    args.output.write_text(json.dumps(plan, indent=1) + "\n")
    print(len(plan["tasks"]), "tasks;", sum(t["tier"] == 1 for t in plan["tasks"]), "drift repeats;",
          "first four:", [t["id"] for t in plan["tasks"][:4]])


if __name__ == "__main__":
    main()
