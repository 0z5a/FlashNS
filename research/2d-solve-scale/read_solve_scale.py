"""Reader and registered analysis of the 2D H100 solve-scale study.

    python3 read_solve_scale.py --plan PLAN --protocol PROTOCOL --results DIR --state DRIVER_STATE --output OUT

Per interior count: acceptance and evaluation counts per cell, paired geometric means (setup plus
optimization, optimization only, optimization per evaluation) for A/C, A/B and B/C over jointly
accepted seeds, the registered decision, and a descriptive Adam/L-BFGS phase split from the progress
rows. Standard library only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

PAIRS = (("A", "C"), ("A", "B"), ("B", "C"))


def gm(values):
    logs = [math.log(v) for v in values]
    return {"geometric_mean": math.exp(sum(logs) / len(logs)), "range": [min(values), max(values)]} if values else None


def phases(progress):
    initial = progress[0]
    adam = [p for p in progress if p["phase"] == "adam"]
    last = progress[-1]
    out = {}
    if adam:
        a = adam[-1]
        out["adam_ms_per_evaluation"] = 1000 * (a["optimization_wall_seconds"] - initial["optimization_wall_seconds"]) / max(1, a["loss_gradient_evaluations"])
        evals = last["loss_gradient_evaluations"] - a["loss_gradient_evaluations"]
        if evals > 0:
            out["lbfgs_ms_per_evaluation"] = 1000 * (last["optimization_wall_seconds"] - a["optimization_wall_seconds"]) / evals
            out["lbfgs_evaluations"] = evals
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "protocol", "results", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    protocol = json.loads(args.protocol.read_text())
    state = json.loads(args.state.read_text())
    rows = []
    for t in plan["tasks"]:
        record = state["tasks"].get(t["id"], {})
        row = {"id": t["id"], "cell": t["cell"], "interior": t["interior"], "seed": t["seed"], "lane": record.get("lane"),
               "driver_state": record.get("state"), "process_seconds": record.get("seconds")}
        path = args.results / f"{t['id']}.json"
        if path.exists():
            report = json.loads(path.read_text())
            row["status"] = report.get("status")
            row["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if not args.self_test and "solve" in report:
                s = report["solve"]
                assert report["problem"]["interior_count"] == t["interior"], t["id"]
                assert s["parameter_initial_sha256"] == protocol["frozen_inputs"]["theta_sha256"][str(t["seed"])], t["id"]
                row.update(converged=bool(s["converged"]), setup_seconds=s["setup_seconds"], optimization_seconds=s["optimization_wall_seconds"],
                           setup_plus_optimization_seconds=s["setup_plus_optimization_seconds"], evaluations=s["loss_gradient_evaluations"],
                           adam_updates=s["adam_updates"], lbfgs_iterations=s["lbfgs_iterations"], **phases(report["progress"]))
                row["optimization_ms_per_evaluation"] = 1000 * row["optimization_seconds"] / row["evaluations"]
            if report.get("status") != "passed":
                row["error_tail"] = str(report.get("error"))[-600:]
        rows.append(row)
    result = {"protocol_version": protocol["version"], "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              "tasks": len(plan["tasks"]), "driver_counts": state.get("counts"), "rows": rows}
    if args.self_test:
        result["self_test"] = True
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({"tasks": result["tasks"], "results": sum("status" in r for r in rows)}))
        return
    sizes = {}
    for n in protocol["interior_counts"]:
        by = {(r["cell"], r["seed"]): r for r in rows if r["interior"] == n and r.get("converged")}
        cells = {c: {"accepted": sum((c, s) in by for s in protocol["seeds"]),
                     "evaluations": sorted(by[(c, s)]["evaluations"] for s in protocol["seeds"] if (c, s) in by),
                     "setup_seconds": sorted(round(by[(c, s)]["setup_seconds"], 3) for s in protocol["seeds"] if (c, s) in by),
                     "adam_ms_per_evaluation": sorted(round(by[(c, s)].get("adam_ms_per_evaluation", float("nan")), 4) for s in protocol["seeds"] if (c, s) in by),
                     "lbfgs_ms_per_evaluation": sorted(round(by[(c, s)].get("lbfgs_ms_per_evaluation", float("nan")), 4) for s in protocol["seeds"] if (c, s) in by)}
                 for c in protocol["cells"]}
        pairs = {}
        for a, b in PAIRS:
            joint = [s for s in protocol["seeds"] if (a, s) in by and (b, s) in by]
            ratio = lambda key: [by[(a, s)][key] / by[(b, s)][key] for s in joint]
            opt = ratio("optimization_seconds")
            favour = sum(r > 1 for r in opt)
            entry = {"jointly_accepted": len(joint), "favour_second": favour,
                     "setup_plus_optimization": gm(ratio("setup_plus_optimization_seconds")),
                     "optimization_only": gm(opt), "optimization_per_evaluation": gm(ratio("optimization_ms_per_evaluation"))}
            if len(joint) < 5:
                entry["decision"] = "not decidable (fewer than 5 jointly accepted pairs)"
            elif entry["optimization_only"]["geometric_mean"] > 1.03 and favour >= 5:
                entry["decision"] = f"solve-level gain of {b} over {a}"
            else:
                entry["decision"] = "no solve-level gain"
            pairs[f"{a}/{b}"] = entry
        sizes[str(n)] = {"cells": cells, "pairs": pairs}
    result["sizes"] = sizes
    result["failures"] = [r for r in rows if r.get("status") != "passed"]
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({n: {p: (round(v["optimization_only"]["geometric_mean"], 3) if v["optimization_only"] else None, v["decision"])
                          for p, v in d["pairs"].items()} for n, d in sizes.items()}))


if __name__ == "__main__":
    main()
