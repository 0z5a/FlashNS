"""Supplementary reader of the 2D H100 retest under the H20 protocol (registered before the run).

    python3 read_h20_h100_supplement.py --root BATCH_ROOT --output OUT.json

The primary analysis is the unchanged audit_h20.py (paired setup-plus-optimization geometric means
against gl8_native over jointly accepted, CPU-rechecked solves). This reader adds, for the same pairs:
optimization-only and per-evaluation ratios, the Adam/L-BFGS phase cost per evaluation of every solve,
and each solve's compiler-cache state, i.e. how many earlier cases of the same backend and compile
mode ran on its lane (qualification included). Standard library only; writes no absolute paths.
"""
import argparse
import json
import math
from pathlib import Path

CANDIDATE = "gl8_native"


def phases(progress):
    rows = [p for p in progress if "optimization_wall_seconds" in p and "loss_gradient_evaluations" in p]
    out = {}
    if not rows:
        return out
    adam = [p for p in rows if p.get("phase") == "adam"]
    initial, last = rows[0], rows[-1]
    if adam:
        a = adam[-1]
        out["adam_ms_per_evaluation"] = 1000 * (a["optimization_wall_seconds"] - initial["optimization_wall_seconds"]) / max(1, a["loss_gradient_evaluations"])
        evaluations = last["loss_gradient_evaluations"] - a["loss_gradient_evaluations"]
        if evaluations > 0:
            out["lbfgs_ms_per_evaluation"] = 1000 * (last["optimization_wall_seconds"] - a["optimization_wall_seconds"]) / evaluations
            out["lbfgs_evaluations"] = evaluations
    return out


def geometric(values):
    if not values:
        return None
    logs = [math.log(v) for v in values]
    return {"geometric_mean": math.exp(sum(logs) / len(logs)), "range": [min(values), max(values)], "pairs": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.root / "audit.json").read_text())
    accepted = {(a["lane"], a["name"]) for a in audit["attempts"] if a.get("outcome") == "ACCEPTED_CPU_RECHECKED"}
    solves, lanes = [], {}
    for index in range(4):
        result_path = args.root / ("lane" + str(index)) / "result.json"
        if not result_path.is_file():
            continue
        rows = [r for r in json.loads(result_path.read_text())["rows"] if "started" in r]
        rows.sort(key=lambda r: r["started"])
        lanes[str(index)] = len(rows)
        for position, row in enumerate(rows):
            case = row["case"]
            if not case.get("solve"):
                continue
            mode = bool(case.get("compiled"))
            prior = sum(1 for r in rows[:position] if r["case"]["backend"] == case["backend"] and bool(r["case"].get("compiled")) == mode)
            entry = {"lane": index, "name": row["name"], "backend": case["backend"], "seed": case["seed"], "compiled": mode,
                     "position_on_lane": position, "earlier_cases_same_backend_and_mode": prior,
                     "accepted": (index, row["name"]) in accepted}
            path = args.root / ("lane" + str(index)) / (row["name"] + ".json")
            if path.is_file():
                report = json.loads(path.read_text())
                s = report.get("solve", {})
                if "optimization_wall_seconds" in s:
                    entry.update(setup_seconds=s["setup_seconds"], optimization_seconds=s["optimization_wall_seconds"],
                                 evaluations=s["loss_gradient_evaluations"],
                                 optimization_ms_per_evaluation=1000 * s["optimization_wall_seconds"] / s["loss_gradient_evaluations"],
                                 **phases(report.get("progress", [])))
            solves.append(entry)
    comparisons = []
    for lane in sorted({s["lane"] for s in solves}):
        here = [s for s in solves if s["lane"] == lane and s["accepted"] and "optimization_seconds" in s]
        for backend in sorted({s["backend"] for s in here} - {CANDIDATE}):
            metrics = {"setup_plus_optimization": [], "optimization_only": [], "optimization_per_evaluation": []}
            for seed in sorted({s["seed"] for s in here}):
                a = next((s for s in here if s["seed"] == seed and s["backend"] == CANDIDATE), None)
                b = next((s for s in here if s["seed"] == seed and s["backend"] == backend), None)
                if a and b:
                    metrics["setup_plus_optimization"].append((b["setup_seconds"] + b["optimization_seconds"]) / (a["setup_seconds"] + a["optimization_seconds"]))
                    metrics["optimization_only"].append(b["optimization_seconds"] / a["optimization_seconds"])
                    metrics["optimization_per_evaluation"].append(b["optimization_ms_per_evaluation"] / a["optimization_ms_per_evaluation"])
            comparisons.append({"lane": lane, "baseline": backend, "candidate": CANDIDATE,
                                **{k: geometric(v) for k, v in metrics.items()}})
    backends = {}
    for s in solves:
        if "optimization_seconds" not in s:
            continue
        key = f"{s['lane']}|{s['backend']}"
        b = backends.setdefault(key, {"solves": 0, "accepted": 0})
        b["solves"] += 1
        b["accepted"] += int(s["accepted"])
        for field in ("setup_seconds", "adam_ms_per_evaluation", "lbfgs_ms_per_evaluation"):
            values = [x[field] for x in solves if f"{x['lane']}|{x['backend']}" == key and field in x]
            b[field] = [min(values), max(values)] if values else None
        first = [x["setup_seconds"] for x in solves if f"{x['lane']}|{x['backend']}" == key and "setup_seconds" in x and x["earlier_cases_same_backend_and_mode"] == 0]
        b["setup_seconds_when_first_of_backend_and_mode_on_lane"] = first
    result = {"scope": "registered supplementary reader; primary statistic is audit_h20.py",
              "candidate": CANDIDATE, "lanes_read": lanes, "comparisons": comparisons, "backends": backends, "solves": solves}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps([{k: (round(c[k]["geometric_mean"], 3) if isinstance(c.get(k), dict) else c.get(k))
                       for k in ("lane", "baseline", "setup_plus_optimization", "optimization_only", "optimization_per_evaluation")}
                      for c in comparisons]))


if __name__ == "__main__":
    main()
