"""Reader and registered analysis of the 2D H100 precision study (FP64, FP32, FP32+TF32).

    python3 read_precision.py --plan PLAN --protocol PROTOCOL --results DIR --state DRIVER_STATE --output OUT

Rechecks every result, then reports replay accuracy against the FP64 oracle, replay speed ratios
with percentile-bootstrap intervals, and for complete solves the FP64-accepted counts, paired
FP64/reduced and C/A ratios, the Adam/L-BFGS phase split and the registered decisions.
Requires only numpy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

REDUCED = ("fp32", "tf32")


def phases(progress):
    rows = [p for p in progress if "optimization_wall_seconds" in p and "loss_gradient_evaluations" in p]
    out = {}
    adam = [p for p in rows if p.get("phase") == "adam"]
    if rows and adam:
        a, initial, last = adam[-1], rows[0], rows[-1]
        out["adam_ms_per_evaluation"] = 1000 * (a["optimization_wall_seconds"] - initial["optimization_wall_seconds"]) / max(1, a["loss_gradient_evaluations"])
        evaluations = last["loss_gradient_evaluations"] - a["loss_gradient_evaluations"]
        if evaluations > 0:
            out["lbfgs_ms_per_evaluation"] = 1000 * (last["optimization_wall_seconds"] - a["optimization_wall_seconds"]) / evaluations
    return out


def ratio_ci(base, cand, rng, resamples):
    base, cand = np.asarray(base), np.asarray(cand)
    point = float(np.median(base) / np.median(cand))
    if len(base) < 2 or len(cand) < 2:
        return {"ratio": point, "ci95": None, "effect": None}
    draws = np.median(rng.choice(base, (resamples, len(base))), 1) / np.median(rng.choice(cand, (resamples, len(cand))), 1)
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return {"ratio": point, "ci95": ci, "effect": not (ci[0] >= 0.97 and ci[1] <= 1.03) and (ci[0] > 1.03 or ci[1] < 0.97)}


def paired(first, second, metric):
    """first/second: {seed: row}; ratio first/second over jointly accepted seeds."""
    seeds = sorted(s for s in first if s in second and first[s]["accepted"] and second[s]["accepted"])
    values = [metric(first[s]) / metric(second[s]) for s in seeds]
    if not values:
        return {"pairs": 0}
    logs = [math.log(v) for v in values]
    return {"pairs": len(values), "geometric_mean": math.exp(sum(logs) / len(logs)), "range": [min(values), max(values)],
            "favour_second": sum(v > 1 for v in values), "favour_first": sum(v < 1 for v in values)}


def speed_decision(opt):
    if opt["pairs"] < 5:
        return "not decidable"
    if opt["geometric_mean"] > 1.03 and opt["favour_second"] >= 5:
        return "second faster"
    if opt["geometric_mean"] < 1 / 1.03 and opt["favour_first"] >= 5:
        return "second slower"
    return "no gain"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "protocol", "results", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    protocol = json.loads(args.protocol.read_text())
    state = json.loads(args.state.read_text())
    rows, failures = [], []
    for t in plan["tasks"]:
        record = state["tasks"].get(t["id"], {})
        row = {k: t[k] for k in ("id", "kind", "cell", "precision", "interior", "seed", "tier")}
        row.update(lane=record.get("lane"), driver_state=record.get("state"), process_seconds=record.get("seconds"))
        path = args.results / f"{t['id']}.json"
        if path.exists():
            report = json.loads(path.read_text())
            row["status"] = report.get("status")
            row["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if not args.self_test and report.get("status") == "passed":
                assert report["precision"]["name"] == t["precision"], t["id"]
                assert report["precision"]["cuda_matmul_allow_tf32"] == (t["precision"] == "tf32"), t["id"]
                if t["kind"] == "replay":
                    assert report["dynamo_before"] == report["dynamo_after"], t["id"]
                    row["median_ms"] = report["median_ms"]
                    if t["precision"] == "fp64":
                        assert report["correctness_against_q10"]["passed"], t["id"]
                        row["gradient_max_abs_error"] = report["correctness_against_q10"]["gradient_max_abs_error"]
                    else:
                        c = report["correctness_against_fp64_q10"]
                        assert c["passed"], t["id"]
                        row.update({k: c[k] for k in ("loss_relative_error", "gradient_relative_l2", "gradient_max_over_max_abs", "gradient_max_abs_error")})
                else:
                    s = report["solve"]
                    row.update(converged=s["converged"], setup_seconds=s["setup_seconds"], optimization_seconds=s["optimization_wall_seconds"],
                               evaluations=s["loss_gradient_evaluations"], independent_metrics=s.get("independent_metrics"),
                               **phases(report.get("progress", [])))
            elif report.get("status") != "passed":
                failures.append({"id": t["id"], "error_tail": str(report.get("error"))[-600:]})
        if t["kind"] == "solve":
            row["accepted"] = bool(row.get("converged"))
        rows.append(row)
    result = {"protocol_version": protocol["version"], "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              "tasks": len(plan["tasks"]), "driver_counts": state.get("counts"), "prebuild_exit": state.get("checks", {}).get("prebuild_exit"),
              "passed": sum(r.get("status") == "passed" for r in rows), "failures": failures, "rows": rows}
    if args.self_test:
        result["self_test"] = True
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: result[k] for k in ("tasks", "passed")}))
        return
    rng = np.random.default_rng(20261016)
    resamples = 10000
    replay = [r for r in rows if r["kind"] == "replay" and "median_ms" in r]
    cells, accuracy, speed = {}, {}, {}
    for r in replay:
        cells.setdefault((r["cell"], r["precision"], r["interior"]), []).append(r)
    summary = {f"{c}|{p}|{n}": {"processes": len(v), "median_ms": float(np.median([x["median_ms"] for x in v])),
                                 "min_ms": min(x["median_ms"] for x in v), "max_ms": max(x["median_ms"] for x in v)}
               for (c, p, n), v in sorted(cells.items())}
    for (c, p, n), v in sorted(cells.items()):
        key = f"{c}|{p}|{n}"
        if p == "fp64":
            accuracy[key] = {"largest_gradient_max_abs_error": max(x["gradient_max_abs_error"] for x in v)}
        else:
            accuracy[key] = {k: max(x[k] for x in v) for k in ("gradient_relative_l2", "loss_relative_error", "gradient_max_over_max_abs")}
    for c in protocol["cells"]:
        for n in protocol["replay"]["interior_counts"]:
            base = cells.get((c, "fp64", n))
            for p in REDUCED:
                other = cells.get((c, p, n))
                if base and other:
                    speed[f"{c}|fp64/{p}|{n}"] = ratio_ci([x["median_ms"] for x in base], [x["median_ms"] for x in other], rng, resamples)
        for p in protocol["precisions"]:
            for n in protocol["replay"]["interior_counts"]:
                a, cc = cells.get(("A", p, n)), cells.get(("C", p, n))
                if a and cc:
                    speed[f"A/C|{p}|{n}"] = ratio_ci([x["median_ms"] for x in a], [x["median_ms"] for x in cc], rng, resamples)
    solves = [r for r in rows if r["kind"] == "solve"]
    by = {}
    for r in solves:
        if "optimization_seconds" in r or r.get("status") is not None:
            by.setdefault((r["cell"], r["precision"], r["interior"]), {})[r["seed"]] = r
    solve_cells, comparisons = {}, {}
    for (c, p, n), group in sorted(by.items()):
        done = [g for g in group.values() if "optimization_seconds" in g]
        solve_cells[f"{c}|{p}|{n}"] = {
            "runs": len(group), "accepted": sum(g["accepted"] for g in group.values()),
            "evaluations": sorted(g["evaluations"] for g in done),
            "optimization_seconds": sorted(round(g["optimization_seconds"], 3) for g in done),
            "adam_ms_per_evaluation": sorted(round(g["adam_ms_per_evaluation"], 4) for g in done if "adam_ms_per_evaluation" in g),
            "lbfgs_ms_per_evaluation": sorted(round(g["lbfgs_ms_per_evaluation"], 4) for g in done if "lbfgs_ms_per_evaluation" in g)}
    opt = lambda g: g["optimization_seconds"]
    per_eval = lambda g: g["optimization_seconds"] / g["evaluations"]
    total = lambda g: g["setup_seconds"] + g["optimization_seconds"]
    decisions = {}
    for n in protocol["solves"]["interior_counts"]:
        for c in protocol["cells"]:
            base = by.get((c, "fp64", n), {})
            for p in REDUCED:
                other = by.get((c, p, n), {})
                key = f"{c}|fp64/{p}|{n}"
                o = paired(base, other, opt)
                comparisons[key] = {"optimization_only": o, "per_evaluation": paired(base, other, per_eval),
                                    "setup_plus_optimization": paired(base, other, total)}
                accepted = sum(g["accepted"] for g in other.values())
                ran = sum(g.get("status") is not None for g in other.values())
                sufficiency = ("not decidable" if ran < 6 else "sufficient" if accepted == 6
                               else "insufficient" if accepted <= 4 else "unresolved")
                decisions[key] = {"training_sufficiency": sufficiency, "accepted": accepted, "ran": ran, "solve_speed": speed_decision(o)}
        for p in protocol["precisions"]:
            a, cc = by.get(("A", p, n), {}), by.get(("C", p, n), {})
            key = f"A/C|{p}|{n}"
            o = paired(a, cc, opt)
            comparisons[key] = {"optimization_only": o, "per_evaluation": paired(a, cc, per_eval), "setup_plus_optimization": paired(a, cc, total)}
            decisions[key] = {"solve_speed": speed_decision(o)}
    result.update(replay_cells=summary, replay_accuracy=accuracy, replay_speed=speed, solve_cells=solve_cells,
                  solve_comparisons=comparisons, decisions=decisions, bootstrap={"resamples": resamples, "seed": 20261016})
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "tasks": result["tasks"], "failures": len(failures),
                      "decisions": decisions}))


if __name__ == "__main__":
    main()
