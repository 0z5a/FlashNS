"""Reader and registered analysis of the 2D L-BFGS GH200 study.

    python read_lbfgs_study.py --root ROOT --plan plan-2d-lbfgs-gh200-20261023.json --output OUT.json

ROOT/lbfgs holds the plan, protocol, control/driver-state.json and results/<task>.json written by
light_solve.py. Per size: identity of the stock/light pair (final parameter SHA-256, evaluation and
L-BFGS iteration counts, acceptance), paired geometric means of optimization-only and setup plus
optimization seconds (stock over light) over jointly accepted pairs, pairs favouring light, phase
costs per evaluation, and the registered decision. Drift repeats: identity with the first run and
optimization time within 5%. Exit 0 whenever the analysis ran; the decision is in the output.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

IDENTITY_FIELDS = ("parameter_final_sha256", "loss_gradient_evaluations", "lbfgs_iterations", "adam_updates", "converged")


def phase_ms(history, phase):
    rows = [r for r in history if r["phase"] in (("initial", "adam") if phase == "adam" else ("lbfgs",))]
    if phase == "lbfgs":
        rows = [r for r in history if r["phase"] == "adam"][-1:] + rows
    if len(rows) < 2:
        return None
    de = rows[-1]["loss_gradient_evaluations"] - rows[0]["loss_gradient_evaluations"]
    return 1e3 * (rows[-1]["optimization_wall_seconds"] - rows[0]["optimization_wall_seconds"]) / de if de else None


def row_of(report):
    solve = report.get("solve") or {}
    return {"status": report.get("status"), "converged": solve.get("converged"),
            "identity": {k: solve.get(k) for k in IDENTITY_FIELDS},
            "setup_seconds": solve.get("setup_seconds"), "optimization_seconds": solve.get("optimization_wall_seconds"),
            "adam_ms_per_evaluation": phase_ms(solve.get("history", []), "adam") if solve else None,
            "lbfgs_ms_per_evaluation": phase_ms(solve.get("history", []), "lbfgs") if solve else None,
            "light": report.get("light"), "lbfgs_module_sha256": (report.get("wrapper") or {}).get("torch_optim_lbfgs_module_sha256")}


def geometric(values):
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else None


def decide_size(pairs, margin=1.03):
    """pairs: {seed: (stock_row, light_row)}; returns the per-size statistics and decision."""
    accepted = {s: p for s, p in pairs.items() if all(r["status"] == "passed" and r["converged"] for r in p)}
    identical = {s: p[0]["identity"] == p[1]["identity"] for s, p in pairs.items()}
    differing = sorted(s for s, same in identical.items() if not same)
    ratios = {s: p[0]["optimization_seconds"] / p[1]["optimization_seconds"] for s, p in accepted.items()}
    total = {s: (p[0]["setup_seconds"] + p[0]["optimization_seconds"]) / (p[1]["setup_seconds"] + p[1]["optimization_seconds"]) for s, p in accepted.items()}
    out = {"pairs": len(pairs), "jointly_accepted": len(accepted), "identical_pairs": sum(identical.values()), "differing_seeds": differing,
           "optimization_stock_over_light": {"geometric_mean": geometric(list(ratios.values())),
                                             "range": [min(ratios.values()), max(ratios.values())] if ratios else None, "per_seed": ratios},
           "setup_plus_optimization_stock_over_light": {"geometric_mean": geometric(list(total.values())), "per_seed": total},
           "pairs_favouring_light": sum(r > 1 for r in ratios.values()),
           "phase_ms_per_evaluation": {cell: {ph: statistics.median(v) if (v := [p[i][f"{ph}_ms_per_evaluation"] for p in accepted.values() if p[i][f"{ph}_ms_per_evaluation"] is not None]) else None
                                              for ph in ("adam", "lbfgs")} for i, cell in enumerate(("stock", "light"))},
           "setup_seconds": {cell: [p[i]["setup_seconds"] for p in accepted.values()] for i, cell in enumerate(("stock", "light"))}}
    if len(accepted) < 5:
        out["decision"] = "NOT_DECIDABLE"
    elif differing:
        out["decision"] = "TRAJECTORY_DIFFERS"
    elif len(accepted) == len(pairs) and out["optimization_stock_over_light"]["geometric_mean"] > margin and out["pairs_favouring_light"] >= 5:
        out["decision"] = "EXECUTION_GAIN_IDENTICAL"
    else:
        out["decision"] = "IDENTICAL_NO_GAIN"
    return out


def analyse(protocol, plan, rows):
    """rows: {task id: row_of(report)}. Pure function used by the tests."""
    sizes = {}
    for n in protocol["interior_counts"]:
        pairs = {}
        for seed in protocol["seeds"]:
            s, l = rows.get(f"stock-n{n}-s{seed}"), rows.get(f"light-n{n}-s{seed}")
            if s and l:
                pairs[seed] = (s, l)
        sizes[str(n)] = decide_size(pairs)
    drift = []
    for t in plan["tasks"]:
        if t["tier"] != 1:
            continue
        first, again = rows.get(t["repeat_of"]), rows.get(t["id"])
        ok = bool(first and again and first["status"] == again["status"] == "passed" and first["optimization_seconds"] and again["optimization_seconds"])
        same = ok and first["identity"] == again["identity"]
        ratio = again["optimization_seconds"] / first["optimization_seconds"] if ok else None
        drift.append({"id": t["id"], "repeat_of": t["repeat_of"], "identical": same, "optimization_ratio": ratio,
                      "within_5_percent": ratio is not None and abs(ratio - 1) <= 0.05})
    valid = all(d["identical"] and d["within_5_percent"] for d in drift) and len(drift) == 4
    return {"sizes": sizes, "drift": drift, "timing_valid": valid,
            "decisions": {n: s["decision"] + ("" if valid else " (NOT_VALID timing)") for n, s in sizes.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = args.root / "lbfgs"
    plan = json.loads((study / args.plan).read_text())
    protocol = json.loads((study / plan["protocol"]).read_text())
    state = json.loads((study / "control" / "driver-state.json").read_text())
    rows, tasks = {}, {}
    for t in plan["tasks"]:
        path = study / "results" / (t["id"] + ".json")
        if path.is_file():
            report = json.loads(path.read_text())
            rows[t["id"]] = row_of(report)
            tasks[t["id"]] = {"status": report.get("status"), "lane": state["tasks"].get(t["id"], {}).get("lane"),
                              "exit": state["tasks"].get(t["id"], {}).get("exit"), "cell": t["cell"], "interior": t["interior"], "seed": t["seed"]}
        else:
            tasks[t["id"]] = {"status": "missing", "cell": t["cell"], "interior": t["interior"], "seed": t["seed"]}
    result = {"protocol_version": protocol["version"], "plan_sha256": plan["protocol_sha256"], "driver_state": state["state"],
              "driver_checks": state.get("checks"), "tasks": tasks,
              "lbfgs_module_sha256_match": all(r["lbfgs_module_sha256"] == protocol["runtime"]["torch_optim_lbfgs_module_sha256"] for r in rows.values()),
              "rows": rows, **analyse(protocol, plan, rows)}
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"decisions": result["decisions"], "timing_valid": result["timing_valid"],
                      **{n: {"identical": s["identical_pairs"], "gm": s["optimization_stock_over_light"]["geometric_mean"], "favour": s["pairs_favouring_light"]}
                         for n, s in result["sizes"].items()}}, indent=1))


if __name__ == "__main__":
    main()
