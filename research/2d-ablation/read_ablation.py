"""Reader and registered analysis of the 2D representation/execution ablation.

    python3 read_ablation.py --plan PLAN --protocol PROTOCOL --results DIR --state DRIVER_STATE --output OUT

Rechecks every result (status, Q10 oracle, compile counters, topology hash), forms cell medians
(median over processes of per-process median ms), applies the A/A and drift gates, and reports the
registered effects with percentile-bootstrap intervals, the scaling tables, the kernel-time
breakdown of the profiled processes and the development solves. Requires only numpy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EFFECTS = {"representation_compiled": ("A", "B"), "representation_eager": ("D", "E"),
           "execution_GL8": ("B", "C"), "total": ("A", "C"), "explicit_transpose": ("F", "B")}


def category(name):
    n = name.lower()
    if any(k in n for k in ("gemm", "cutlass", "xmma", "cublas", "sm90_", "sm80_", "dgemm", "gemv")):
        return "GEMM (cuBLAS/CUTLASS)"
    if any(k in n for k in ("packed_fwd", "packed_rev", "packed_seed", "_fwd(", "_rev(", "_seed(")) or n.startswith("r") and ("_fwd" in n or "_rev" in n):
        return "native islands"
    if "triton" in n:
        return "Inductor-generated"
    if "reduce" in n or "reduction" in n:
        return "reduction"
    if "memcpy" in n or "memset" in n or "copy" in n:
        return "copy"
    return "other"


def ratio_ci(base, cand, rng, resamples):
    base, cand = np.asarray(base), np.asarray(cand)
    point = float(np.median(base) / np.median(cand))
    if len(base) < 2 or len(cand) < 2:
        return point, None
    b = rng.choice(base, (resamples, len(base)))
    c = rng.choice(cand, (resamples, len(cand)))
    draws = np.median(b, axis=1) / np.median(c, axis=1)
    return point, [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def classify(point, threshold):
    if abs(point - 1) < threshold:
        return "no effect"
    return "favours second cell" if point > 1 else "favours first cell"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "protocol", "results", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    protocol = json.loads(args.protocol.read_text())
    state = json.loads(args.state.read_text())
    threshold = 0.03
    rng = np.random.default_rng(20261016)
    resamples = 10000
    rows, failures = [], []
    for t in plan["tasks"]:
        record = state["tasks"].get(t["id"], {})
        path = args.results / f"{t['id']}.json"
        row = {"id": t["id"], "kind": t["kind"], "cell": t["cell"], "label": t["label"], "interior": t["interior"],
               "hidden": t["hidden"], "seed": t["seed"], "tier": t["tier"], "lane": record.get("lane"),
               "driver_state": record.get("state"), "process_seconds": record.get("seconds")}
        if path.exists():
            report = json.loads(path.read_text())
            row["status"] = report.get("status")
            row["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if args.self_test:
                rows.append(row)
                continue
            if report.get("status") == "passed":
                net = report.get("network", {})
                assert net.get("widths") == [2, *t["hidden"], 3], (t["id"], net.get("widths"))
                row["initial_parameters_sha256"] = net.get("initial_parameters_sha256")
                if t["kind"] == "replay":
                    assert report["correctness_against_q10"]["passed"], t["id"]
                    assert report["dynamo_before"] == report["dynamo_after"], t["id"]
                    row.update(median_ms=report["median_ms"], oracle_gradient_max_abs_error=report["correctness_against_q10"]["gradient_max_abs_error"],
                               setup_seconds=report["engine"].get("setup_seconds"), capture_seconds=report["engine"].get("capture_seconds"),
                               peak_allocated_bytes=report.get("engine_setup_peak_allocated_bytes"),
                               ready_reserved_bytes=report.get("engine_ready_reserved_bytes"))
                    if report.get("cuda_kernel_events"):
                        cats = {}
                        for e in report["cuda_kernel_events"]:
                            cats[category(e["name"])] = cats.get(category(e["name"]), 0.0) + e["device_time_us"]
                        total = sum(cats.values())
                        row["profile"] = {"kernels": len(report["cuda_kernel_events"]), "device_us": total,
                                          "share": {k: v / total for k, v in sorted(cats.items())} if total else {},
                                          "inductor_gemm_template": any("triton" in e["name"].lower() and ("mm" in e["name"].lower() or "tem" in e["name"].lower())
                                                                        for e in report["cuda_kernel_events"])}
                else:
                    s = report["solve"]
                    row.update(converged=s.get("converged"), setup_seconds=s.get("setup_seconds"),
                               optimization_seconds=s.get("optimization_wall_seconds"),
                               setup_plus_optimization_seconds=s.get("setup_plus_optimization_seconds"),
                               evaluations=s.get("loss_gradient_evaluations"))
            else:
                failures.append({"id": t["id"], "error_tail": str(report.get("error"))[-600:]})
        rows.append(row)
    result = {"protocol_version": protocol["version"], "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              "tasks": len(plan["tasks"]), "driver_counts": state.get("counts"),
              "passed": sum(r.get("status") == "passed" for r in rows), "failures": failures, "rows": rows}
    if args.self_test:
        result["self_test"] = True
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: result[k] for k in ("tasks", "passed")}))
        return
    timed = [r for r in rows if r["kind"] == "replay" and "median_ms" in r]
    cells = {}
    for r in timed:
        cells.setdefault((r["label"], r["interior"], tuple(r["hidden"])), []).append(r["median_ms"])
    summary = {f"{k[0]}|{k[1]}|{'x'.join(map(str, k[2]))}": {"processes": len(v), "median_ms": float(np.median(v)),
                                                               "min_ms": float(min(v)), "max_ms": float(max(v))}
               for k, v in sorted(cells.items())}
    base = tuple(protocol["original_hidden"])
    gates = {"AA": {}, "drift": []}
    for n in protocol["core_points"]:
        a, aa = cells.get(("A", n, base)), cells.get(("A_AA", n, base))
        if a and aa:
            r = float(np.median(aa) / np.median(a))
            gates["AA"][str(n)] = {"ratio": r, "pass": abs(r - 1) < 0.03}
    by_id = {r["id"]: r for r in rows}
    for t in plan["tasks"]:
        if "drift_of" in t and "median_ms" in by_id.get(t["id"], {}) and "median_ms" in by_id.get(t["drift_of"], {}):
            r = by_id[t["id"]]["median_ms"] / by_id[t["drift_of"]]["median_ms"]
            gates["drift"].append({"task": t["drift_of"], "ratio": r, "pass": abs(r - 1) < 0.05})
    gates["all_pass"] = bool(gates["AA"]) and all(g["pass"] for g in gates["AA"].values()) and all(g["pass"] for g in gates["drift"])
    effects = {}
    for n in sorted({k[1] for k in cells if k[2] == base}):
        for name, (first, second) in EFFECTS.items():
            x, y = cells.get((first, n, base)), cells.get((second, n, base))
            if x and y:
                point, ci = ratio_ci(x, y, rng, resamples)
                effects.setdefault(name, {})[str(n)] = {"ratio": point, "ci95": ci, "reading": classify(point, threshold)}
    shapes = {}
    for hidden in [tuple(h) for h in protocol["shape_hidden"]] + [base]:
        for n in protocol["shape_points"]:
            for name in ("representation_compiled", "execution_GL8", "total"):
                first, second = EFFECTS[name]
                x, y = cells.get((first, n, hidden)), cells.get((second, n, hidden))
                if x and y:
                    point, ci = ratio_ci(x, y, rng, resamples)
                    shapes.setdefault("x".join(map(str, hidden)), {}).setdefault(name, {})[str(n)] = {
                        "ratio": point, "ci95": ci, "reading": classify(point, threshold)}
    profiles = {f"{r['label']}|{r['interior']}": r["profile"] for r in timed if "profile" in r}
    solves = [r for r in rows if r["kind"] == "solve" and "converged" in r]
    paired = {}
    for first, second in (("A", "C"), ("A", "B"), ("B", "C")):
        logs = []
        for s in protocol["solve_seeds"]:
            x = next((r for r in solves if r["cell"] == first and r["seed"] == s and r["converged"]), None)
            y = next((r for r in solves if r["cell"] == second and r["seed"] == s and r["converged"]), None)
            if x and y:
                logs.append(math.log(x["setup_plus_optimization_seconds"] / y["setup_plus_optimization_seconds"]))
        if logs:
            paired[f"{first}/{second}"] = {"pairs": len(logs), "geometric_mean": math.exp(sum(logs) / len(logs)),
                                           "range": [math.exp(min(logs)), math.exp(max(logs))]}
    lanes = {}
    for r in timed:
        lanes.setdefault(f"{r['label']}|{r['interior']}|{'x'.join(map(str, r['hidden']))}", {}).setdefault(str(r["lane"]), []).append(r["median_ms"])
    result.update(cells=summary, gates=gates, effects=effects, shapes=shapes, profiles=profiles,
                  solves={"rows": solves, "converged": sum(bool(r["converged"]) for r in solves), "paired": paired},
                  lanes=lanes, effect_threshold=threshold, bootstrap={"resamples": resamples, "seed": 20261016})
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "tasks": result["tasks"], "gates_all_pass": gates["all_pass"],
                      "failures": len(failures)}))


if __name__ == "__main__":
    main()
