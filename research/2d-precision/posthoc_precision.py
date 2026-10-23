"""Post-hoc checks of the 2D precision study (not part of the registered analysis).

    python3 posthoc_precision.py RESULTS_DIR OUT.json

1. TF32 solves: evaluations, Adam updates and L-BFGS iterations at the end, the final FP64
   validation metrics, and which gates fail.
2. FP32 and FP64 solves: final FP64 validation metrics, to compare the accuracy reached.
Standard library only; writes no absolute paths.
"""
import glob
import json
import os
import sys

THRESHOLDS = {"pde_rms": 0.02, "boundary_uv_rms": 0.01, "outflow_p_rms": 0.01,
              "relative_l2_u": 0.01, "relative_l2_v": 0.02, "relative_l2_p": 0.02}


def main():
    results, out = sys.argv[1], sys.argv[2]
    solves = {}
    for path in sorted(glob.glob(os.path.join(results, "solve-*.json"))):
        name = os.path.basename(path)[:-5]
        report = json.load(open(path))
        s = report.get("solve", {})
        metrics = s.get("metrics") or {}
        solves[name] = {"status": report.get("status"), "converged": s.get("converged"),
                        "evaluations": s.get("loss_gradient_evaluations"), "adam_updates": s.get("adam_updates"),
                        "lbfgs_iterations": s.get("lbfgs_iterations"), "final_metrics": metrics,
                        "failing_gates": sorted(k for k, v in metrics.items() if v > THRESHOLDS[k]),
                        "worst_gate_over_limit": max((v / THRESHOLDS[k] for k, v in metrics.items()), default=None)}
    summary = {}
    for name, row in solves.items():
        key = "|".join(name.split("-")[1:4])
        group = summary.setdefault(key, {"solves": 0, "evaluations": [], "lbfgs_iterations": [], "worst_gate_over_limit": [], "failing_gates": {}})
        group["solves"] += 1
        group["evaluations"].append(row["evaluations"])
        group["lbfgs_iterations"].append(row["lbfgs_iterations"])
        group["worst_gate_over_limit"].append(row["worst_gate_over_limit"])
        for gate in row["failing_gates"]:
            group["failing_gates"][gate] = group["failing_gates"].get(gate, 0) + 1
    for group in summary.values():
        for k in ("evaluations", "lbfgs_iterations", "worst_gate_over_limit"):
            values = [v for v in group[k] if v is not None]
            group[k] = [min(values), max(values)] if values else None
    json.dump({"scope": "post hoc; not part of the registered analysis", "summary": summary, "solves": solves}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
