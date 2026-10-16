"""Post-hoc Adam/L-BFGS phase split of the 2,048-point ablation solves (not a registered analysis).

    python3 phase_split_2048.py ABLATION_RESULTS_DIR ABLATION_RESULT.json OUT.json

Applies read_solve_scale.phases -- the split this study's reader reports -- to the 18 archived
development solves of the 2D ablation (cells A, B, C, six formal seeds, 2,048 points). It also
reports, per phase, the time per evaluation left after subtracting the cell's registered replay median.
Standard library only.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_solve_scale import phases  # noqa: E402


def main():
    results, registered_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    registered = json.load(open(registered_path))
    replay = {c: registered["cells"][f"{c}|2048|64x64"]["median_ms"] for c in "ABC"}
    solves = {}
    for path in sorted(glob.glob(os.path.join(results, "*-solve-s*.json"))):
        name = os.path.basename(path)[:-5]
        report = json.load(open(path))
        s = report["solve"]
        row = {"cell": name[0], "seed": int(name.split("-s")[-1]), "converged": s["converged"],
               "evaluations": s["loss_gradient_evaluations"], "adam_updates": s["adam_updates"],
               "lbfgs_iterations": s["lbfgs_iterations"], "progress_rows": len(report["progress"]),
               **phases(report["progress"])}
        for phase in ("adam", "lbfgs"):
            if f"{phase}_ms_per_evaluation" in row:
                row[f"{phase}_ms_per_evaluation_minus_replay"] = row[f"{phase}_ms_per_evaluation"] - replay[name[0]]
        solves[name] = row
    summary = {}
    for c in "ABC":
        rows = [r for r in solves.values() if r["cell"] == c]
        summary[c] = {"solves": len(rows), "replay_ms": replay[c]}
        for key in ("adam_ms_per_evaluation", "lbfgs_ms_per_evaluation", "lbfgs_evaluations"):
            values = [r[key] for r in rows if key in r]
            summary[c][key] = [min(values), max(values)] if values else None
    result = {"scope": "post hoc; descriptive split of archived 2,048-point ablation solves, not part of either registered analysis",
              "method": "read_solve_scale.phases: Adam = (last Adam progress row - first row) optimization wall time over its evaluation count; "
                        "L-BFGS = (last row - last Adam row) over the evaluations between them",
              "summary": summary, "solves": solves}
    open(out, "w").write(json.dumps(result, indent=1) + "\n")
    print(json.dumps({c: {k: v if not isinstance(v, list) else [round(x, 3) for x in v] for k, v in s.items()} for c, s in summary.items()}))


if __name__ == "__main__":
    main()
