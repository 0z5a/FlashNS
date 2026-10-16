"""Post-hoc checks of the 2D H100 solve-scale study (not part of the registered analysis).

    python3 posthoc_solve_scale.py SOLVE_SCALE_RESULT.json ABLATION_RESULT.json PHASE_SPLIT_2048.json DRIVER_STATE.json \
        ABLATION_DRIVER_STATE.json ABLATION_PLAN.json OUT.json

1. Phase cost against gradient replay: per cell and interior count, the Adam and L-BFGS time per
   evaluation minus the ablation's registered replay median for the same cell and count (different
   processes and node, so approximate). The 2,048-point rows come from phase_split_2048.py.
2. Setup and compiler caches: each lane keeps its own Inductor cache. A solve is "first on lane"
   when no earlier solve of the same cell and interior count started on that lane; setup ranges are
   reported for first and later solves, with start times relative to the first task.
3. The same check for the ablation's 2,048-point development solves: for each solve, the number of
   earlier tasks of the same cell at 2,048 points and the original 64x64 topology on its lane.
Standard library only.
"""
import json
import sys


def span(values):
    return [min(values), max(values)] if values else None


def main():
    result_path, ablation_path, split_path, state_path, ablation_state_path, ablation_plan_path, out = sys.argv[1:8]
    result = json.load(open(result_path))
    ablation = json.load(open(ablation_path))
    split = json.load(open(split_path))
    state = json.load(open(state_path))["tasks"]
    rows = [r for r in result["rows"] if r.get("status") == "passed"]
    t0 = min(state[r["id"]]["started_unix"] for r in rows)
    overhead = {"2048": {c: {"adam_minus_replay_ms": span([s["adam_ms_per_evaluation_minus_replay"] for s in split["solves"].values() if s["cell"] == c]),
                             "lbfgs_minus_replay_ms": span([s["lbfgs_ms_per_evaluation_minus_replay"] for s in split["solves"].values() if s["cell"] == c]),
                             "replay_ms": split["summary"][c]["replay_ms"]} for c in "ABC"}}
    for n in sorted({r["interior"] for r in rows}):
        overhead[str(n)] = {}
        for c in "ABC":
            replay = ablation["cells"][f"{c}|{n}|64x64"]["median_ms"]
            cell = [r for r in rows if r["cell"] == c and r["interior"] == n]
            overhead[str(n)][c] = {"replay_ms": replay,
                                   "adam_minus_replay_ms": span([r["adam_ms_per_evaluation"] - replay for r in cell]),
                                   "lbfgs_minus_replay_ms": span([r["lbfgs_ms_per_evaluation"] - replay for r in cell])}
    ordered = sorted(rows, key=lambda r: state[r["id"]]["started_unix"])
    seen, setup = set(), []
    for r in ordered:
        key = (r["lane"], r["cell"], r["interior"])
        setup.append({"task": r["id"], "lane": r["lane"], "start_seconds": round(state[r["id"]]["started_unix"] - t0, 1),
                      "first_on_lane": key not in seen, "setup_seconds": r["setup_seconds"]})
        seen.add(key)
    setup_summary = {}
    for n in sorted({r["interior"] for r in rows}):
        for c in "ABC":
            group = [s for s in setup if s["task"].startswith(f"{c}-n{n}-")]
            setup_summary[f"{c}|{n}"] = {"first_on_lane": span([s["setup_seconds"] for s in group if s["first_on_lane"]]),
                                         "later": span([s["setup_seconds"] for s in group if not s["first_on_lane"]]),
                                         "started_at_batch_start": [s["setup_seconds"] for s in group if s["start_seconds"] < 1]}
    a_state = json.load(open(ablation_state_path))["tasks"]
    a_plan = {t["id"]: t for t in json.load(open(ablation_plan_path))["tasks"]}
    a_order = sorted((v["started_unix"], k, v.get("lane")) for k, v in a_state.items() if "started_unix" in v)
    a_rows = {r["id"]: r for r in ablation["rows"]}
    ablation_solves = []
    for started, k, lane in a_order:
        if a_plan[k]["kind"] != "solve":
            continue
        cell = a_plan[k]["cell"]
        prior = sum(1 for t, kk, ll in a_order if t < started and ll == lane and a_plan[kk]["cell"] == cell
                    and a_plan[kk]["interior"] == 2048 and a_plan[kk]["hidden"] == [64, 64] and a_plan[kk]["label"][:1] == cell)
        ablation_solves.append({"task": k, "lane": lane, "setup_seconds": a_rows[k]["setup_seconds"], "prior_same_cell_tasks_on_lane": prior})
    report = {"scope": "post hoc; not part of the registered analysis",
              "overhead_note": "replay medians are the ablation's registered cell medians (other processes and node); differences are approximate",
              "phase_minus_replay": overhead,
              "lbfgs_minus_replay_all_cells_ms": span([v for n in overhead.values() for c in n.values() for v in c["lbfgs_minus_replay_ms"]]),
              "adam_minus_replay_all_cells_ms": span([v for n in overhead.values() for c in n.values() for v in c["adam_minus_replay_ms"]]),
              "setup_summary": setup_summary, "setup_by_start": setup,
              "ablation_2048_solve_setup": {"solves": ablation_solves,
                                            "after_same_cell_on_lane": span([s["setup_seconds"] for s in ablation_solves if s["prior_same_cell_tasks_on_lane"]]),
                                            "first_of_cell_on_lane": [s["setup_seconds"] for s in ablation_solves if not s["prior_same_cell_tasks_on_lane"]]}}
    open(out, "w").write(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"lbfgs_minus_replay": [round(x, 2) for x in report["lbfgs_minus_replay_all_cells_ms"]],
                      "adam_minus_replay": [round(x, 3) for x in report["adam_minus_replay_all_cells_ms"]],
                      "setup": {k: v for k, v in setup_summary.items()},
                      "ablation_first_of_cell_on_lane": [(s["task"], s["setup_seconds"]) for s in ablation_solves if not s["prior_same_cell_tasks_on_lane"]]}))


if __name__ == "__main__":
    main()
