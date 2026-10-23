"""Post-hoc checks of the 2D ablation (not part of the registered analysis).

    python3 posthoc_ablation.py RESULTS_DIR REGISTERED_RESULT.json OUT.json

1. 2,048-point bimodality: per-process median, within-process sample spread and the SM clock that
   bench_gpu recorded before timing, after warmup and after timing.
2. Solves: optimization-only and per-evaluation ratios next to the registered setup-plus-optimization
   ratio, and the per-evaluation optimizer time left after subtracting the cell's replay median.
3. Oracle: largest gradient error against the dense Q10 oracle over all timed processes.
4. Profiles: kernel time in microseconds by category for the profiled processes.
Standard library only.
"""
import glob
import json
import math
import os
import statistics
import sys


def clock(snapshot):
    return int(snapshot.strip().splitlines()[-1].split(",")[3].split()[0])


def main():
    results, registered_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    registered = json.load(open(registered_path))
    reports = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(results, "*.json"))}
    bimodal = {}
    for name, r in sorted(reports.items()):
        if "-n2048-h64x64-r" in name and "DRIFT" not in name and "median_ms" in r:
            samples = [s["gpu_ms_per_gradient"] for s in r["samples"]]
            bimodal.setdefault(name.split("-")[0], []).append({
                "task": name, "median_ms": r["median_ms"], "within_process_spread": max(samples) / min(samples) - 1,
                "sm_mhz": [clock(r["device_before"]), clock(r["device_warmed"]), clock(r["device_after"])]})
    replay = {c: registered["cells"][f"{c}|2048|64x64"]["median_ms"] for c in "ABC"}
    solves = {}
    for name, r in reports.items():
        if "-solve-s" in name:
            s = r["solve"]
            solves[name] = {"cell": name[0], "seed": int(name.split("-s")[-1]), "converged": s["converged"],
                            "setup_seconds": s["setup_seconds"], "optimization_seconds": s["optimization_wall_seconds"],
                            "evaluations": s["loss_gradient_evaluations"],
                            "optimization_ms_per_evaluation": 1000 * s["optimization_wall_seconds"] / s["loss_gradient_evaluations"]}
            solves[name]["residual_ms_per_evaluation"] = solves[name]["optimization_ms_per_evaluation"] - replay[name[0]]
    seeds = sorted({v["seed"] for v in solves.values()})
    get = lambda c, s: next(v for v in solves.values() if v["cell"] == c and v["seed"] == s)
    pairs = {}
    for a, b in (("A", "C"), ("A", "B"), ("B", "C")):
        def gm(f):
            logs = [math.log(f(get(a, s)) / f(get(b, s))) for s in seeds]
            return {"geometric_mean": math.exp(sum(logs) / len(logs)), "range": [math.exp(min(logs)), math.exp(max(logs))]}
        pairs[f"{a}/{b}"] = {"setup_plus_optimization": gm(lambda v: v["setup_seconds"] + v["optimization_seconds"]),
                             "optimization_only": gm(lambda v: v["optimization_seconds"]),
                             "optimization_per_evaluation": gm(lambda v: v["optimization_ms_per_evaluation"])}
    oracle = [r["correctness_against_q10"]["gradient_max_abs_error"] for r in reports.values() if "correctness_against_q10" in r]
    profiles = {}
    for key, p in registered["profiles"].items():
        profiles[key] = {"kernels": p["kernels"], "device_us": p["device_us"],
                         "us": {k: v * p["device_us"] for k, v in p["share"].items()}, "inductor_gemm_template": p["inductor_gemm_template"]}
    result = {"scope": "post-hoc checks; not part of the registered analysis",
              "bimodality_2048": bimodal,
              "bimodality_summary": {"sm_mhz_values": sorted({m for rows in bimodal.values() for row in rows for m in row["sm_mhz"]}),
                                     "largest_within_process_spread": max(row["within_process_spread"] for rows in bimodal.values() for row in rows)},
              "solves": solves, "solve_pairs": pairs,
              "solve_summary": {c: {"median_setup_seconds": statistics.median(get(c, s)["setup_seconds"] for s in seeds),
                                    "optimization_ms_per_evaluation": [min(get(c, s)["optimization_ms_per_evaluation"] for s in seeds),
                                                                       max(get(c, s)["optimization_ms_per_evaluation"] for s in seeds)],
                                    "replay_ms": replay[c], "evaluations": sorted(get(c, s)["evaluations"] for s in seeds)} for c in "ABC"},
              "oracle": {"processes": len(oracle), "largest_gradient_abs_error": max(oracle)},
              "profiles_us": profiles}
    open(out, "w").write(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"sm_mhz": result["bimodality_summary"]["sm_mhz_values"], "oracle": result["oracle"],
                      "pairs": {k: {m: round(v[m]["geometric_mean"], 3) for m in v} for k, v in pairs.items()}}))


if __name__ == "__main__":
    main()
