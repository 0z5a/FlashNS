"""Recompute the README comparisons from the published historical JSON."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    sources = {}

    def read(name):
        data = (ROOT / name).read_bytes()
        sources[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    solver = read("artifacts/hopper_followup/solver-summary.json")
    if solver["original_converged"] != 17 or solver["final_converged"] != 24:
        raise ValueError("unexpected solver convergence counts")
    backends = solver["backends"]
    flashns_seconds = backends["HopperTMA"]["median_total_wall_seconds"]
    solve_rows = {}
    for name, result in backends.items():
        if result["converged"] != 3 or len(result["runs"]) != 3:
            raise ValueError(f"incomplete solver backend: {name}")
        seconds = result["median_total_wall_seconds"]
        solve_rows[name] = {
            "median_accepted_run_seconds": seconds,
            "ratio_of_medians_over_hopper_tma": seconds / flashns_seconds,
            "all_attempts_process_seconds": result["all_attempts_process_wall_seconds_sum"],
        }

    science = read("experiments/cuda_jet_h100/artifacts/scientific_benchmark.json")
    if not science["completed"] or not science["all_requested_backends_validated"]:
        raise ValueError("scientific-library benchmark was not fully validated")
    case = next(row for row in science["complete_gradient"] if row["case"]["global_batch"] == 16384)
    gradients = {
        name: {
            "baseline_median_ms": case["medians"][name + "_replay"]["cuda_event_ms"],
            "flashns_f_median_ms": case["medians"]["F_replay"]["cuda_event_ms"],
            "median_paired_ratio": case["paired_speedup_medians"][name + "_over_F_replay"],
        }
        for name in ("Torch_jet_compiled", "cuEquivariance")
    }
    mathdx = read("experiments/cuda_jet_h100/artifacts_h100b/mathdx_benchmark.json")
    if not mathdx["passed"]:
        raise ValueError("MathDx benchmark was not validated")
    mathdx_case = next(row for row in mathdx["complete_gradient"] if row["case"]["global_batch"] == 65536)
    mathdx_row = {
        "baseline_median_ms": mathdx_case["medians"]["MathDx64_replay"]["cuda_event_ms"],
        "flashns_f_plus_median_ms": mathdx_case["medians"]["F+_replay"]["cuda_event_ms"],
        "median_paired_ratio": mathdx_case["paired_speedup_medians"]["MathDx64_over_F+_replay"],
    }
    return {
        "scope": "Historical FP64 observations; no GPU rerun. Solve ratios are ratios of three accepted-run medians after the documented budget extension.",
        "original_converged": solver["original_converged"],
        "final_converged": solver["final_converged"],
        "complete_solve_h100_nvl": solve_rows,
        "complete_gradient_h100_16384_points": gradients,
        "complete_gradient_separate_h100_mathdx_65536_points": mathdx_row,
        "source_sha256": sources,
    }


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, allow_nan=False))
