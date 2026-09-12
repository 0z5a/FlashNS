"""Summarize complete v8 observations using paired run/seed ratios."""

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio_summary(pairs, *, unit, bootstrap_samples=20000):
    if not pairs or not all(math.isfinite(x) and x > 0 for pair in pairs for x in pair):
        raise ValueError("finite positive paired times required")
    ratios = [float(baseline)/float(candidate) for baseline, candidate in pairs]
    result = {"paired_ratios": ratios, "paired_median_baseline_over_candidate": statistics.median(ratios),
              "pair_count": len(ratios), "resampling_unit": unit, "bootstrap_95_percent_interval": None}
    if len(ratios) >= 3:
        rng = random.Random(590219)
        draws = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(bootstrap_samples))

        def quantile(fraction):
            at = fraction*(len(draws)-1)
            lower = math.floor(at)
            return draws[lower] + (at-lower)*(draws[min(lower+1, len(draws)-1)]-draws[lower])

        result["bootstrap_95_percent_interval"] = [quantile(0.025), quantile(0.975)]
        result["bootstrap_samples"] = bootstrap_samples
    return result


def candidate_key(item):
    return "/".join(item[key] for key in ("layout", "seed_mode", "activation"))


def benchmark(path, baseline=None):
    source = json.loads(path.read_text())
    if not source.get("completed"):
        raise ValueError("benchmark is incomplete")
    baseline = baseline or source["reference_key"]
    keys = sorted(source["medians"])
    if baseline not in keys:
        raise ValueError("baseline is absent from observations")
    rounds = source["rounds"]
    if any(set(row["observations"]) != set(keys) for row in rounds):
        raise ValueError("unpaired or missing benchmark observation")
    result = {"kind": "resident_full_loss_parameter_gradient_graph", "baseline": baseline,
              "completed": True, "problem": source["problem"], "absolute_medians": source["medians"],
              "paired_comparisons": {}, "limits": "within one GPU run; blocks, not individual replays, are resampled; no solver accuracy or cross-day inference"}
    for key in keys:
        result["paired_comparisons"][key] = {
            metric: ratio_summary([(r["observations"][baseline][metric], r["observations"][key][metric]) for r in rounds],
                                  unit="paired randomized measurement block")
            for metric in ("cuda_event_ms", "synchronized_wall_ms")}
    return result


def suite(path, baseline=None):
    source = json.loads(path.read_text())
    if not source.get("completed") or len(source["runs"]) != len(source["schedule"]):
        raise ValueError("suite is incomplete")
    baseline = baseline or (candidate_key(source["baseline"]) if source.get("baseline") else None)
    keys = sorted({candidate_key(row) for row in source["schedule"]})
    if baseline not in keys:
        raise ValueError("select an observed baseline explicitly")
    protocol_path = path.parent / "protocol.json"
    if digest(protocol_path) != source["protocol_sha256"]:
        raise ValueError("protocol changed since the schedule was frozen")
    protocol = json.loads(protocol_path.read_text())
    failures, observations, points, initial, binaries = [], {}, {}, {}, set()
    for planned, row in zip(source["schedule"], source["runs"]):
        key, seed = candidate_key(row), row["seed"]
        if row["seed"] != planned["seed"] or candidate_key(planned) != key or (seed, key) in observations:
            raise ValueError("run does not match the frozen schedule, or is duplicated")
        observations[seed, key] = row
        filename = path.parent / row["output"]
        if row["exit_code"] or not row.get("completed") or not filename.exists():
            failures.append({"seed": seed, "candidate": key, "reason": "failed process or incomplete evidence"})
            continue
        child = json.loads(filename.read_text())
        if child["source_hashes"] != source["source_hashes"] or child["result"] != row["result"] or child["protocol"] != protocol:
            raise ValueError("child result, source, or objective differs from the frozen suite")
        saved = path.parent / child["checkpoint"]["file"]
        if not saved.is_file() or digest(saved) != child["checkpoint"]["sha256"]:
            raise ValueError("checkpoint is missing or changed")
        scientific = child["problem"]
        for name in ("training_points_sha256", "pde_weights_sha256", "boundary_weights_sha256", "validation_points_sha256", "validation_boundary_sha256"):
            if name in points and points[name] != scientific[name]:
                raise ValueError("point set or globally normalized objective differs")
            points[name] = scientific[name]
        start = child["result"]["parameter_initial_sha256"]
        if seed in initial and initial[seed] != start:
            raise ValueError("candidate initialization differs for a paired seed")
        initial[seed] = start
        binaries.add(child["backend_setup"]["native_build"]["library_sha256"])
        if not row["result"]["converged"]:
            failures.append({"seed": seed, "candidate": key, "reason": "stopping thresholds not met within the frozen budget"})
        else:
            for metric_set in (row["result"]["metrics"], row["result"]["independent_metrics"]):
                if not all(isinstance(metric_set[name], (int, float)) and math.isfinite(metric_set[name]) and metric_set[name] <= limit
                           for name, limit in protocol["thresholds"].items()):
                    raise ValueError("claimed convergence does not satisfy both acceptance checks")
    seeds = sorted({seed for seed, _ in observations})
    if any((seed, key) not in observations for seed in seeds for key in keys):
        raise ValueError("suite lacks a scheduled pair")
    if len(binaries) > 1:
        raise ValueError("native binary changed within the suite")
    result = {"kind": "fixed_accuracy_complete_solver", "phase": source["phase"], "baseline": baseline,
              "completed": True, "seeds": seeds, "all_converged": not failures, "failures": failures,
              "point_hashes": points, "binary_sha256": sorted(binaries), "candidates": {},
              "paired_comparisons": None, "all_attempts_process_wall_seconds": sum(r["process_wall_seconds"] for r in source["runs"]),
              "limits": "seed-level paired percentile bootstrap; small samples and one host/day limit generalization; no all-suite speedup if any scheduled run fails"}
    for key in keys:
        rows = [observations[seed, key] for seed in seeds]
        result["candidates"][key] = {"attempts": len(rows),
            "successes": sum(not any(f["seed"] == r["seed"] and f["candidate"] == key for f in failures) for r in rows),
            "all_attempts_process_wall_seconds": sum(r["process_wall_seconds"] for r in rows),
            "runs": [{"seed": r["seed"], "exit_code": r["exit_code"], "process_wall_seconds": r["process_wall_seconds"], "result": r.get("result")} for r in rows]}
    if not failures:
        result["paired_comparisons"] = {}
        for key in keys:
            result["paired_comparisons"][key] = {}
            for metric in ("total_wall_seconds", "optimization_wall_seconds", "process_wall_seconds"):
                def value(seed, name):
                    row = observations[seed, name]
                    return row[metric] if metric == "process_wall_seconds" else row["result"][metric]
                result["paired_comparisons"][key][metric] = ratio_summary([(value(seed, baseline), value(seed, key)) for seed in seeds], unit="paired initialization seed")
            result["candidates"][key]["medians"] = {
                name: statistics.median(observations[seed, key]["result"][name] for seed in seeds)
                for name in ("total_wall_seconds", "optimization_wall_seconds", "loss_gradient_evaluations", "adam_updates", "lbfgs_iterations", "peak_allocated_bytes")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark", type=Path)
    group.add_argument("--suite", type=Path)
    parser.add_argument("--baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite an analysis")
    source = args.suite or args.benchmark
    result = suite(source, args.baseline) if args.suite else benchmark(source, args.baseline)
    result.update(input_sha256=digest(source), analysis_source_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
