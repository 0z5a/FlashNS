"""Registered decision of the C36 v6 cutoff-width study (pre-registered with its protocol).

    python -m box376.decide_c36_v6 --reader READER.json --protocol PROTOCOL --output OUT.json

Reads the unchanged in-batch reader (box376.read_c36_v2) and applies
protocol['decision_rule'] within this instance only:
- F1_pass: at least one run passes physical acceptance at an observed checkpoint;
- F2_near_miss: no run passes, and some run's final observation exceeds no acceptance limit
  by more than the registered factor (worst value over limit at most near_miss_factor);
- F0_incomplete: no pass and no near miss, and some registered run lacks a complete final
  observation (failed or killed training, observation not run or failed): no scientific reading;
- F3_fail: otherwise, that is no pass and no near miss with all four complete finals.
The worst over-limit factor covers every gated field metric of the gated datasets (the
observer's own limits, box376.validation.LIMITS), the initial and boundary velocity, and the
finer trajectory refinement's endpoint error and refinement change. Standard library only.
"""
import argparse
import hashlib
import json
from pathlib import Path

FIELD_LIMITS = {"velocity_relative_l2": 1e-2, "normalized_divergence_rms": 1e-3,
                "normalized_momentum_rms": 1e-2, "normalized_momentum_gradient_rms": 2e-2,
                "normalized_divergence_gradient_rms": 2e-2,
                "normalized_corrected_pressure_rms": 1e-3, "normalized_pressure_gauge_rms": 1e-3}


def over_limit(row, protocol):
    """Worst value/limit over every acceptance test of one complete observation, with the test name."""
    spec = protocol["physical"]
    tests = {}
    for name in spec["gated_datasets"]:
        metrics = row["datasets"][name]["global"]
        for metric, limit in FIELD_LIMITS.items():
            value = metrics.get(metric)
            if metric == "velocity_relative_l2" and metrics.get("velocity_reference_is_zero") is True:
                value, limit = metrics.get("normalized_velocity_error_rms"), 1e-3
            assert value is not None, (name, metric)
            tests[f"{name}:{metric}"] = value / limit
    for name, value in zip(("initial", "boundary"), row["initial_boundary_normalized_velocity_rms"]):
        tests[f"{name}:normalized_velocity_rms"] = value / spec["initial_boundary_velocity_normalized_rms_limit"]
    trajectory = row["trajectory"]
    tests["trajectory:maximum_endpoint_error"] = trajectory["maximum_endpoint_error"] / spec["maximum_trajectory_endpoint_error"]
    tests["trajectory:maximum_refinement_change"] = trajectory["maximum_refinement_change"] / spec["maximum_trajectory_refinement_change"]
    worst = max(tests, key=tests.get)
    return tests[worst], worst, {k: v for k, v in tests.items() if v > 1}


def decide(protocol, reader):
    rule = protocol["decision_rule"]
    runs = {}
    for run in reader["runs"]:
        key = f"{run['engine']}:{run['seed']}"
        observed = {tag: row for tag, row in run["physical"].items() if row.get("state") == "COMPLETE"}
        entry = {"engine": run["engine"], "seed": run["seed"], "training_state": run["training_state"],
                 "training_gate_reached": run["training_gate_first"] is not None,
                 "observed_tags": sorted(observed), "passing_tags": sorted(t for t, r in observed.items() if r["physical_acceptance_pass"])}
        if "final" in observed:
            factor, worst, failing = over_limit(observed["final"], protocol)
            entry.update(final_worst_over_limit=factor, final_worst_test=worst, final_failing_tests=failing,
                         final_stratified_pressure_over_limit=observed["final"]["datasets"]["stratified"]["global"]["normalized_corrected_pressure_rms"] / 1e-3)
        runs[key] = entry
    passing = [k for k, r in runs.items() if r["passing_tags"]]
    finals = [r["final_worst_over_limit"] for r in runs.values() if "final_worst_over_limit" in r]
    missing = sorted(k for k in rule["runs"] if k not in runs or "final_worst_over_limit" not in runs[k])
    if passing:
        outcome = {"key": "F1_pass", "runs": passing}
    elif finals and min(finals) <= rule["near_miss_factor"]:
        outcome = {"key": "F2_near_miss", "best_final_worst_over_limit": min(finals)}
    elif missing:
        outcome = {"key": "F0_incomplete", "runs": missing}
    else:
        outcome = {"key": "F3_fail", "best_final_worst_over_limit": min(finals)}
    outcome["text"] = rule["outcomes_in_order"][outcome["key"]]
    return {"outcome": outcome, "runs": runs, "runs_without_complete_final_observation": missing,
            "follow_up": rule["follow_up"][outcome["key"]], "formal_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    reader = json.loads(args.reader.read_text())
    assert reader["protocol_sha256"] == hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    result = {"protocol_sha256": reader["protocol_sha256"],
              "reader_sha256": hashlib.sha256(args.reader.read_bytes()).hexdigest(), **decide(protocol, reader)}
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps(result["outcome"]))


if __name__ == "__main__":
    main()
