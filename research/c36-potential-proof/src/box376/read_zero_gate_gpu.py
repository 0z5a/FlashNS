"""Reader and registered classification of the C36 zero-gate folding GPU batch.

    python3 read_zero_gate_gpu.py --protocol PROTOCOL --science DIR --state DRIVER_STATE --output OUT

Applies the registered gates to each run (runner COMPLETE; NumPy audit PASS with 78 comparisons,
20 windows and errors within the original budget; folding reader PASS for the folded run),
classifies the folded ratio by its bootstrap interval, and reports the descriptive per-arm folding
factor and the unfolded replication. Standard library only; writes no absolute paths.
"""
import argparse
import hashlib
import json
from pathlib import Path

UNFOLDED_20261013 = 1.3239623727308008


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def classify(interval, rule):
    low, high = interval
    if low > 1.03:
        return "analytic advantage under folding"
    if low >= 0.97 and high <= 1.03:
        return "no effect"
    if high < 0.97:
        return "parameter AD faster under folding"
    return "unresolved"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "science", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.protocol)
    state = load(args.state)
    runs = {}
    for name in ("folded", "unfolded"):
        report = load(args.science / name / "report.json")
        audit = load(args.science / (name + "-audit.json"))
        row = {"driver": state["runs"].get(name), "runner_state": report and report.get("state"),
               "audit_state": audit and audit.get("state")}
        if report is not None:
            row["report_sha256"] = hashlib.sha256((args.science / name / "report.json").read_bytes()).hexdigest()
            row["zero_gate_folding_flag"] = report.get("zero_gate_folding")
            for key in ("gpu", "torch", "cuda", "setup_seconds", "folding_family_counts", "failure"):
                if key in report:
                    row[key] = report[key] if key != "failure" else str(report[key])[-1500:]
        if audit is not None:
            for key in ("ratio", "interval", "candidate_mean_seconds", "baseline_mean_seconds",
                        "maximum_error_over_original_budget", "paired_windows_checked",
                        "gradient_and_objective_comparisons", "zero_gate_folding"):
                if key in audit:
                    row[key] = audit[key]
        row["usable"] = bool(report and audit and report.get("state") == "COMPLETE" and audit.get("state") == "PASS"
                             and audit.get("gradient_and_objective_comparisons") == 78
                             and audit.get("paired_windows_checked") == 20
                             and audit.get("maximum_error_over_original_budget", 2) <= 1
                             and (name == "unfolded" or (audit.get("zero_gate_folding") or {}).get("state") == "PASS")
                             and bool(report.get("zero_gate_folding")) == (name == "folded"))
        runs[name] = row
    result = {"protocol_version": protocol["version"],
              "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              "self_test": bool(state.get("self_test")), "prerequisites_passed": state.get("prerequisites_passed"),
              "runs": runs}
    folded, unfolded = runs["folded"], runs["unfolded"]
    if folded["usable"]:
        result["primary"] = {"ratio": folded["ratio"], "interval": folded["interval"],
                             "classification": classify(folded["interval"], protocol["classification"])}
    else:
        result["primary"] = {"classification": "no timing claim: folded run failed a registered gate or did not run"}
    if unfolded["usable"]:
        result["replication"] = {"unfolded_ratio": unfolded["ratio"], "interval": unfolded["interval"],
                                 "unfolded_20261013": UNFOLDED_20261013}
    if folded["usable"] and unfolded["usable"]:
        result["folding_factor"] = {
            "analytic_candidate": unfolded["candidate_mean_seconds"] / folded["candidate_mean_seconds"],
            "parameter_AD_control": unfolded["baseline_mean_seconds"] / folded["baseline_mean_seconds"],
            "note": "descriptive: window means of sequential processes on the same GPU"}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"primary": result["primary"], "folding_factor": result.get("folding_factor"),
                      "replication": result.get("replication")}))


if __name__ == "__main__":
    main()
