"""Retain capped outcomes and rerun only failures with one uniform larger cap.

All acceptance metrics, frequencies, data, initialization and optimizer choices
remain identical. Already converged runs need no further iterations and retain
their measured stopping points. Fresh reruns include their entire setup/training.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from run_suite import sha256, telemetry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hopper-build", type=Path, required=True)
    parser.add_argument("--lbfgs-max-blocks", type=int, default=500)
    args = parser.parse_args()
    previous = json.loads(args.suite.read_text())
    if not previous["completed"] or not previous["all_runs_completed"]:
        raise RuntimeError("complete the original fixed-budget suite first")
    if args.output_dir.exists():
        raise RuntimeError("extension requires a new directory")
    previous_protocol = args.suite.parent / "protocol.json"
    if sha256(previous_protocol) != previous["protocol_sha256"]:
        raise RuntimeError("original protocol changed")
    configuration = json.loads(previous_protocol.read_text())
    original_cap = configuration["lbfgs_max_blocks"]
    if args.lbfgs_max_blocks <= original_cap:
        raise ValueError("the supplementary cap must be larger")
    configuration["lbfgs_max_blocks"] = args.lbfgs_max_blocks
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    for row in previous["runs"]:
        path = args.suite.parent / row["output"]
        if sha256(path) != row["output_sha256"]:
            raise RuntimeError(f"original run changed: {path}")
        child = json.loads(path.read_text())
        for relative, expected in child["source_hashes"].items():
            if sha256(root / relative) != expected:
                raise RuntimeError(f"solver source changed: {relative}")
    args.output_dir.mkdir(parents=True)
    protocol = args.output_dir / "protocol.json"
    protocol.write_text(json.dumps(configuration, indent=2) + "\n")
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "original_suite": str(args.suite),
        "original_suite_sha256": sha256(args.suite),
        "original_protocol_sha256": sha256(previous_protocol),
        "protocol_sha256": sha256(protocol),
        "extension_source_sha256": sha256(__file__),
        "only_protocol_change": {
            "lbfgs_max_blocks": [original_cap, args.lbfgs_max_blocks]
        },
        "policy": "Uniform larger available cap for all backends/seeds; previously converged runs retain their stopping points; all original nonconverged runs restart from the same initialization under the supplementary cap. No threshold, optimizer, data, network or check-frequency changes. Original capped failures remain in the original suite.",
        "timing_policy": "effective time-to-acceptance is the full converged attempt; additionally report actual sum of all attempt process wall times. This is a supplementary extension, not a replacement of the original fixed-budget experiment.",
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "PYTHONPATH",
            )
        },
        "runs": [],
        "completed": False,
    }
    output = args.output_dir / "suite.json"

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for number, original in enumerate(previous["runs"], 1):
        row = {key: original[key] for key in ("seed", "backend")}
        row["original_output"] = str(args.suite.parent / original["output"])
        row["original_output_sha256"] = original["output_sha256"]
        row["original_converged"] = original["result"]["converged"]
        if row["original_converged"]:
            row.update(
                {
                    "reused_converged_run": True,
                    "result": original["result"],
                    "process_wall_seconds": original["process_wall_seconds"],
                    "all_attempts_process_wall_seconds": original[
                        "process_wall_seconds"
                    ],
                    "output": row["original_output"],
                    "output_sha256": row["original_output_sha256"],
                    "completed": True,
                    "returncode": 0,
                }
            )
        else:
            name = f"{number:02d}-{row['seed']}-{row['backend']}"
            child_output = args.output_dir / (name + ".json")
            child_log = args.output_dir / (name + ".log")
            row["reused_converged_run"] = False
            row["before"] = telemetry()
            command = [
                sys.executable,
                str(here / "solve.py"),
                "--backend",
                row["backend"],
                "--seed",
                str(row["seed"]),
                "--output",
                str(child_output),
                "--protocol",
                str(protocol),
                "--hopper-build",
                str(args.hopper_build),
            ]
            row["command"] = command
            print(
                {
                    "starting_extension": number,
                    "backend": row["backend"],
                    "seed": row["seed"],
                },
                flush=True,
            )
            started = perf_counter()
            with child_log.open("w") as stream:
                process = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            row["process_wall_seconds"] = perf_counter() - started
            row["all_attempts_process_wall_seconds"] = (
                row["process_wall_seconds"] + original["process_wall_seconds"]
            )
            row["returncode"] = process.returncode
            row["after"] = telemetry()
            row["output"] = str(child_output)
            row["log"] = str(child_log)
            row["log_sha256"] = sha256(child_log)
            row["completed"] = False
            if child_output.exists():
                row["output_sha256"] = sha256(child_output)
                child = json.loads(child_output.read_text())
                row["completed"] = child["completed"]
                if row["completed"]:
                    row["result"] = {
                        key: value
                        for key, value in child["result"].items()
                        if key not in ("history", "backend_setup")
                    }
                    row["checkpoint_sha256"] = child["checkpoint_sha256"]
            print(
                {
                    "finished_extension": number,
                    "backend": row["backend"],
                    "seed": row["seed"],
                    "converged": row.get("result", {}).get("converged"),
                    "process_wall_seconds": row["process_wall_seconds"],
                },
                flush=True,
            )
        report["runs"].append(row)
        save()
    report["completed"] = True
    report["all_runs_completed"] = all(
        row["completed"] and row["returncode"] == 0 for row in report["runs"]
    )
    report["all_runs_converged"] = all(
        row.get("result", {}).get("converged", False) for row in report["runs"]
    )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    save()


if __name__ == "__main__":
    main()
