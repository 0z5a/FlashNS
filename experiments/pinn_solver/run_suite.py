"""Sequential paired-seed solves after a frozen protocol and numerical preflight."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def telemetry():
    result = {}
    for name, args in {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ],
        "compute_processes": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
    }.items():
        proc = subprocess.run(args, text=True, capture_output=True, check=False)
        result[name] = {"returncode": proc.returncode, "stdout": proc.stdout}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--hopper-build", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("formal suite must use a new directory")

    from backends import BACKENDS
    from solve import protocol

    here = Path(__file__).resolve().parent
    root = here.parents[1]
    preflight = json.loads(args.preflight.read_text())
    pilot = json.loads(args.pilot.read_text())
    if not preflight["passed"] or set(preflight["backends"]) != set(BACKENDS):
        raise RuntimeError("all backends must pass the numerical preflight")
    for relative, expected in preflight["source_hashes"].items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"source changed after preflight: {relative}")
    configuration = protocol()
    if not pilot["completed"] or not pilot["result"]["converged"]:
        raise RuntimeError("the distinct development pilot must have converged")
    if pilot["protocol"] != configuration:
        raise RuntimeError("protocol changed after the development pilot")

    args.output_dir.mkdir(parents=True)
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(configuration, indent=2) + "\n")
    randomizer = random.Random(90109)
    schedule = []
    for seed in configuration["formal_initialization_seeds"]:
        order = list(BACKENDS)
        randomizer.shuffle(order)
        schedule.extend({"seed": seed, "backend": backend} for backend in order)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha256(protocol_path),
        "preflight_sha256": sha256(args.preflight),
        "pilot_sha256": sha256(args.pilot),
        "suite_source_sha256": sha256(__file__),
        "randomization_seed": 90109,
        "schedule": schedule,
        "policy": "24 sequential, fresh-process runs; 3 paired initialization seeds; no per-backend tuning; compiler caches populated by the preflight; nonconvergence and exceptions remain in results",
        "process_wall_timing": "includes Python/dependency imports, source hashing, symbolic check, problem preparation, backend setup, training, stop checks, checkpoint writing and process teardown; nvidia-smi probes excluded",
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
    suite_path = args.output_dir / "suite.json"

    def save():
        suite_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for number, item in enumerate(schedule, 1):
        name = f"{number:02d}-{item['seed']}-{item['backend']}"
        output = args.output_dir / (name + ".json")
        log = args.output_dir / (name + ".log")
        command = [
            sys.executable,
            str(here / "solve.py"),
            "--backend",
            item["backend"],
            "--seed",
            str(item["seed"]),
            "--output",
            str(output),
            "--protocol",
            str(protocol_path),
            "--hopper-build",
            str(args.hopper_build),
        ]
        row = {**item, "output": output.name, "log": log.name, "command": command}
        row["before"] = telemetry()
        print({"starting": number, "of": len(schedule), **item}, flush=True)
        start = perf_counter()
        with log.open("w") as stream:
            process = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        row["process_wall_seconds"] = perf_counter() - start
        row["returncode"] = process.returncode
        row["after"] = telemetry()
        row["log_sha256"] = sha256(log)
        if output.exists():
            child = json.loads(output.read_text())
            row["output_sha256"] = sha256(output)
            row["completed"] = child.get("completed", False)
            if row["completed"]:
                row["result"] = {
                    key: value
                    for key, value in child["result"].items()
                    if key not in ("history", "backend_setup")
                }
                row["checkpoint_sha256"] = child["checkpoint_sha256"]
        report["runs"].append(row)
        save()
        print(
            {
                "finished": number,
                **item,
                "returncode": row["returncode"],
                "process_wall_seconds": row["process_wall_seconds"],
                "converged": row.get("result", {}).get("converged"),
            },
            flush=True,
        )
    report["completed"] = True
    report["all_runs_completed"] = all(
        row.get("completed") and row["returncode"] == 0 for row in report["runs"]
    )
    report["all_runs_converged"] = all(
        row.get("result", {}).get("converged", False) for row in report["runs"]
    )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    save()


if __name__ == "__main__":
    main()
