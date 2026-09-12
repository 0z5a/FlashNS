"""Freeze a v8 run schedule before timing; retain failures and per-process cost."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from runtime import HERE, ROOT, NativeCUDA, ensure_preflight, frozen_protocol, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "regression", "formal"), default="pilot")
    parser.add_argument("--selection", type=Path, help="formal: previously frozen candidate/baseline selection JSON")
    args = parser.parse_args()
    if args.phase == "formal" and args.selection is None:
        raise RuntimeError("formal runs require an explicit frozen selection; development choices are not inferred")
    candidates = [{"layout": layout, "seed_mode": seed, "activation": "cuda"} for layout in ("dense", "split", "compact") for seed in ("autograd", "cuda")]
    selection_hash = None
    if args.selection:
        selection = json.loads(args.selection.read_text())
        candidates = selection["candidates"]
        if selection["source_hashes"] != source_hashes():
            raise RuntimeError("selection source differs from current source")
        if len(candidates) < 2 or selection.get("baseline") not in candidates:
            raise RuntimeError("selection must identify a baseline and at least one candidate")
        if selection.get("protocol_sha256") != hashlib.sha256(json.dumps(frozen_protocol(), sort_keys=True).encode()).hexdigest():
            raise RuntimeError("selection protocol differs from the current frozen protocol")
        selection_hash = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    for candidate in candidates:
        if set(candidate) != {"layout", "seed_mode", "activation"} or candidate["layout"] not in ("dense", "split", "compact") or candidate["seed_mode"] not in ("autograd", "explicit", "compiled", "cuda") or candidate["activation"] != "cuda":
            raise ValueError("invalid candidate selection")
    if len({json.dumps(c, sort_keys=True) for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate")
    native = NativeCUDA(args.build)
    if ensure_preflight(args.preflight, native).get("quick"):
        raise RuntimeError("full preflight required before freezing a performance schedule")
    seeds = {"pilot": [361901, 361902], "regression": [75101, 75102, 75103], "formal": list(range(75201, 75211))}[args.phase]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = frozen_protocol()
    protocol_path = args.output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2)+"\n")
    schedule = []
    rng = random.Random(905812)
    for seed in seeds:
        order = list(candidates)
        rng.shuffle(order)
        schedule.extend({"seed": seed, **candidate} for candidate in order)
    report = {"completed": False, "phase": args.phase, "source_hashes": source_hashes(), "selection_sha256": selection_hash,
              "baseline": selection["baseline"] if args.selection else None,
              "selection": selection if args.selection else None,
              "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(), "schedule": schedule,
              "cache_policy": "preflight-populated dependency/compiler caches; fresh process/parameters/optimizer/graphs per solve", "runs": []}
    suite_path = args.output / "suite.json"
    suite_path.write_text(json.dumps(report, indent=2)+"\n")
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    for index, item in enumerate(schedule):
        name = f"{index:03d}-{item['seed']}-{item['layout']}-{item['seed_mode']}-{item['activation']}"
        output = args.output / (name + ".json")
        command = [sys.executable, str(HERE / "solve.py"), "--build", str(args.build.resolve()), "--preflight", str(args.preflight.resolve()),
                   "--protocol", str(protocol_path.resolve()), "--output", str(output.resolve()), "--seed", str(item["seed"]),
                   "--layout", item["layout"], "--seed-mode", item["seed_mode"], "--activation", item["activation"]]
        started = perf_counter()
        with (args.output / (name + ".log")).open("w") as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=ROOT)
        row = dict(item, exit_code=child.returncode, process_wall_seconds=perf_counter()-started, output=output.name)
        if output.exists():
            payload = json.loads(output.read_text())
            row["completed"] = payload.get("completed", False)
            row["result"] = payload.get("result")
        report["runs"].append(row)
        suite_path.write_text(json.dumps(report, indent=2)+"\n")
        print(name, "exit", child.returncode, "converged", bool(row.get("result", {}).get("converged") if row.get("result") else False), flush=True)
    report["completed"] = True
    report["all_converged"] = all(row["exit_code"] == 0 and row.get("completed") and row.get("result") and row["result"]["converged"] for row in report["runs"])
    report["all_attempts_process_wall_seconds"] = sum(row["process_wall_seconds"] for row in report["runs"])
    report["speedup_summary"] = "no aggregate speedup computed when any run fails; paired summary is a separate analysis"
    suite_path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("completed", "all_converged", "all_attempts_process_wall_seconds")}, indent=2))
    if any(row["exit_code"] for row in report["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
