"""Reproduce local v8 checks with fresh reports; never build or run CUDA."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import torch

from runtime import ROOT, Problem, compare, make_step, parameters, snapshot, views
from backends import nested_step
from validate_host import validate as validate_host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    report = {"passed": False, "completed": False, "target": "CPU and HOST C++; CUDA execution pending",
              "python": platform.python_version(), "platform": platform.platform(),
              "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "sympy", "mpmath", "torch", "pytest")},
              "source_hashes": snapshot(args.output / "validation.json"), "checks": []}
    report_path = args.output / "validation.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    def record(name, function):
        start = perf_counter()
        try:
            detail = function()
            report["checks"].append({"name": name, "passed": True, "wall_seconds": perf_counter()-start, "detail": detail})
        except Exception as error:
            report["checks"].append({"name": name, "passed": False, "error": str(error)})
            save()
            raise
        save()
        print(name, "passed", flush=True)

    def imported_reference():
        reference = ROOT / "sources/flashns_v8_reference"
        manifest = json.loads((reference / "MANIFEST.json").read_text())
        entries = manifest["files"]
        for entry in entries:
            path = reference / entry["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise AssertionError(str(path))
        return {"verified_payload_files": len(entries)}

    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")

    def command(name, arguments):
        with (args.output / (name + ".log")).open("w") as log:
            subprocess.run(arguments, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        return {"command": arguments, "log": name + ".log"}

    def full_problem():
        problem = Problem(device="cpu")
        flat = parameters(361901)
        weights, biases = views(flat)
        expected_value, expected_gradient = nested_step(problem, flat)
        rows = []
        for layout in ("dense", "split", "compact"):
            for seed in ("autograd", "explicit"):
                step = make_step(problem, layout=layout, seed=seed, activation="tensor")
                value, gradients = step(weights, biases)
                gradient = torch.cat([g.reshape(-1) for g in gradients])
                rows.append({"layout": layout, "seed": seed, "loss_error_ratio": compare(value, expected_value),
                             "all_parameters_error_ratio": compare(gradient, expected_gradient), "metadata": step.metadata})
        return {"problem": problem.meta, "parameters": flat.numel(), "checks": rows,
                "reference": "independent scalar nested AD", "is_gpu_performance_measurement": False}

    record("imported_reference_integrity", imported_reference)
    record("original_reference_suite", lambda: command("reference", [sys.executable,
           str(ROOT / "sources/flashns_v8_reference/tests/verify_math.py"), "--handoff-root", str(ROOT.parent),
           "--output", str((args.output / "reference.json").resolve())]))
    record("shared_host_cpp_math", validate_host)
    record("project_pytest", lambda: command("pytest", [sys.executable, "-m", "pytest", "-q", "--junitxml", str((args.output / "pytest.xml").resolve())]))
    record("full_2560_point_loss_gradient", full_problem)
    report.update(passed=True, completed=True)
    save()
    print(report_path)


if __name__ == "__main__":
    main()
