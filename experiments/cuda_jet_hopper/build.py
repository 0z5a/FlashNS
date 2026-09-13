"""Build a finite, recorded FP64 Hopper configuration grid."""

import argparse
import hashlib
import itertools
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def configurations(smoke=False):
    configurations = [
        {
            "m": m,
            "n": n,
            "stages": stages,
            "copy": copy,
            "separate_epilogue": 0,
            "swizzle": 1,
        }
        for m, n, stages, copy in itertools.product(
            (32, 64), (32, 64), (1, 2, 3), (0, 1, 2)
        )
    ]
    configurations += [
        {
            "m": 64,
            "n": 64,
            "stages": stages,
            "copy": copy,
            "separate_epilogue": 1,
            "swizzle": 1,
        }
        for stages, copy in itertools.product((1, 2, 3), (0, 1, 2))
    ]
    configurations += [
        {
            "m": 64,
            "n": 64,
            "stages": 2,
            "copy": copy,
            "separate_epilogue": 0,
            "swizzle": 0,
        }
        for copy in (0, 1, 2)
    ]
    if smoke:
        configurations = [
            c
            for c in configurations
            if c["m"] == c["n"] == 64
            and c["stages"] == 2
            and c["separate_epilogue"] == 0
        ]
    return configurations


def name(config):
    return "m{m}_n{n}_s{stages}_c{copy}_e{separate_epilogue}_w{swizzle}".format(
        **config
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        HERE / "hopper.cu",
        HERE.parent / "cuda_jet_h100/stable_jet.cuh",
    ]
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "cuda_version": subprocess.check_output(["nvcc", "--version"], text=True),
        "fast_math": False,
        "architecture": "sm_90",
        "libraries": {},
        "predeclared_configurations": configurations(args.smoke),
    }
    source_dir = args.output / "sources"
    source_dir.mkdir()
    for path in paths:
        (source_dir / path.name).write_bytes(path.read_bytes())
    for config in report["predeclared_configurations"]:
        identifier = name(config)
        output = args.output / f"lib{identifier}.so"
        command = [
            "nvcc",
            "-O3",
            "-std=c++17",
            "-arch=sm_90",
            "-shared",
            "-Xcompiler=-fPIC",
            "-lineinfo",
            "--ptxas-options=-v",
            "-L/usr/local/cuda/lib64/stubs",
            "-lcuda",
            *[f"-DHOPPER_{key.upper()}={value}" for key, value in config.items()],
            str(HERE / "hopper.cu"),
            "-o",
            str(output),
        ]
        started = perf_counter()
        process = subprocess.run(command, text=True, capture_output=True)
        (args.output / f"{identifier}.log").write_text(process.stdout + process.stderr)
        report["libraries"][identifier] = {
            "config": config,
            "command": command,
            "seconds": perf_counter() - started,
            "returncode": process.returncode,
            "path": output.name,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()
            if process.returncode == 0
            else None,
        }
        (args.output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            {
                "configuration": identifier,
                "returncode": process.returncode,
                "seconds": report["libraries"][identifier]["seconds"],
            },
            flush=True,
        )
        if process.returncode:
            print(process.stderr[-8000:], flush=True)
            process.check_returncode()
    report["completed"] = True
    report["finished_utc"] = datetime.now(UTC).isoformat()
    (args.output / "build.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
