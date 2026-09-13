"""Build one immutable set of sm-specific libraries before launching workers."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMIT = "cb4247394dd82148787aed73e5dc7cef33cbf862"


def main():
    artifacts = HERE / "artifacts"
    artifacts.mkdir(exist_ok=True)
    cutlass = ROOT / "third_party/cutlass"
    commit = subprocess.check_output(
        ["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != COMMIT:
        raise RuntimeError("unexpected CUTLASS revision")
    subprocess.run(
        ["git", "-C", str(cutlass), "diff", "--exit-code", "HEAD", "--", "include"],
        check=True,
        capture_output=True,
    )
    capability = torch.cuda.get_device_capability()
    base = [
        "nvcc",
        "-O3",
        "-std=c++17",
        f"-arch=sm_{capability[0]}{capability[1]}",
        "--expt-relaxed-constexpr",
        "-shared",
        "-Xcompiler=-fPIC",
        "-lineinfo",
        "--ptxas-options=-v",
    ]
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "cutlass_commit": commit,
        "capability": capability,
        "fast_math": False,
        "libraries": {},
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                HERE / "generate_stable_header.py",
                HERE / "stable_jet.cuh",
                HERE / "stable_kernels.cu",
                HERE / "stable_dgrad.cu",
            ]
        },
    }
    for name, source, options in [
        ("stable", "stable_kernels.cu", []),
        ("dgrad_n64", "stable_dgrad.cu", [f"-I{cutlass / 'include'}"]),
        (
            "dgrad_n32",
            "stable_dgrad.cu",
            [f"-I{cutlass / 'include'}", "-DFLASHNS_TILE_N=32"],
        ),
    ]:
        path = artifacts / f"lib{name}.so"
        command = base + options + [str(HERE / source), "-o", str(path)]
        start = perf_counter()
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        (artifacts / f"build_{name}.log").write_text(result.stdout + result.stderr)
        report["libraries"][name] = {
            "command": command,
            "seconds": perf_counter() - start,
            "returncode": result.returncode,
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            if result.returncode == 0
            else None,
        }
        (artifacts / "build.json").write_text(json.dumps(report, indent=2) + "\n")
        result.check_returncode()
        print(
            {"built": name, "seconds": report["libraries"][name]["seconds"]}, flush=True
        )


if __name__ == "__main__":
    main()
