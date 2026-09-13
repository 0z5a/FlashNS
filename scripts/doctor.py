"""Read-only environment snapshot for the native CUDA experiment.

No installation, driver changes, or implied build certification. A separate
native build/run smoke test is required. No complete environment dump.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def command(argv):
    executable = shutil.which(argv[0])
    if executable is None:
        return {"available": False}
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=20, check=False
        )
        return {
            "available": True,
            "path": executable,
            "returncode": result.returncode,
            "stdout": result.stdout[:18000],
            "stderr": result.stderr[:4000],
        }
    except subprocess.TimeoutExpired:
        return {"available": True, "path": executable, "timeout": True}


def snapshot():
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "scope": "read-only inventory; actual native build/run and sanitizer results are separate",
        "packages": {},
        "commands": {},
    }
    for name in [
        "torch",
        "numpy",
        "scipy",
        "sympy",
        "mpmath",
        "pytest",
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "ninja",
        "setuptools",
        "wheel",
        "packaging",
    ]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    for name, argv in {
        "gpu_driver": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap,mig.mode.current,memory.total",
            "--format=csv",
        ],
        "nvcc": ["nvcc", "--version"],
        "host_cxx": ["c++", "--version"],
        "compute_sanitizer": ["compute-sanitizer", "--version"],
        "nsight_compute": ["ncu", "--version"],
        "nsight_systems": ["nsys", "--version"],
    }.items():
        report["commands"][name] = command(argv)
    if report["packages"]["torch"]:
        import torch

        available = torch.cuda.is_available()
        report["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cxx11_abi": torch.compiled_with_cxx11_abi(),
            "cuda_available": available,
            "matmul_tensor_dtype": "float64 for all experiment paths",
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": torch.cuda.get_device_capability(i),
                    "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
            if available
            else [],
        }
        library = Path(torch.__file__).parent / "lib/libtorch_cuda.so"
        if library.exists():
            report["torch_resolved_cuda_libraries"] = command(["ldd", str(library)])
    cutlass = Path("third_party/cutlass")
    report["cutlass"] = {"available": cutlass.is_dir()}
    if (cutlass / ".git").exists():
        report["cutlass"]["commit"] = command(
            ["git", "-C", str(cutlass), "rev-parse", "HEAD"]
        )
    if (cutlass / "include/cutlass/version.h").is_file():
        report["cutlass"]["version_header_sha256"] = hashlib.sha256(
            (cutlass / "include/cutlass/version.h").read_bytes()
        ).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/environment.json")
    )
    args = parser.parse_args()
    report = snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    ready = bool(report.get("torch", {}).get("cuda_available"))
    print(
        {
            "report": str(args.output),
            "cuda_available": ready,
            "native_build_tested": False,
        }
    )
    return 1 if args.require_gpu and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
