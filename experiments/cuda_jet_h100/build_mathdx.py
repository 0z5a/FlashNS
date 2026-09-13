"""Compile the isolated MathDx adapter to a CUDA-driver-loadable sm_90 cubin."""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    dependencies = json.loads((HERE / "artifacts/mathdx_dependencies.json").read_text())
    toolkit = Path(dependencies["toolkit"])
    sdk = next(
        p.parent for p in Path(dependencies["sdk_root"]).rglob("include/cublasdx.hpp")
    )
    cutlass = sdk.parent / "external/cutlass/include"
    version_text = (cutlass / "cutlass/version.h").read_text()
    version = [
        int(re.search(rf"#define CUTLASS_{part}\s+(\d+)", version_text).group(1))
        for part in ("MAJOR", "MINOR", "PATCH")
    ]
    encoded = version[0] * 10000 + version[1] * 100 + version[2]
    output = HERE / "artifacts/mathdx.cubin"
    command = [
        str(toolkit / "bin/nvcc"),
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-arch=sm_90",
        "--cubin",
        "-lineinfo",
        "--ptxas-options=-v",
        "-DCUBLASDX_NO_FATBIN_AVAILABLE",
        f"-DCUBLASDX_CUTLASS_VERSION={encoded}",
        f"-I{sdk}",
        f"-I{cutlass}",
        str(HERE / "mathdx.cu"),
        "-o",
        str(output),
    ]
    start = perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    (HERE / "artifacts/build_mathdx.log").write_text(result.stdout + result.stderr)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "seconds": perf_counter() - start,
        "returncode": result.returncode,
        "path": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest()
        if not result.returncode
        else None,
        "fast_math": False,
        "accumulation": "native FP64",
        "cutlass": version,
        "mathdx_dependencies": dependencies,
        "cublasdx_version_header": (sdk / "cublasdx/cublasdx_version.hpp").read_text(),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), HERE / "mathdx.cu")
        },
        "loading": "CUDA driver API; no CUDA 13 runtime linked into the Torch CUDA 12.8 process",
    }
    (HERE / "artifacts/build_mathdx.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print({"returncode": result.returncode, "seconds": report["seconds"]}, flush=True)
    result.check_returncode()


if __name__ == "__main__":
    main()
