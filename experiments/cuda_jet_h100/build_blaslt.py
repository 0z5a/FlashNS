"""Build the isolated FP64 Lt C ABI against the installed Torch CUDA library."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import nvidia.cublas

HERE = Path(__file__).resolve().parent


def main():
    root = Path(next(iter(nvidia.cublas.__path__)))
    library = root / "lib/libcublasLt.so.12"
    artifacts = HERE / "artifacts"
    artifacts.mkdir(exist_ok=True)
    output = artifacts / "libblaslt.so"
    command = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-arch=sm_90",
        "-shared",
        "-Xcompiler=-fPIC",
        f"-I{root / 'include'}",
        str(HERE / "blaslt.cu"),
        f"-Xlinker={library}",
        f"-Xlinker=-rpath,{root / 'lib'}",
        "-o",
        str(output),
    ]
    start = perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    (artifacts / "build_blaslt.log").write_text(result.stdout + result.stderr)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "seconds": perf_counter() - start,
        "returncode": result.returncode,
        "fast_math": False,
        "path": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest()
        if not result.returncode
        else None,
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), HERE / "blaslt.cu")
        },
        "cublaslt_library": str(library),
        "cublaslt_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    (artifacts / "build_blaslt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)
    result.check_returncode()


if __name__ == "__main__":
    main()
