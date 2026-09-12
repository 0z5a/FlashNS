"""Build fresh v8 CUDA sources on the target device; no host installation changes."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import torch

from generate_residual import header_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nvcc = shutil.which("nvcc")
    if nvcc is None or not torch.cuda.is_available():
        raise RuntimeError("nvcc and a target CUDA GPU are required; no CUDA build performed")
    if (HERE / "residual_generated.cuh").read_text() != header_text():
        raise RuntimeError("generated residual source differs from the current IR")
    args.output.mkdir(parents=True, exist_ok=False)
    major, minor = torch.cuda.get_device_capability()
    library = args.output / "libflashns_v8.so"
    command = [nvcc, "-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC", "--fmad=false",
               f"-arch=sm_{major}{minor}", "--ptxas-options=-v", str(HERE / "native.cu"), "-o", str(library)]
    sources = [HERE / "native.cu", HERE / "residual_generated.cuh", HERE / "coordinate_jet.cuh",
               HERE / "coordinate_wgrad.cuh",
               HERE / "coordinate_activation_wgrad.cuh",
               HERE / "native.py", HERE / "generate_residual.py",
               HERE.parent / "cuda_jet_h100/stable_jet.cuh", ROOT / "src/flashns/ns_seed.py"]
    snapshot = args.output / "executed_sources"
    snapshot.mkdir()
    hashes = {}
    for path in sources:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        hashes[str(path.relative_to(ROOT))] = sha
        (snapshot / (sha + "-" + path.name)).write_bytes(data)
    start = perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    (args.output / "build.log").write_text(result.stdout + result.stderr)
    metadata = {"compiled": result.returncode == 0, "command": command, "exit_code": result.returncode,
                "build_seconds": perf_counter()-start, "source_sha256": hashes,
                "device_name": torch.cuda.get_device_name(), "compute_capability": [major, minor],
                "torch": torch.__version__, "nvcc": subprocess.check_output([nvcc, "--version"], text=True),
                "library": library.name, "precision": "FP64, no fast math, fmad=false",
                "gpu_correctness_and_sanitizers": "pending; compilation alone is not validation"}
    if result.returncode == 0:
        metadata["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (args.output / "build.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
