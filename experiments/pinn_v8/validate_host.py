"""Compile and check the exact shared scalar math as HOST C++, never as CUDA."""

import argparse
import ctypes
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from flashns.ns_seed import NSLoss, explicit_seed
from generate_residual import header_text

HERE = Path(__file__).resolve().parent


def validate():
    header = HERE / "residual_generated.cuh"
    if header.read_text() != header_text():
        raise RuntimeError("generated CUDA/HOST math differs from the current IR")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("a C++17 host compiler is required")
    with tempfile.TemporaryDirectory(prefix="flashns-v8-host-") as tmp:
        path = Path(tmp)
        wrapper = '#include "' + str(header) + '"\nextern "C" double seed(const double* z,double w,double nu,double g,double* out){return flashns_v8::residual_seed(z,w,nu,g,out);}\n'
        (path / "check.cpp").write_text(wrapper)
        library = path / "host_math.so"
        command = [compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off", "-shared", "-fPIC", str(path / "check.cpp"), "-o", str(library)]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        loaded = ctypes.CDLL(str(library))
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        loaded.seed.argtypes = [ptr, ctypes.c_double, ctypes.c_double, ctypes.c_double, ptr]
        loaded.seed.restype = ctypes.c_double
        rng = np.random.default_rng(92413)
        worst = 0.0
        cases = 0
        for nu in (0.0, 0.07, 0.2):
            for scale in (0.01, 0.4, 3.0):
                for weight in (0.0, 0.1, 1.0):
                    z = rng.normal(size=30) * scale
                    out = np.empty_like(z)
                    value = loaded.seed(z, weight, nu, 0.2, out)
                    expected, derivative = explicit_seed(torch.tensor(z.reshape(1, 10, 3)), torch.tensor([weight], dtype=torch.float64), NSLoss(viscosity=nu))
                    for actual, target in ((np.array(value), expected.numpy()), (out, derivative.numpy().reshape(-1))):
                        error = np.max(np.abs(actual-target) / (2e-12 + 2e-11*np.abs(target)))
                        if not np.isfinite(error) or error > 1:
                            raise AssertionError(f"HOST math error/tolerance={error}")
                        worst = max(worst, float(error))
                    cases += 1
        return {"passed": True, "target": "HOST C++ only; CUDA build/stream/barrier validation pending",
                "platform": platform.platform(), "cases": cases, "max_error_over_tolerance": worst,
                "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(), "compiler_command": command,
                "compiler_output": completed.stdout + completed.stderr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite a validation report")
    result = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
