"""CUDA-driver loading of native-FP64 cuBLASDx cubins in the Torch 12.8 process."""

import ctypes as ct
import hashlib
import json
from pathlib import Path

import torch
from gpu import Native

HERE = Path(__file__).resolve().parent
SHAPES = ((32, 2), (32, 3), (32, 32), (3, 32), (64, 2), (64, 3), (64, 64), (3, 64))


class MathDx:
    def __init__(self):
        torch.cuda.init()
        torch.empty(1, device="cuda:0")
        self.build = json.loads((HERE / "artifacts/build_mathdx.json").read_text())
        path = HERE / "artifacts" / self.build["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.build["sha256"]:
            raise RuntimeError("MathDx cubin hash changed")
        self.lib = ct.CDLL("libcuda.so.1")
        self.lib.cuModuleLoad.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p]
        self.lib.cuModuleLoad.restype = ct.c_int
        self.lib.cuModuleGetFunction.argtypes = [
            ct.POINTER(ct.c_void_p),
            ct.c_void_p,
            ct.c_char_p,
        ]
        self.lib.cuModuleGetFunction.restype = ct.c_int
        self.lib.cuModuleGetGlobal_v2.argtypes = [
            ct.POINTER(ct.c_uint64),
            ct.POINTER(ct.c_size_t),
            ct.c_void_p,
            ct.c_char_p,
        ]
        self.lib.cuModuleGetGlobal_v2.restype = ct.c_int
        self.lib.cuMemcpyDtoH_v2.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_size_t]
        self.lib.cuMemcpyDtoH_v2.restype = ct.c_int
        self.lib.cuFuncSetAttribute.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
        self.lib.cuFuncSetAttribute.restype = ct.c_int
        self.lib.cuLaunchKernel.argtypes = [
            ct.c_void_p,
            *([ct.c_uint] * 7),
            ct.c_void_p,
            ct.POINTER(ct.c_void_p),
            ct.POINTER(ct.c_void_p),
        ]
        self.lib.cuLaunchKernel.restype = ct.c_int
        self.module = ct.c_void_p()
        self.check(self.lib.cuModuleLoad(ct.byref(self.module), str(path).encode()))
        self.kernels, self.configurations = {}, {}
        for tile in (32, 64):
            for n, k in SHAPES:
                name = f"{tile}_{n}_{k}"
                function, address, size = ct.c_void_p(), ct.c_uint64(), ct.c_size_t()
                self.check(
                    self.lib.cuModuleGetFunction(
                        ct.byref(function), self.module, f"dx_{name}".encode()
                    )
                )
                self.check(
                    self.lib.cuModuleGetGlobal_v2(
                        ct.byref(address),
                        ct.byref(size),
                        self.module,
                        f"cfg_{name}".encode(),
                    )
                )
                if size.value != 16:
                    raise RuntimeError("unexpected cubin configuration metadata")
                config = (ct.c_int * 4)()
                self.check(self.lib.cuMemcpyDtoH_v2(config, address.value, size.value))
                # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, CUDA driver header.
                self.check(self.lib.cuFuncSetAttribute(function, 8, config[3]))
                self.kernels[(tile, n, k)] = function
                self.configurations[name] = {
                    "block": list(config[:3]),
                    "dynamic_shared_bytes": config[3],
                }

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"CUDA driver status {status}")

    def mm(self, a, b, *, transpose_b=False, tile=64):
        Native.check((a, b))
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("rank two contiguous matrices required")
        m, k = a.shape
        n, kb = b.shape if transpose_b else (b.shape[1], b.shape[0])
        if kb != k or (tile, n, k) not in self.kernels:
            raise ValueError("unsupported MathDx tile or matrix shape")
        output = torch.empty(m, n, device=a.device, dtype=a.dtype)
        if not m:
            return output
        if a.device.index != 0:
            raise ValueError("this isolated module is loaded on GPU 0")
        config = self.configurations[f"{tile}_{n}_{k}"]
        stream = torch.cuda.current_stream(a.device)
        values = [
            ct.c_void_p(a.data_ptr()),
            ct.c_void_p(b.data_ptr()),
            ct.c_void_p(output.data_ptr()),
            ct.c_int64(m),
            ct.c_int(int(transpose_b)),
        ]
        arguments = (ct.c_void_p * len(values))(
            *(ct.cast(ct.pointer(value), ct.c_void_p) for value in values)
        )
        self.check(
            self.lib.cuLaunchKernel(
                self.kernels[(tile, n, k)],
                (m + tile - 1) // tile,
                1,
                1,
                *config["block"],
                config["dynamic_shared_bytes"],
                stream.cuda_stream,
                arguments,
                None,
            )
        )
        for value in (a, b, output):
            value.record_stream(stream)
        return output

    def gemm(
        self, a, b, transpose_a=False, transpose_b=False, role="unspecified", *, tile=64
    ):
        # Global wgrad reduction remains Torch GEMM. The finite Dx pool covers
        # forward and dgrad shapes for width 32/64; unsupported tiny tests use Torch.
        n = b.shape[0] if transpose_b else b.shape[1]
        k = a.shape[0] if transpose_a else a.shape[1]
        if transpose_a or (tile, n, k) not in self.kernels:
            return torch.mm(a.T if transpose_a else a, b.T if transpose_b else b)
        return self.mm(a, b, transpose_b=transpose_b, tile=tile)

    def report(self):
        return {
            "build": self.build,
            "configurations": self.configurations,
            "coverage": "native FP64 forward/dgrad shapes at C=32/64, K=2/3/32/64; wgrad and other shapes use Torch",
            "lifetime": "cubin retained for this CUDA context; keep this object alive with captured graphs",
            "selection": "two explicit M tiles 32 and 64; finite adapter comparison, not globally tuned MathDx performance",
        }
