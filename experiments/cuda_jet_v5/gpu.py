"""Explicit current-stream CUDA bindings; first parameter VJP only."""

import ctypes
import hashlib
import json
from pathlib import Path

import torch

from flashns.jet_spec import JetSpec

HERE = Path(__file__).resolve().parent
BACKENDS = ("B1", "U", "F", "U_n32", "F_n32", "U3", "F3")


class Native:
    def __init__(self):
        self.build = json.loads((HERE / "artifacts/build.json").read_text())
        self.libraries = {}
        for name, data in self.build["libraries"].items():
            path = HERE / "artifacts" / data["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != data["sha256"]:
                raise RuntimeError(f"compiled library hash changed: {name}")
            self.libraries[name] = ctypes.CDLL(str(path))
        self.activation = self.libraries["stable"].flashns_stable_activation
        self.activation.argtypes = (
            [ctypes.c_int, ctypes.c_bool]
            + [ctypes.c_void_p] * 5
            + [ctypes.c_size_t] * 2
            + [ctypes.c_void_p]
        )
        self.activation.restype = ctypes.c_int
        self.tail_launch = self.libraries["stable"].flashns_stable_tail3
        self.tail_launch.argtypes = self.activation.argtypes
        self.tail_launch.restype = ctypes.c_int
        self.dgrad_launch = {}
        for tile in (32, 64):
            function = self.libraries[f"dgrad_n{tile}"].flashns_stable_dgrad_vjp
            function.argtypes = (
                [ctypes.c_int, ctypes.c_bool]
                + [ctypes.c_void_p] * 5
                + [ctypes.c_int] * 3
                + [ctypes.c_void_p]
            )
            function.restype = ctypes.c_int
            self.dgrad_launch[tile] = function

    @staticmethod
    def check(tensors):
        first = tensors[0]
        for value in tensors:
            if (
                not value.is_cuda
                or value.dtype != torch.float64
                or value.device != first.device
                or not value.is_contiguous()
            ):
                raise ValueError("expected contiguous FP64 tensors on one CUDA device")
            if torch.is_grad_enabled() and value.requires_grad:
                raise NotImplementedError(
                    "manual parameter VJP; double backward/HVP unsupported"
                )

    @staticmethod
    def call(function, args, tensors):
        with torch.cuda.device(tensors[0].device):
            stream = torch.cuda.current_stream()
            code = function(*args, stream.cuda_stream)
            for value in tensors:
                value.record_stream(stream)
        if code:
            raise RuntimeError(f"native CUDA launch returned {code}")

    def forward(self, value, dim):
        q = JetSpec(dim).q
        self.check((value,))
        if value.ndim != 3 or value.shape[1] != q:
            raise ValueError("expected canonical full [B,Q,C] jets")
        hidden = torch.empty_like(value)
        aux = torch.empty(
            value.shape[0], value.shape[2], device=value.device, dtype=value.dtype
        )
        self.call(
            self.activation,
            [
                dim,
                False,
                value.data_ptr(),
                None,
                None,
                hidden.data_ptr(),
                aux.data_ptr(),
                value.shape[0],
                value.shape[2],
            ],
            (value, hidden, aux),
        )
        return hidden, aux

    def vjp(self, hidden, aux, adjoint, dim):
        q = JetSpec(dim).q
        self.check((hidden, aux, adjoint))
        if (
            hidden.ndim != 3
            or hidden.shape[1] != q
            or adjoint.shape != hidden.shape
            or aux.shape != (hidden.shape[0], hidden.shape[2])
        ):
            raise ValueError("incompatible full jets, auxiliary state or adjoint")
        output = torch.empty_like(hidden)
        self.call(
            self.activation,
            [
                dim,
                True,
                hidden.data_ptr(),
                adjoint.data_ptr(),
                aux.data_ptr(),
                output.data_ptr(),
                None,
                hidden.shape[0],
                hidden.shape[2],
            ],
            (hidden, aux, adjoint, output),
        )
        return output

    def dgrad(self, d, weight, hidden, aux, dim, backend="B1"):
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend}")
        q = JetSpec(dim).q
        self.check((d, weight, hidden, aux))
        if (
            d.ndim != 3
            or d.shape[1] != q
            or weight.ndim != 2
            or d.shape[2] != weight.shape[0]
            or hidden.shape != (d.shape[0], q, weight.shape[1])
            or aux.shape != (d.shape[0], weight.shape[1])
            or weight.shape[0] == 0
        ):
            raise ValueError(
                "expected D[B,Q,Cout], W[Cout,Cin], H[B,Q,Cin], aux[B,Cin]"
            )
        if backend == "B1" or (backend in ("U3", "F3") and weight.shape[0] != 3):
            return self.vjp(hidden, aux, d @ weight, dim)
        output = torch.empty_like(hidden)
        pointers = [v.data_ptr() for v in (d, weight, hidden, aux, output)]
        fused = backend.startswith("F")
        if backend in ("U3", "F3"):
            self.call(
                self.tail_launch,
                [dim, fused, *pointers, d.shape[0], weight.shape[1]],
                (d, weight, hidden, aux, output),
            )
        else:
            tile = 32 if backend.endswith("n32") else 64
            if (
                d.shape[0] > (2**31 - 1) // 20
                or weight.shape[0] > 2**31 - 17
                or weight.shape[1] > tile * 65535
            ):
                raise ValueError("CUTLASS integer or grid dimension limit exceeded")
            self.call(
                self.dgrad_launch[tile],
                [dim, fused, *pointers, d.shape[0], weight.shape[0], weight.shape[1]],
                (d, weight, hidden, aux, output),
            )
        return output if fused else self.vjp(hidden, aux, output, dim)
