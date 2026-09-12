"""Current-stream bindings for matched Hopper copies and explicit parameter VJP."""

import ctypes
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cuda_jet_h100"))
from gpu import Native

from flashns.jet_spec import JetSpec


class Hopper(Native):
    def __init__(self, build_directory, configuration, cache_limit=16):
        start = perf_counter()
        super().__init__()
        if torch.cuda.get_device_capability() != (9, 0):
            raise ValueError("this experiment is built specifically for sm_90")
        self.directory = Path(build_directory)
        report = json.loads((self.directory / "build.json").read_text())
        self.entry = report["libraries"][configuration]
        self.config = self.entry["config"]
        path = self.directory / self.entry["path"]
        if (
            self.entry["returncode"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != self.entry["sha256"]
        ):
            raise RuntimeError("Hopper binary provenance mismatch")
        self.library = ctypes.CDLL(str(path))
        self.initialize = self.library.flashns_hopper_initialize
        self.initialize.argtypes, self.initialize.restype = [], ctypes.c_int
        self.query_resources = self.library.flashns_hopper_resources
        self.query_resources.argtypes = [
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        self.query_resources.restype = ctypes.c_int
        self.create = self.library.flashns_hopper_create
        self.create.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.create.restype = ctypes.c_int
        self.destroy = self.library.flashns_hopper_destroy
        self.destroy.argtypes, self.destroy.restype = [ctypes.c_void_p], None
        self.launch = self.library.flashns_hopper_launch
        self.launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool] + [
            ctypes.c_void_p
        ] * 4
        self.launch.restype = ctypes.c_int
        code = self.initialize()
        if code:
            raise RuntimeError(f"Hopper initialization failed: {code}")
        self.cache_limit = cache_limit
        self.plans = OrderedDict()
        self.plan_statistics = {
            "created": 0,
            "during_graph_capture": 0,
            "host_seconds": 0.0,
            "evicted": 0,
        }
        self.fallback_calls = 0
        self.initialize_seconds = perf_counter() - start

    def resources(self):
        keys = (
            "registers_per_thread",
            "threads_per_cta",
            "dynamic_shared_bytes",
            "static_shared_bytes",
            "local_bytes_per_thread",
            "predicted_active_ctas_per_sm",
            "max_dynamic_shared_bytes",
            "binary_version",
            "ptx_version",
        )
        output = []
        for dim in (2, 3):
            for fused in (False, True):
                data = (ctypes.c_longlong * 9)()
                code = self.query_resources(dim, fused, data)
                if code:
                    raise RuntimeError(f"resource query failed: {code}")
                values = dict(zip(keys, data))
                values.update(
                    dimension=dim,
                    fused=fused,
                    predicted_active_warps_per_sm=values["predicted_active_ctas_per_sm"]
                    * values["threads_per_cta"]
                    // 32,
                    occupancy_source="cudaOccupancyMaxActiveBlocksPerMultiprocessor; prediction, not a hardware counter",
                )
                output.append(values)
        return output

    def plan(self, d, weight, dim):
        key = (
            d.data_ptr(),
            weight.data_ptr(),
            tuple(d.shape),
            tuple(weight.shape),
            dim,
            str(d.device),
        )
        if key in self.plans:
            self.plans.move_to_end(key)
            return self.plans[key][0]
        start = perf_counter()
        result = ctypes.c_void_p()
        code = self.create(
            dim,
            d.data_ptr(),
            weight.data_ptr(),
            d.shape[0],
            weight.shape[0],
            weight.shape[1],
            ctypes.byref(result),
        )
        if code:
            raise RuntimeError(f"Hopper plan encoding failed: {code}")
        self.plan_statistics["host_seconds"] += perf_counter() - start
        self.plan_statistics["created"] += 1
        self.plan_statistics["during_graph_capture"] += int(
            torch.cuda.is_current_stream_capturing()
        )
        # Tensor maps are immutable launch parameters. Encoding performs host
        # work only, including when capture first assigns its pool addresses.
        # Report that setup work separately; replay does not encode descriptors.
        self.plans[key] = (result, d, weight)
        if len(self.plans) > self.cache_limit:
            _, old = self.plans.popitem(last=False)
            self.destroy(old[0])
            self.plan_statistics["evicted"] += 1
        return result

    def raw_dgrad(self, d, weight, hidden, aux, dim, fused, output=None):
        q = JetSpec(dim).q
        self.check((d, weight, hidden, aux))
        if (
            d.ndim != 3
            or d.shape[1] != q
            or weight.ndim != 2
            or d.shape[2] != weight.shape[0]
            or hidden.shape != (d.shape[0], q, weight.shape[1])
            or aux.shape != (d.shape[0], weight.shape[1])
        ):
            raise ValueError(
                "expected D[B,Q,Cout], W[Cout,Cin], H[B,Q,Cin], aux[B,Cin]"
            )
        if weight.shape[0] not in (32, 64) or weight.shape[1] not in (32, 64):
            raise ValueError("matched Hopper path supports Cin/Cout=32/64 only")
        if d.data_ptr() % 16 or weight.data_ptr() % 16:
            raise ValueError("TMA and matched-copy inputs require 16-byte alignment")
        if output is None:
            output = torch.empty_like(hidden)
        self.check((d, output))
        if output.shape != hidden.shape:
            raise ValueError("output shape mismatch")
        if d.shape[0] == 0:
            return output
        handle = self.plan(d, weight, dim)
        self.call(
            self.launch,
            [handle, dim, fused, hidden.data_ptr(), aux.data_ptr(), output.data_ptr()],
            (d, weight, hidden, aux, output),
        )
        return output

    def dgrad(self, d, weight, hidden, aux, dim, backend="F"):
        if backend not in ("U", "F", "B1"):
            raise ValueError(backend)
        if backend == "B1":
            return super().dgrad(d, weight, hidden, aux, dim, "B1")
        if weight.shape[0] == 3:
            self.fallback_calls += 1
            return super().dgrad(d, weight, hidden, aux, dim, backend + "3")
        if weight.shape[0] not in (32, 64) or weight.shape[1] not in (32, 64):
            self.fallback_calls += 1
            return super().dgrad(d, weight, hidden, aux, dim, "B1")
        output = self.raw_dgrad(d, weight, hidden, aux, dim, backend == "F")
        return output if backend == "F" else self.vjp(hidden, aux, output, dim)

    def close(self):
        for plan in self.plans.values():
            self.destroy(plan[0])
        self.plans.clear()

    def __del__(self):
        if hasattr(self, "plans"):
            self.close()
