"""Bounded, process-local FP64 plan selection for serialized CUDA submissions.

Plans are not portable caches. Graphs and callers must retain this object and
serialize use of its workspaces, including establishing cross-stream ordering.
"""

import ctypes as ct
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
from common import comparison
from gpu import Native

HERE = Path(__file__).resolve().parent


def alignment(value):
    address = value.data_ptr()
    return min(256, address & -address)


class Library:
    def __init__(self):
        self.build = json.loads((HERE / "artifacts/build_blaslt.json").read_text())
        path = HERE / "artifacts" / self.build["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.build["sha256"]:
            raise RuntimeError("Lt binary hash changed")
        self.lib = ct.CDLL(str(path))
        self.lib.flashns_lt_create.argtypes = [
            ct.c_int64,
            ct.c_int64,
            ct.c_int64,
            ct.c_bool,
            ct.c_bool,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_size_t,
            ct.c_int,
            ct.POINTER(ct.c_void_p),
        ]
        self.lib.flashns_lt_create.restype = ct.c_int
        self.lib.flashns_lt_count.argtypes = [ct.c_void_p]
        self.lib.flashns_lt_count.restype = ct.c_int
        self.lib.flashns_lt_info.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_size_t),
            ct.POINTER(ct.c_float),
        ]
        self.lib.flashns_lt_info.restype = ct.c_int
        self.lib.flashns_lt_run.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_void_p,
        ]
        self.lib.flashns_lt_run.restype = ct.c_int
        self.lib.flashns_lt_destroy.argtypes = [ct.c_void_p]
        self.lib.flashns_lt_destroy.restype = None


class Plan:
    def __init__(self, library, key, budget, candidates):
        self.library, self.key = library, key
        self.pointer = ct.c_void_p()
        self.workspace = None
        self.selected = "torch"
        _, m, n, k, ta, tb, aa, ab = key
        self.status = library.lib.flashns_lt_create(
            m, n, k, ta, tb, aa, ab, budget, candidates, ct.byref(self.pointer)
        )

    def __del__(self):
        if self.pointer:
            self.library.lib.flashns_lt_destroy(self.pointer)

    def info(self, index):
        attrs, workspace, waves = (ct.c_int * 6)(), ct.c_size_t(), ct.c_float()
        status = self.library.lib.flashns_lt_info(
            self.pointer, index, attrs, ct.byref(workspace), ct.byref(waves)
        )
        return {
            "index": index,
            "heuristic_status": status,
            "workspace_bytes": workspace.value,
            "predicted_waves": waves.value,
            **dict(
                zip(
                    ("algorithm", "tile", "split_k", "reduction", "swizzle", "stages"),
                    attrs,
                )
            ),
        }

    def run(self, index, a, b, output):
        if index == "torch":
            return torch.mm(
                a.T if self.key[4] else a, b.T if self.key[5] else b, out=output
            )
        stream = torch.cuda.current_stream(a.device)
        status = self.library.lib.flashns_lt_run(
            self.pointer,
            index,
            a.data_ptr(),
            b.data_ptr(),
            output.data_ptr(),
            self.workspace.data_ptr() if self.workspace is not None else None,
            self.workspace.numel() if self.workspace is not None else 0,
            stream.cuda_stream,
        )
        for value in (a, b, output, self.workspace):
            if value is not None:
                value.record_stream(stream)
        if status:
            raise RuntimeError(f"cuBLASLt status {status}")
        return output


class Gemm:
    def __init__(self, budget=8 * 1024**2, candidates=8, repeats=7, inner=16):
        self.library = Library()
        self.budget, self.candidates = budget, candidates
        self.repeats, self.inner = repeats, inner
        self.plans, self.records = {}, []
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def __call__(self, a, b, transpose_a=False, transpose_b=False, role="unspecified"):
        Native.check((a, b))
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("rank two contiguous storage required")
        m, k = (a.shape[1], a.shape[0]) if transpose_a else a.shape
        kb, n = (b.shape[1], b.shape[0]) if transpose_b else b.shape
        if k != kb or min(m, n, k) <= 0:
            raise ValueError("positive compatible dimensions required")
        key = (
            a.device.index,
            m,
            n,
            k,
            transpose_a,
            transpose_b,
            alignment(a),
            alignment(b),
        )
        if key not in self.plans:
            if self.frozen or torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"untuned shape after plan freeze: {key}")
            self.plans[key] = self.select(key, a, b, role)
        plan = self.plans[key]
        output = torch.empty((m, n), dtype=a.dtype, device=a.device)
        return plan.run(plan.selected, a, b, output)

    def select(self, key, a, b, role):
        start = perf_counter()
        plan = Plan(self.library, key, self.budget, self.candidates)
        plan.workspace = torch.empty(self.budget, dtype=torch.uint8, device=a.device)
        output = torch.empty((key[1], key[2]), dtype=a.dtype, device=a.device)
        reference = torch.mm(a.T if key[4] else a, b.T if key[5] else b)
        candidates = [{"index": "torch", "workspace_bytes": None}]
        if not plan.status:
            candidates.extend(
                plan.info(i)
                for i in range(self.library.lib.flashns_lt_count(plan.pointer))
            )
        functions, successful = {}, []
        for item in candidates:
            if item.get("heuristic_status", 0):
                item["failure"] = "heuristic status not success"
                continue
            function = partial(plan.run, item["index"], a, b, output)
            try:
                item["validation"] = comparison(function(), reference)
            except (RuntimeError, AssertionError) as error:
                item["failure"] = str(error)
                continue
            functions[item["index"]] = function
            successful.append(item)
        if "torch" not in functions:
            raise RuntimeError("Torch baseline validation failed")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        with torch.cuda.stream(stream):
            for index, function in functions.items():
                for _ in range(3):
                    function()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(self.inner):
                        function()
                graphs[index] = graph
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        observations = []
        indices = list(graphs)
        for repeat in range(self.repeats):
            order = indices[repeat % len(indices) :] + indices[: repeat % len(indices)]
            if repeat % 2:
                order.reverse()
            row = {"repeat": repeat, "order": order, "cuda_event_ms": {}}
            for index in order:
                first, last = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                first.record()
                graphs[index].replay()
                last.record()
                last.synchronize()
                row["cuda_event_ms"][str(index)] = first.elapsed_time(last) / self.inner
            observations.append(row)
        medians = {
            index: statistics.median(
                row["cuda_event_ms"][str(index)] for row in observations
            )
            for index in indices
        }
        best = min(medians, key=medians.get)
        plan.selected = "torch" if medians["torch"] <= medians[best] * 1.02 else best
        selected_workspace = next(
            item["workspace_bytes"]
            for item in successful
            if item["index"] == plan.selected
        )
        torch.cuda.synchronize()
        del graph, graphs, functions, function
        plan.workspace = (
            torch.empty(selected_workspace, dtype=torch.uint8, device=a.device)
            if selected_workspace
            else None
        )
        comparison(plan.run(plan.selected, a, b, output), reference)
        self.records.append(
            {
                "key": {
                    "device": key[0],
                    "m": key[1],
                    "n": key[2],
                    "k": key[3],
                    "transpose_a": key[4],
                    "transpose_b": key[5],
                    "alignment_a": key[6],
                    "alignment_b": key[7],
                },
                "first_use_role": role,
                "a_stride": list(a.stride()),
                "b_stride": list(b.stride()),
                "create_status": plan.status,
                "candidates": candidates,
                "paired_observations": observations,
                "median_cuda_event_ms": {str(k): v for k, v in medians.items()},
                "selected": plan.selected,
                "selected_lt_workspace_bytes": selected_workspace or 0,
                "torch_workspace_bytes": None,
                "selection_seconds": perf_counter() - start,
            }
        )
        return plan

    def report(self):
        props = torch.cuda.get_device_properties()
        return {
            "gpu": props.name,
            "uuid": str(props.uuid),
            "capability": [props.major, props.minor],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "build": self.library.build,
            "precision": "FP64 native, stable_aux_a1",
            "jet": {"dimension": 2, "order": 3, "q": 10},
            "layout": "contiguous row-major storage; output alignment 256 bytes",
            "concurrency": "serialized calls; cross-stream ordering required; one workspace per plan",
            "persistence": "process-local only; never reload algorithms on another environment",
            "workspace_budget_bytes_per_plan": self.budget,
            "requested_candidates": self.candidates,
            "selection": "7 paired rounds, graph of 16 identical GEMMs per timing; Torch wins within 2 percent",
            "scope": "finite heuristic pool plus Torch, not global optimum; repeats reuse identical matrices",
            "torch_workspace_accounting": "internal workspace not exposed by this API",
            "plans": self.records,
        }
