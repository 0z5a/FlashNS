"""Common data/Graph/optimizer integration for v8 experiments."""

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "pinn_solver"))

from problem import Problem, digest, evaluate, independent_metrics, parameters, symbolic_check, views
from backends import Adam
from common import residual_per_point

from flashns.jet_packed import PackedLayout, TensorActivation
from flashns.pinn_packed import PackedPINN
from native import NativeCUDA


def compare(actual, expected, atol=2e-12, rtol=2e-11):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(expected, torch.Tensor):
        expected = expected.detach().cpu().numpy()
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("shape or finite-value mismatch")
    ratio = float(np.max(np.abs(actual-expected)/(atol+rtol*np.abs(expected)), initial=0))
    if ratio > 1:
        raise AssertionError(f"max error/tolerance={ratio}")
    return ratio


def source_hashes():
    files = list(HERE.glob("*.py")) + list(HERE.glob("*.cu")) + list(HERE.glob("*.cuh"))
    files += [ROOT / "src/flashns" / name for name in ("jet_spec.py", "jet_stable.py", "jet_packed.py", "ns_seed.py", "pinn_packed.py", "distributed_contract.py")]
    files += [HERE.parent / "pinn_solver" / name for name in ("problem.py", "backends.py")]
    files += [HERE.parent / "cuda_jet_h100" / name for name in ("common.py", "stable_jet.cuh", "gpu.py", "scientific_backends.py")]
    files += [HERE.parent / "cuda_jet/input/jet_reference.py", HERE.parent / "cuda_jet_hopper/hopper.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def snapshot(output):
    hashes = source_hashes()
    versions = Path(output).parent / "source_versions"
    versions.mkdir(parents=True, exist_ok=True)
    for name, sha in hashes.items():
        path = ROOT / name
        destination = versions / (sha + "-" + path.name)
        if not destination.exists():
            destination.write_bytes(path.read_bytes())
    return hashes


def frozen_protocol(path=None):
    path = Path(path) if path else ROOT / "sources/flashns_v8_reference/evidence/frozen_solver_protocol.json"
    return json.loads(path.read_text())


def make_step(problem, *, layout, seed, activation="cuda", native=None, trace=False, first_wgrad_mode="torch"):
    if first_wgrad_mode not in ("torch", "coordinate", "fused"):
        raise ValueError("unknown first wgrad mode")
    if first_wgrad_mode != "torch" and (native is None or activation != "cuda"):
        raise ValueError("native CUDA activation required for first wgrad mode")
    first_options = {}
    if first_wgrad_mode != "torch":
        first_options["coordinate_wgrad"] = native.coordinate_wgrad
    if first_wgrad_mode == "fused":
        first_options["first_activation_wgrad"] = native.coordinate_activation_wgrad
    if activation == "cuda":
        if native is None:
            raise ValueError("native CUDA activation requested without a validated build")
        factory = native.activation_factory
    elif activation in ("tensor", "compiled"):
        factory = lambda spec: TensorActivation(spec, compiled=activation == "compiled")
    else:
        raise ValueError("unsupported activation implementation")
    ni = problem.interior_count
    return PackedPINN(problem.x, ni, problem.pde_weights[:ni], problem.boundary_weights[ni:],
                      problem.target[ni:], layout=layout, seed=seed, activation_factory=factory,
                      cuda_seed=native.seed if native else None, residual_reference=residual_per_point, trace=trace, **first_options)


class Engine:
    def __init__(self, problem, initial, *, layout, seed, activation, native, first_wgrad_mode="torch"):
        start = perf_counter()
        self.flat = torch.nn.Parameter(initial.to(problem.x.device).clone())
        self.step = make_step(problem, layout=layout, seed=seed, activation=activation, native=native, first_wgrad_mode=first_wgrad_mode)
        self.metadata = dict(self.step.metadata, activation=activation, first_wgrad_mode=first_wgrad_mode,
                             logical_gemm_calls_per_evaluation=8 * len(self.step.layouts))
        if native:
            self.metadata["native_build"] = native.metadata
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(4):
                self.eager()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        capture = perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss, self.gradient = self.eager()
        self.metadata["gradient_capture_seconds"] = perf_counter() - capture
        self.metadata["setup_seconds"] = perf_counter() - start
        self.closure_calls = 0

    def eager(self):
        ws, bs = views(self.flat)
        loss, gradients = self.step(ws, bs)
        return loss, torch.cat([g.reshape(-1) for g in gradients])

    def replay(self):
        self.graph.replay()
        self.closure_calls += 1
        return self.loss, self.gradient

    def closure(self):
        loss, gradient = self.replay()
        self.flat.grad = gradient
        return loss


class EvaluationActivation:
    """Same stable full-Q native forward for every candidate's stop checks."""

    def __init__(self, native):
        self.native = native

    def forward(self, value, dim):
        layout = PackedLayout(value.shape[0], 0, dim)
        h, aux = self.native.activation_factory(layout).forward(value.flatten(0, 1))
        return h.view_as(value), aux


def ensure_preflight(path, native, *, activation="cuda"):
    report = json.loads(Path(path).read_text())
    if not report.get("passed") or report.get("source_hashes") != source_hashes():
        raise RuntimeError("successful preflight for the exact current source is required")
    if report.get("binary_sha256") != native.metadata["library_sha256"]:
        raise RuntimeError("preflight and execution binary differ")
    if activation != "cuda":
        raise RuntimeError("the current GPU preflight covers CUDA activation only; extend it before timing another activation")
    return report


def device_metadata():
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {"name": properties.name, "index": device, "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "hardware_counters": None}
