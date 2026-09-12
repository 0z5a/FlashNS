"""The same full physical loss and parameter gradient across solver backends."""

import json
import sys
from pathlib import Path
from time import perf_counter

import torch

from problem import HERE, NU, fields, views

sys.path.insert(0, str(HERE.parent / "cuda_jet_h100"))
from common import seed_coordinates
from gpu import Native
from scientific_backends import CuEqNative, PhysicsNeMoStep, TorchJetNative

sys.path.insert(0, str(HERE.parent / "cuda_jet_hopper"))
from hopper import Hopper

BACKENDS = (
    "B1",
    "cuBLASLt",
    "F",
    "HopperTMA",
    "TorchJetCompiled",
    "CuEquivariance",
    "TorchNestedAD",
    "PhysicsNeMo",
)


class FusedOriginal(Native):
    def dgrad(self, d, weight, hidden, aux, dim, backend="F"):
        return super().dgrad(
            d, weight, hidden, aux, dim, "F3" if weight.shape[0] == 3 else "F"
        )


def manual_step(problem, flat, native, mode, gemm=None):
    weights, biases = views(flat)
    with torch.no_grad():
        value = seed_coordinates(problem.x)
        checkpoints, auxiliary = [value], [None]
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            if gemm is None:
                value = value @ weight.T
            else:
                value = gemm(
                    value.flatten(0, 1),
                    weight,
                    transpose_b=True,
                    role=f"solver_forward_{index}",
                ).view(len(problem.x), 10, weight.shape[0])
            value[:, 0] += bias
            if index + 1 < len(weights):
                value, aux = native.forward(value, 2)
            else:
                aux = None
            checkpoints.append(value)
            auxiliary.append(aux)
    leaf = value.detach().requires_grad_()
    loss = problem.jet_loss(leaf)
    (d,) = torch.autograd.grad(loss, leaf)
    dws, dbs = [None] * len(weights), [None] * len(weights)
    with torch.no_grad():
        for index in reversed(range(len(weights))):
            if gemm is None:
                dws[index] = d.flatten(0, 1).T @ checkpoints[index].flatten(0, 1)
            else:
                dws[index] = gemm(
                    d.flatten(0, 1),
                    checkpoints[index].flatten(0, 1),
                    transpose_a=True,
                    role=f"solver_wgrad_{index}",
                )
            dbs[index] = d[:, 0].sum(0)
            if index:
                if gemm is not None:
                    bar = gemm(
                        d.flatten(0, 1), weights[index], role=f"solver_dgrad_{index}"
                    ).view_as(checkpoints[index])
                    d = native.vjp(checkpoints[index], auxiliary[index], bar, 2)
                else:
                    d = native.dgrad(
                        d, weights[index], checkpoints[index], auxiliary[index], 2, mode
                    )
    return loss.detach(), torch.cat([gradient.flatten() for gradient in dws + dbs])


def nested_step(problem, flat, informer=None):
    coordinates = problem.x.detach().requires_grad_()
    weights, biases = views(flat.detach())
    parameters = [parameter.detach().requires_grad_() for parameter in weights + biases]
    prediction = fields(coordinates, parameters[:3], parameters[3:])

    def gradient(value):
        return torch.autograd.grad(
            value.sum(), coordinates, create_graph=True, retain_graph=True
        )[0]

    if informer is None:
        u, v, p = prediction.unbind(-1)
        gu, gv, gp = gradient(u), gradient(v), gradient(p)
        lapu = gradient(gu[:, 0])[:, 0] + gradient(gu[:, 1])[:, 1]
        lapv = gradient(gv[:, 0])[:, 0] + gradient(gv[:, 1])[:, 1]
        residuals = (
            u * gu[:, 0] + v * gu[:, 1] + gp[:, 0] - NU * lapu,
            u * gv[:, 0] + v * gv[:, 1] + gp[:, 1] - NU * lapv,
            gu[:, 0] + gv[:, 1],
        )
    else:
        values = informer.forward(
            {
                "coordinates": coordinates,
                **{
                    name: prediction[:, index : index + 1]
                    for index, name in enumerate(("u", "v", "p"))
                },
            }
        )
        residuals = tuple(
            values[name].reshape(-1)
            for name in ("momentum_x", "momentum_y", "continuity")
        )
    terms = [
        residual.square() + 0.2 * gradient(residual).square().sum(-1)
        for residual in residuals
    ]
    loss = 0.5 * (
        problem.pde_weights * torch.stack(terms).sum(0)
    ).sum() + problem.boundary_loss(prediction)
    gradients = torch.autograd.grad(loss, parameters)
    return loss.detach(), torch.cat(
        [gradient.detach().flatten() for gradient in gradients]
    )


class Engine:
    def __init__(
        self, problem, initial, backend, hopper_build=None, hopper_configuration=None
    ):
        if backend not in BACKENDS:
            raise ValueError(backend)
        start = perf_counter()
        self.problem, self.backend = problem, backend
        self.flat = torch.nn.Parameter(initial.to(device=problem.x.device).clone())
        self.native = None
        self.gemm = None
        self.informer = None
        self.metadata = {"backend": backend}
        if backend in ("B1", "cuBLASLt"):
            self.native, self.mode = Native(), "B1"
            if backend == "cuBLASLt":
                from blaslt import Gemm

                self.gemm = Gemm()
        elif backend == "F":
            self.native, self.mode = FusedOriginal(), "F"
        elif backend == "HopperTMA":
            if hopper_configuration is None:
                result = json.loads((Path(hopper_build) / "benchmark.json").read_text())
                hopper_configuration = result["selected"]["2"]["tma"]
            self.native, self.mode = Hopper(hopper_build, hopper_configuration), "F"
            self.metadata["hopper_configuration"] = hopper_configuration
        elif backend == "TorchJetCompiled":
            self.native, self.mode = TorchJetNative(compiled=True), "B1"
        elif backend == "CuEquivariance":
            self.native, self.mode = CuEqNative(), "B1"
        elif backend == "PhysicsNeMo":
            self.informer = PhysicsNeMoStep().informer
        if self.native is not None:
            self.metadata["native_build"] = self.native.build
        if isinstance(self.native, Hopper):
            self.metadata["hopper_binary_sha256"] = self.native.entry["sha256"]
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(4):
                self.eager()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        capture_start = perf_counter()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss, self.gradient = self.eager()
        self.metadata["gradient_capture_seconds"] = perf_counter() - capture_start
        self.metadata["setup_seconds"] = perf_counter() - start
        self.closure_calls = 0

    def eager(self):
        if self.native is None:
            return nested_step(self.problem, self.flat, self.informer)
        return manual_step(self.problem, self.flat, self.native, self.mode, self.gemm)

    def replay(self):
        self.graph.replay()
        self.closure_calls += 1
        return self.loss, self.gradient

    def closure(self):
        loss, gradient = self.replay()
        self.flat.grad = gradient
        return loss

    def finish_metadata(self):
        if self.gemm is not None:
            self.metadata["gemm_planning"] = self.gemm.report()
        if isinstance(self.native, Hopper):
            self.metadata["hopper_plans"] = self.native.plan_statistics
            self.metadata["fallback_calls_during_setup"] = self.native.fallback_calls
        if isinstance(self.native, CuEqNative):
            self.metadata["cueq_descriptors"] = self.native.descriptors
        return self.metadata


class Adam:
    """One common FP64 Adam update, captured independently of the gradient graph."""

    def __init__(self, flat, gradient, learning_rate=1e-3):
        self.flat, self.gradient = flat, gradient
        self.m, self.v = torch.zeros_like(flat), torch.zeros_like(flat)
        self.count = flat.new_zeros(())
        self.lr = flat.new_tensor(learning_rate)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.eager()

    def eager(self):
        with torch.no_grad():
            self.count.add_(1)
            self.m.mul_(0.9).add_(self.gradient, alpha=0.1)
            self.v.mul_(0.999).addcmul_(self.gradient, self.gradient, value=0.001)
            corrected_m = self.m / (1 - 0.9**self.count)
            corrected_v = self.v / (1 - 0.999**self.count)
            self.flat.sub_(self.lr * corrected_m / (corrected_v.sqrt() + 1e-8))

    def step(self):
        self.graph.replay()
