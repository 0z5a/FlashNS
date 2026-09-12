"""Adapters using public scientific-library APIs at the same FP64 target.

The cuEquivariance adapter executes full jets and their explicit VJP as segmented
polynomials. The PhysicsNeMo adapter differentiates scalar network fields.
"""

import hashlib
import importlib.metadata
import itertools
import json

import torch
from gpu import Native

from flashns.jet_spec import JetSpec
from flashns.jet_stable import stable_a1, stable_tanh, tanh_jet, tanh_vjp


def versions():
    names = (
        "torch",
        "nvidia-physicsnemo",
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-cu12",
        "cuequivariance-ops-torch-cu12",
        "numpy",
        "scipy",
        "sympy",
        "opt-einsum",
        "networkx",
    )
    return {name: importlib.metadata.version(name) for name in names}


class CuEqNative(Native):
    """FP64 library polynomials, with common GEMM and loss/seed code."""

    def __init__(self, *, forward_mode="cueq", method="uniform_1d"):
        super().__init__()
        self.forward_mode, self.method = forward_mode, method
        self.polynomials, self.descriptors = {}, []

    def polynomial(self, dim, channels, kind, device):
        key = (dim, channels, kind, str(device))
        if key in self.polynomials:
            return self.polynomials[key]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("polynomial construction must precede capture")
        import cuequivariance as cue
        import cuequivariance_ops_torch  # noqa: F401 -- fail rather than silently fall back
        import cuequivariance_torch as cuet

        indices = JetSpec(dim).coefficient_order
        q = len(indices)
        lookup = {a: i for i, a in enumerate(indices)}
        operations, audit = [], []

        def operation(mapping, segment_counts, paths):
            product = cue.SegmentedTensorProduct.from_subscripts(
                ",".join(["u"] * len(mapping))
            )
            for axis, count in enumerate(segment_counts):
                for _ in range(count):
                    product.add_segment(axis, (channels,))
            for path, coefficient in paths:
                product.add_path(*path, c=coefficient)
            operations.append((cue.Operation(mapping), product))
            audit.append(
                {"mapping": mapping, "segments": segment_counts, "paths": paths}
            )

        full = cue.SegmentedOperand([(channels,)] * q)
        if kind == "vjp":
            aux = cue.SegmentedOperand([(channels,)])
            operation([1, 2, 3], [1, q, q], [((0, i, i), 1.0) for i in range(q)])
            paths = []
            for b, beta in enumerate(indices):
                for i, alpha in enumerate(indices):
                    if not all(x >= y for x, y in zip(alpha, beta)) or i == b:
                        continue
                    delta = tuple(x - y for x, y in zip(alpha, beta))
                    for j, gamma in enumerate(indices):
                        if all(x >= y for x, y in zip(delta, gamma)):
                            k = lookup[tuple(x - y for x, y in zip(delta, gamma))]
                            paths.append(((j, k, i, b), -1.0))
            operation([0, 0, 2, 3], [q, q, q, q], paths)
            polynomial = cue.SegmentedPolynomial([full, aux, full], [full], operations)
        elif kind == "forward":
            coefficients = cue.SegmentedOperand([(channels,)] * 4)
            operation([1, 2], [4, q], [((0, 0), 1.0)])
            for power in (1, 2, 3):
                paths = []
                for factors in itertools.product(range(1, q), repeat=power):
                    alpha = tuple(
                        sum(indices[i][axis] for i in factors) for axis in range(dim)
                    )
                    if alpha in lookup:
                        paths.append(((power, *factors, lookup[alpha]), 1.0))
                operation([1, *([0] * power), 2], [4, *([q] * power), q], paths)
            polynomial = cue.SegmentedPolynomial(
                [full, coefficients], [full], operations
            )
        else:
            raise ValueError(kind)
        module = cuet.SegmentedPolynomial(
            polynomial, method=self.method, math_dtype=torch.float64
        ).to(device=device, dtype=torch.float64)
        if self.method != "naive" and "Naive" in type(module.m).__name__:
            raise RuntimeError("requested CUDA backend silently fell back to naive")
        self.descriptors.append(
            {
                "dimension": dim,
                "channels": channels,
                "kind": kind,
                "method": self.method,
                "math_dtype": "float64",
                "device": str(device),
                "canonical_operations_sha256": hashlib.sha256(
                    json.dumps(audit, sort_keys=True).encode()
                ).hexdigest(),
                "path_counts": [len(item["paths"]) for item in audit],
                "module": str(module),
            }
        )
        self.polynomials[key] = module
        return module

    def forward(self, z, dim):
        if self.forward_mode == "native":
            return super().forward(z, dim)
        self.check((z,))
        if z.ndim != 3 or z.shape[1] != JetSpec(dim).q:
            raise ValueError("full [B,Q,C] jets required")
        if z.shape[0] == 0:
            return torch.empty_like(z), torch.empty_like(z[:, 0])
        t = z[:, 0].tanh()
        aux = stable_a1(z[:, 0])
        coefficient = torch.stack((t, aux, -t * aux, aux * (t * t - 1 / 3)), 1)
        module = self.polynomial(dim, z.shape[2], "forward", z.device)
        (h,) = module([z.flatten(1), coefficient.flatten(1)])
        return h.view_as(z), aux.contiguous()

    def vjp(self, hidden, aux, adjoint, dim):
        self.check((hidden, aux, adjoint))
        if hidden.shape != adjoint.shape or aux.shape != (
            hidden.shape[0],
            hidden.shape[2],
        ):
            raise ValueError("incompatible auxiliary state or adjoint")
        if hidden.shape[0] == 0:
            return torch.empty_like(hidden)
        module = self.polynomial(dim, hidden.shape[2], "vjp", hidden.device)
        (output,) = module([hidden.flatten(1), aux, adjoint.flatten(1)])
        return output.view_as(hidden)

    def dgrad(self, d, weight, hidden, aux, dim, backend="B1"):
        if backend != "B1":
            raise ValueError(
                "scientific-library adapter uses common library GEMM plus polynomial VJP"
            )
        return self.vjp(hidden, aux, d @ weight, dim)


def torch_forward_2d(z):
    return tanh_jet(z, 2)


def torch_forward_3d(z):
    return tanh_jet(z, 3)


def torch_vjp_2d(h, aux, d):
    return tanh_vjp(h, aux, d, 2)


def torch_vjp_3d(h, aux, d):
    return tanh_vjp(h, aux, d, 3)


class TorchJetNative(Native):
    """The stable tensor-only jet implementation, eager or Torch/Inductor."""

    def __init__(self, compiled=False):
        super().__init__()
        self.compiled = compiled
        self.forward_functions = {2: torch_forward_2d, 3: torch_forward_3d}
        self.vjp_functions = {2: torch_vjp_2d, 3: torch_vjp_3d}
        if compiled:
            self.forward_functions = {
                dim: torch.compile(function, fullgraph=True, dynamic=True)
                for dim, function in self.forward_functions.items()
            }
            self.vjp_functions = {
                dim: torch.compile(function, fullgraph=True, dynamic=True)
                for dim, function in self.vjp_functions.items()
            }

    def forward(self, z, dim):
        return self.forward_functions[dim](z)

    def vjp(self, hidden, aux, adjoint, dim):
        return self.vjp_functions[dim](hidden, aux, adjoint)

    def dgrad(self, d, weight, hidden, aux, dim, backend="B1"):
        if backend != "B1":
            raise ValueError("Torch jet adapter uses common library GEMM")
        return self.vjp(hidden, aux, d @ weight, dim)


class PhysicsNeMoStep:
    """Public PhysicsInformer autodiff for the fixed 2-D steady NS target.

    PhysicsInformer supplies the three residuals and their scalar-field derivatives
    through order two. Torch differentiates those residuals in space once more and
    then computes the parameter gradient, as required by the shared loss.
    """

    def __init__(self, device="cuda:0"):
        from physicsnemo.sym.eq.pde import PDE
        from physicsnemo.sym.eq.phy_informer import PhysicsInformer
        from sympy import Function, Number, Symbol

        class Equations(PDE):
            def __init__(self):
                self.dim = 2
                x, y = Symbol("x"), Symbol("y")
                u, v, p = [Function(name)(x, y) for name in ("u", "v", "p")]
                self.equations = {
                    "momentum_x": u * u.diff(x)
                    + v * u.diff(y)
                    + p.diff(x)
                    - Number(0.07) * (u.diff(x, 2) + u.diff(y, 2)),
                    "momentum_y": u * v.diff(x)
                    + v * v.diff(y)
                    + p.diff(y)
                    - Number(0.07) * (v.diff(x, 2) + v.diff(y, 2)),
                    "continuity": u.diff(x) + v.diff(y),
                }

        self.names = ("momentum_x", "momentum_y", "continuity")
        self.informer = PhysicsInformer(
            list(self.names), Equations(), grad_method="autodiff", device=device
        )

    def __call__(
        self, x, quadrature, denominator, weights, biases, native=None, backend=None
    ):
        coordinates = x.detach().requires_grad_()
        parameters = [p.detach().requires_grad_() for p in weights + biases]
        ws, bs = parameters[: len(weights)], parameters[len(weights) :]
        h = coordinates
        for index, (w, b) in enumerate(zip(ws, bs)):
            h = h @ w.T + b
            if index + 1 < len(ws):
                h = stable_tanh(h)
        residuals = self.informer.forward(
            {
                "coordinates": coordinates,
                **{
                    name: h[:, index : index + 1]
                    for index, name in enumerate(("u", "v", "p"))
                },
            }
        )
        terms = []
        for name in self.names:
            residual = residuals[name].squeeze(-1)
            (gradient,) = torch.autograd.grad(
                residual.sum(), coordinates, create_graph=True, retain_graph=True
            )
            terms.append(residual.square() + 0.2 * gradient.square().sum(-1))
        loss = (quadrature * 0.5 * torch.stack(terms).sum(0)).sum() / denominator
        gradients = torch.autograd.grad(loss, parameters)
        return loss.detach(), [g.detach() for g in gradients]
