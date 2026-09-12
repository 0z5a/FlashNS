"""Explicit FP64 seed for the fixed, factorial-normalized 2-D NS objective.

The nine polynomials are written directly in the canonical Q10 coefficient order.
This is a first-order transpose; no dense residual Jacobian is materialized.
"""

from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True)
class NSLoss:
    viscosity: float = 0.07
    gradient_weight: float = 0.2
    boundary_weight: float = 5.0

    def __post_init__(self):
        if any(not isfinite(v) or v < 0 for v in (
            self.viscosity, self.gradient_weight, self.boundary_weight
        )):
            raise ValueError("finite nonnegative objective coefficients required")


def residual_terms(viscosity=0.07):
    """Terms (coefficient, i, j), j=-1 linear, i=3*jet_index+field.

    Order: ru, rv, div, dx(ru), dx(rv), dx(div), dy(ru), dy(rv), dy(div).
    Mixed third derivatives recover with factor 2, pure thirds with factor 6.
    """
    v = viscosity
    return (
        ((1, 0, 6), (1, 1, 3), (1, 8, -1), (-2*v, 15, -1), (-2*v, 9, -1)),
        ((1, 0, 7), (1, 1, 4), (1, 5, -1), (-2*v, 16, -1), (-2*v, 10, -1)),
        ((1, 6, -1), (1, 4, -1)),
        ((1, 6, 6), (2, 0, 15), (1, 7, 3), (1, 1, 12), (2, 17, -1), (-6*v, 27, -1), (-2*v, 21, -1)),
        ((1, 6, 7), (2, 0, 16), (1, 7, 4), (1, 1, 13), (1, 14, -1), (-6*v, 28, -1), (-2*v, 22, -1)),
        ((2, 15, -1), (1, 13, -1)),
        ((1, 3, 6), (1, 0, 12), (1, 4, 3), (2, 1, 9), (1, 14, -1), (-2*v, 24, -1), (-6*v, 18, -1)),
        ((1, 3, 7), (1, 0, 13), (1, 4, 4), (2, 1, 10), (2, 11, -1), (-2*v, 25, -1), (-6*v, 19, -1)),
        ((1, 12, -1), (2, 10, -1)),
    )


def validate_weights(weights, shape, *, device):
    """Call during preparation, outside CUDA Graph capture and timed replay."""
    if weights.shape != shape or weights.dtype != torch.float64 or weights.device != device:
        raise ValueError("FP64 weights with the expected shape/device required")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("finite nonnegative globally normalized weights required")


def residuals(jets, loss=NSLoss()):
    if jets.ndim != 3 or jets.shape[1:] != (10, 3) or jets.dtype != torch.float64:
        raise ValueError("expected FP64 [N,10,3] jets of u,v,p")
    z = jets.reshape(jets.shape[0], 30)
    outputs = []
    for terms in residual_terms(loss.viscosity):
        value = torch.zeros_like(z[:, 0])
        for coefficient, i, j in terms:
            value = value + coefficient * z[:, i] * (z[:, j] if j >= 0 else 1.0)
        outputs.append(value)
    return torch.stack(outputs, dim=1)


def per_point_loss(jets, loss=NSLoss()):
    r = residuals(jets, loss)
    return 0.5 * (r[:, :3].square().sum(1) + loss.gradient_weight * r[:, 3:].square().sum(1))


def _explicit(jets, weights, loss):
    r = residuals(jets, loss)
    z = jets.reshape(jets.shape[0], 30)
    adjoint = [torch.zeros_like(z[:, 0]) for _ in range(30)]
    total = jets.new_zeros(())
    for k, terms in enumerate(residual_terms(loss.viscosity)):
        scale = 1.0 if k < 3 else loss.gradient_weight
        dr = weights * scale * r[:, k]
        total = total + 0.5 * (dr * r[:, k]).sum()
        for coefficient, i, j in terms:
            if j < 0:
                adjoint[i] = adjoint[i] + coefficient * dr
            else:
                adjoint[i] = adjoint[i] + coefficient * dr * z[:, j]
                adjoint[j] = adjoint[j] + coefficient * dr * z[:, i]
    return total, torch.stack(adjoint, dim=1).reshape_as(jets)


def explicit_seed(jets, weights, loss=NSLoss(), *, validate=True):
    if validate:
        validate_weights(weights, (len(jets),), device=jets.device)
    return _explicit(jets, weights, loss)


def compile_seed(loss=NSLoss(), **compile_options):
    """Compilation/setup belongs outside replay. Validate inputs before capture."""
    return torch.compile(lambda jets, weights: _explicit(jets, weights, loss),
                         fullgraph=True, dynamic=False, **compile_options)
