"""Stable tanh auxiliary state and independent differentiable CPU references.

The CUDA API still supports only a manual first parameter VJP. The autograd
function here is a reference that permits nested derivatives for validation.
"""

from functools import lru_cache

import torch

from flashns.jet_spec import JetSpec


def stable_a1(z0):
    # where chooses the analytic right branch at zero; abs()'s zero subgradient
    # would give wrong higher derivatives in this independent AD reference.
    r = torch.exp(torch.where(z0 >= 0, -z0, z0))
    v = (2 * r) / (1 + r * r)
    return v * v


class _StableTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return value.tanh()

    @staticmethod
    def backward(ctx, adjoint):
        (value,) = ctx.saved_tensors
        return adjoint * stable_a1(value)


def stable_tanh(value):
    return _StableTanh.apply(value)


@lru_cache
def product_pairs(dim):
    indices = JetSpec(dim).coefficient_order
    lookup = {a: i for i, a in enumerate(indices)}
    return tuple(
        tuple(
            (j, lookup[tuple(x - y for x, y in zip(a, b))])
            for j, b in enumerate(indices)
            if all(y <= x for x, y in zip(a, b))
        )
        for a in indices
    )


def multiply(a, b, dim):
    return torch.stack(
        [
            sum((a[:, j] * b[:, k] for j, k in terms), torch.zeros_like(a[:, 0]))
            for terms in product_pairs(dim)
        ],
        1,
    )


def tanh_jet(z, dim):
    spec = JetSpec(dim)
    if z.ndim != 3 or z.shape[1] != spec.q:
        raise ValueError("expected [B,Q,C] full jets")
    t = stable_tanh(z[:, 0])
    aux = stable_a1(z[:, 0])
    delta = torch.cat((torch.zeros_like(z[:, :1]), z[:, 1:]), 1)
    p2 = multiply(delta, delta, dim)
    p3 = multiply(p2, delta, dim)
    h = (
        aux[:, None] * delta
        - (t * aux)[:, None] * p2
        + (aux * (t * t - 1 / 3))[:, None] * p3
    )
    return torch.cat((t[:, None], h[:, 1:]), 1), aux


def tanh_vjp(hidden, aux, adjoint, dim):
    spec = JetSpec(dim)
    g = -multiply(hidden, hidden, dim)
    g = torch.cat((aux[:, None], g[:, 1:]), 1)
    indices = spec.coefficient_order
    lookup = {a: i for i, a in enumerate(indices)}
    return torch.stack(
        [
            sum(
                (
                    adjoint[:, i] * g[:, lookup[tuple(x - y for x, y in zip(a, b))]]
                    for i, a in enumerate(indices)
                    if all(y <= x for x, y in zip(a, b))
                ),
                torch.zeros_like(hidden[:, 0]),
            )
            for b in indices
        ],
        1,
    )


def manifest(dim, source_hash):
    value = JetSpec(dim).manifest(generated_source_hash=source_hash)
    value["activation_aux_state"] = {
        "strategy": "stable_aux_a1",
        "activation_revision": 2,
        "stored": ["H[B,Q,C]", "a1[B,C]"],
        "a1_rule": "(2*exp(-abs(Z0))/(1+exp(-2*abs(Z0))))^2",
        "tail_accuracy": "declared by per-device high-precision tests; relative accuracy not promised after subnormal quantization",
    }
    return value
