"""A fixed weighted steady-NS target shared by isolated and distributed runs."""

import hashlib
import math
import sys
from itertools import pairwise
from pathlib import Path

import torch

from flashns.distributed_contract import global_normalization, shard_bounds
from flashns.jet_spec import JetSpec
from flashns.jet_stable import stable_tanh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "cuda_jet/input"))
import jet_reference as legacy_reference


def source_hashes():
    paths = [
        *HERE.glob("*.py"),
        *HERE.glob("*.cu"),
        *HERE.glob("*.cuh"),
        ROOT / "src/flashns/jet_stable.py",
        ROOT / "src/flashns/jet_spec.py",
        ROOT / "src/flashns/distributed_contract.py",
        HERE.parent / "cuda_jet/input/jet_reference.py",
        HERE.parent / "cuda_jet/input/generate_jet_header.py",
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def make_case(batch, channels=64, seed=7510, weighted=False, coordinate_scale=0.4):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (
        torch.randn(batch, 2, generator=generator, dtype=torch.float64)
        * coordinate_scale
    )
    quadrature = (
        0.5 + torch.rand(batch, generator=generator, dtype=torch.float64)
        if weighted
        else torch.ones(batch, dtype=torch.float64)
    )
    widths = [2, channels, channels, 3]
    weights = [
        torch.randn(o, i, generator=generator, dtype=torch.float64)
        * (0.7 / math.sqrt(i))
        for i, o in pairwise(widths)
    ]
    biases = [
        torch.randn(o, generator=generator, dtype=torch.float64) * 0.1
        for o in widths[1:]
    ]
    meta = {
        "global_batch": batch,
        "widths": widths,
        "seed": seed,
        "weighted": weighted,
        "coordinate_scale": coordinate_scale,
        "global_points_sha256": tensor_hash(x),
        "global_weights_sha256": tensor_hash(quadrature),
        "global_denominator": global_normalization(quadrature.tolist()),
        "parameter_init_sha256": tensor_hash(
            torch.cat([p.flatten() for p in weights + biases])
        ),
        "precision_policy": "stable_aux_a1",
        "workload_source_id": "pinn_jet_local",
    }
    return x, quadrature, weights, biases, meta


def local_case(case, world, rank, device):
    x, q, weights, biases, meta = case
    start, stop = shard_bounds(x.shape[0], world, rank)
    return (
        x[start:stop].to(device),
        q[start:stop].to(device),
        [p.to(device) for p in weights],
        [p.to(device) for p in biases],
        {**meta, "point_id_range": [start, stop], "local_batch": stop - start},
    )


def seed_coordinates(x):
    return legacy_reference.seed_coordinates(x, legacy_reference.JetPlan.dense(2, 3))


def residual_per_point(jet, nu=0.07, gradient_weight=0.2):
    if jet.ndim != 3 or jet.shape[1:] != (10, 3):
        raise ValueError("target requires 2-D third-order jets of u,v,p")
    indices = JetSpec(2).coefficient_order
    lookup = {a: i for i, a in enumerate(indices)}

    def value(field, a):
        return jet[:, lookup[a], field]

    def derivative(field, a, axis, count=1):
        target = list(a)
        target[axis] += count
        return (math.factorial(target[axis]) / math.factorial(a[axis])) * value(
            field, tuple(target)
        )

    terms = []
    for alpha in ((0, 0), (1, 0), (0, 1)):
        adv_u = torch.zeros_like(jet[:, 0, 0])
        adv_v = torch.zeros_like(adv_u)
        for beta in indices:
            if not all(b <= a for a, b in zip(alpha, beta)):
                continue
            gamma = tuple(a - b for a, b in zip(alpha, beta))
            adv_u = (
                adv_u
                + value(0, beta) * derivative(0, gamma, 0)
                + value(1, beta) * derivative(0, gamma, 1)
            )
            adv_v = (
                adv_v
                + value(0, beta) * derivative(1, gamma, 0)
                + value(1, beta) * derivative(1, gamma, 1)
            )
        ru = (
            adv_u
            + derivative(2, alpha, 0)
            - nu * (derivative(0, alpha, 0, 2) + derivative(0, alpha, 1, 2))
        )
        rv = (
            adv_v
            + derivative(2, alpha, 1)
            - nu * (derivative(1, alpha, 0, 2) + derivative(1, alpha, 1, 2))
        )
        div = derivative(0, alpha, 0) + derivative(1, alpha, 1)
        weight = 1.0 if alpha == (0, 0) else gradient_weight
        terms.append(weight * (ru.square() + rv.square() + div.square()))
    return 0.5 * torch.stack(terms).sum(0)


def forward(x, weights, biases, native):
    hidden = seed_coordinates(x)
    checkpoints = [hidden]
    auxiliaries = [None]
    for index, (w, b) in enumerate(zip(weights, biases)):
        z = hidden @ w.T
        z[:, 0] += b
        if index + 1 < len(weights):
            hidden, aux = native.forward(z, 2)
        else:
            hidden, aux = z, None
        checkpoints.append(hidden)
        auxiliaries.append(aux)
    return hidden, checkpoints, auxiliaries


def step(x, quadrature, denominator, weights, biases, native, backend="B1"):
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("positive global denominator required")
    if x.shape[0] == 0:
        return x.new_zeros(()), [torch.zeros_like(p) for p in weights + biases]
    with torch.no_grad():
        jets, checkpoints, auxiliaries = forward(x, weights, biases, native)
    leaf = jets.detach().requires_grad_()
    loss = (quadrature * residual_per_point(leaf)).sum() / denominator
    (d,) = torch.autograd.grad(loss, leaf)
    dws, dbs = [None] * len(weights), [None] * len(weights)
    with torch.no_grad():
        for index in reversed(range(len(weights))):
            dws[index] = d.flatten(0, 1).T @ checkpoints[index].flatten(0, 1)
            dbs[index] = d[:, 0].sum(0)
            if index:
                d = native.dgrad(
                    d,
                    weights[index],
                    checkpoints[index],
                    auxiliaries[index],
                    2,
                    backend,
                )
    return loss.detach(), dws + dbs


def nested_reference(x, quadrature, denominator, weights, biases):
    x = x.detach().requires_grad_()
    parameters = [p.detach().requires_grad_() for p in weights + biases]
    ws, bs = parameters[: len(weights)], parameters[len(weights) :]
    h = x
    for index, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w.T + b
        if index + 1 < len(ws):
            h = stable_tanh(h)
    u, v, p = h.unbind(-1)

    def grad(value):
        return torch.autograd.grad(
            value.sum(), x, create_graph=True, retain_graph=True
        )[0]

    gu, gv, gp = grad(u), grad(v), grad(p)
    lapu = grad(gu[:, 0])[:, 0] + grad(gu[:, 1])[:, 1]
    lapv = grad(gv[:, 0])[:, 0] + grad(gv[:, 1])[:, 1]
    ru = u * gu[:, 0] + v * gu[:, 1] + gp[:, 0] - 0.07 * lapu
    rv = u * gv[:, 0] + v * gv[:, 1] + gp[:, 1] - 0.07 * lapv
    div = gu[:, 0] + gv[:, 1]
    per_point = 0.5 * (
        ru.square()
        + rv.square()
        + div.square()
        + 0.2
        * (
            grad(ru).square().sum(-1)
            + grad(rv).square().sum(-1)
            + grad(div).square().sum(-1)
        )
    )
    loss = (quadrature * per_point).sum() / denominator
    gradients = torch.autograd.grad(loss, parameters)
    return loss.detach(), [g.detach() for g in gradients]


def comparison(actual, expected, atol=2e-12, rtol=2e-11):
    a, b = actual.detach(), expected.detach()
    error = (a - b).abs()
    ratio = float((error / (atol + rtol * b.abs())).max())
    result = {
        "max_abs": float(error.max()),
        "max_tolerance_ratio": ratio,
        "atol": atol,
        "rtol": rtol,
        "passed": math.isfinite(ratio) and ratio <= 1,
    }
    if not result["passed"]:
        raise AssertionError(result)
    return result
