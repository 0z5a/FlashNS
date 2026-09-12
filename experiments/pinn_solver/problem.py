"""An independent, fixed Kovasznay boundary-value problem in physical coordinates."""

import hashlib
import math
import sys
from itertools import pairwise
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cuda_jet_h100"))
from common import residual_per_point, seed_coordinates

from flashns.jet_stable import stable_tanh

NU = 0.07
LAMBDA = -8 * math.pi**2 * NU / (1 + math.sqrt(1 + 16 * math.pi**2 * NU**2))
WIDTHS = (2, 64, 64, 3)


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def snapshot_sources(output):
    root = HERE.parents[1]
    paths = [
        *HERE.glob("*.py"),
        *[
            HERE.parent / "cuda_jet_h100" / name
            for name in ("common.py", "gpu.py", "scientific_backends.py", "blaslt.py")
        ],
        HERE.parent / "cuda_jet_hopper/hopper.py",
        HERE.parent / "cuda_jet/input/jet_reference.py",
        root / "src/flashns/jet_spec.py",
        root / "src/flashns/jet_stable.py",
    ]
    directory = Path(output).parent / "source_versions"
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in sorted(paths):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        saved = directory / (digest + "-" + path.name)
        if saved.exists() and saved.read_bytes() != content:
            raise RuntimeError("source snapshot collision")
        saved.write_bytes(content)
        hashes[str(path.relative_to(root))] = digest
    return hashes


def exact(x):
    exponential = torch.exp(LAMBDA * x[:, 0])
    angle = 2 * math.pi * x[:, 1]
    return torch.stack(
        (
            1 - exponential * torch.cos(angle),
            LAMBDA / (2 * math.pi) * exponential * torch.sin(angle),
            0.5 * (1 - exponential.square()),
        ),
        dim=-1,
    )


def boundary_points(count, *, seed=None):
    if seed is None:
        fraction = torch.linspace(0, 1, count, dtype=torch.float64)
    else:
        fraction = (
            torch.quasirandom.SobolEngine(1, scramble=True, seed=seed)
            .draw(count, dtype=torch.float64)
            .reshape(-1)
        )
    x, y = -0.5 + 1.5 * fraction, -0.5 + 2 * fraction
    return torch.cat(
        (
            torch.stack((x, torch.full_like(x, -0.5)), -1),
            torch.stack((x, torch.full_like(x, 1.5)), -1),
            torch.stack((torch.full_like(y, -0.5), y), -1),
            torch.stack((torch.ones_like(y), y), -1),
        )
    )


class Problem:
    def __init__(
        self,
        interior_count=2048,
        edge_count=128,
        validation_count=8192,
        data_seed=642701,
        device="cuda",
    ):
        interior = torch.quasirandom.SobolEngine(2, scramble=True, seed=data_seed).draw(
            interior_count, dtype=torch.float64
        )
        interior = interior * torch.tensor([1.5, 2.0]) + torch.tensor([-0.5, -0.5])
        boundary = boundary_points(edge_count, seed=data_seed + 1)
        x = torch.cat((interior, boundary))
        pde_weights = torch.zeros(len(x), dtype=torch.float64)
        pde_weights[:interior_count] = 1 / interior_count
        boundary_weights = torch.zeros_like(x[:, :1]).expand(-1, 3).clone()
        boundary_weights[interior_count:, :2] = 1 / len(boundary)
        boundary_weights[-edge_count:, 2] = 1 / edge_count
        validation = torch.quasirandom.SobolEngine(
            2, scramble=True, seed=data_seed + 100
        ).draw(validation_count, dtype=torch.float64)
        validation = validation * torch.tensor([1.5, 2.0]) + torch.tensor([-0.5, -0.5])
        validation_boundary = boundary_points(257)
        self.meta = {
            "workload_source_id": "pinn_kovasznay_solver",
            "relationship_to_openai_ns_formal": "independent manufactured steady NS boundary-value problem",
            "domain": [[-0.5, 1.0], [-0.5, 1.5]],
            "viscosity": NU,
            "lambda": LAMBDA,
            "network": list(WIDTHS),
            "hidden_activation": "stable tanh",
            "output_activation": "linear",
            "coordinates": "physical x,y, no nonlinear coordinate transform",
            "dtype": "float64",
            "interior_count": interior_count,
            "boundary_count": len(boundary),
            "pressure_outflow_count": edge_count,
            "validation_count": validation_count,
            "validation_boundary_count": len(validation_boundary),
            "data_seed": data_seed,
            "training_points_sha256": digest(x),
            "pde_weights_sha256": digest(pde_weights),
            "boundary_weights_sha256": digest(boundary_weights),
            "validation_points_sha256": digest(validation),
            "validation_boundary_sha256": digest(validation_boundary),
            "loss": "0.5 mean_interior(ru^2+rv^2+div^2+0.2*(|grad ru|^2+|grad rv|^2+|grad div|^2)) + 0.5*10*(mean_boundary(|uv-uv_exact|^2)+mean_outflow((p-p_exact)^2))",
            "boundary_condition": "Dirichlet u,v on every edge; exact p at x=1 fixes pressure gauge",
        }
        self.x, self.pde_weights = x.to(device), pde_weights.to(device)
        self.boundary_weights, self.target = (
            boundary_weights.to(device),
            exact(x).to(device),
        )
        self.validation = validation.to(device)
        self.validation_boundary = validation_boundary.to(device)
        self.interior_count, self.edge_count = interior_count, edge_count

    def boundary_loss(self, fields):
        return 5 * (self.boundary_weights * (fields - self.target).square()).sum()

    def jet_loss(self, jets):
        return (self.pde_weights * residual_per_point(jets)).sum() + self.boundary_loss(
            jets[:, 0]
        )


def parameters(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = [
        torch.randn(output, input, generator=generator, dtype=torch.float64)
        * (0.7 / math.sqrt(input))
        for input, output in pairwise(WIDTHS)
    ]
    biases = [
        torch.randn(output, generator=generator, dtype=torch.float64) * 0.1
        for output in WIDTHS[1:]
    ]
    return torch.cat([parameter.flatten() for parameter in weights + biases])


def views(flat):
    sizes = [input * output for input, output in pairwise(WIDTHS)] + list(WIDTHS[1:])
    parts = flat.split(sizes)
    weights = [
        part.view(output, input)
        for part, (input, output) in zip(parts[:3], pairwise(WIDTHS))
    ]
    return weights, list(parts[3:])


def fields(x, weights, biases):
    value = x
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        value = value @ weight.T + bias
        if index + 1 < len(weights):
            value = stable_tanh(value)
    return value


def jets(x, weights, biases, native):
    value = seed_coordinates(x)
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        value = value @ weight.T
        value[:, 0] += bias
        if index + 1 < len(weights):
            value, _ = native.forward(value, 2)
    return value


def residuals_from_jets(value):
    # Canonical order: 00,01,10,02,11,20,03,12,21,30; factorial-normalized.
    u, v = value[:, 0, 0], value[:, 0, 1]
    ru = (
        u * value[:, 2, 0]
        + v * value[:, 1, 0]
        + value[:, 2, 2]
        - NU * 2 * (value[:, 5, 0] + value[:, 3, 0])
    )
    rv = (
        u * value[:, 2, 1]
        + v * value[:, 1, 1]
        + value[:, 1, 2]
        - NU * 2 * (value[:, 5, 1] + value[:, 3, 1])
    )
    div = value[:, 2, 0] + value[:, 1, 1]
    return torch.stack((ru, rv, div), dim=-1)


def evaluate(problem, flat, native):
    weights, biases = views(flat)
    with torch.no_grad():
        prediction = jets(problem.validation, weights, biases, native)
        exact_fields = exact(problem.validation)
        error = prediction[:, 0] - exact_fields
        relative = (error.square().sum(0) / exact_fields.square().sum(0)).sqrt()
        residual = residuals_from_jets(prediction)
        boundary = fields(problem.validation_boundary, weights, biases) - exact(
            problem.validation_boundary
        )
        metric = {
            "pde_rms": residual.square().sum(-1).mean().sqrt(),
            "boundary_uv_rms": boundary[:, :2].square().mean().sqrt(),
            "outflow_p_rms": boundary[-257:, 2].square().mean().sqrt(),
            "relative_l2_u": relative[0],
            "relative_l2_v": relative[1],
            "relative_l2_p": relative[2],
        }
        return {name: float(value) for name, value in metric.items()}


def symbolic_check():
    import sympy as sp

    x, y = sp.symbols("x y", real=True)
    nu = sp.Rational(7, 100)
    lam = 1 / (2 * nu) - sp.sqrt(1 / (4 * nu**2) + 4 * sp.pi**2)
    u = 1 - sp.exp(lam * x) * sp.cos(2 * sp.pi * y)
    v = lam / (2 * sp.pi) * sp.exp(lam * x) * sp.sin(2 * sp.pi * y)
    p = (1 - sp.exp(2 * lam * x)) / 2
    residuals = [
        u * sp.diff(u, x)
        + v * sp.diff(u, y)
        + sp.diff(p, x)
        - nu * (sp.diff(u, x, 2) + sp.diff(u, y, 2)),
        u * sp.diff(v, x)
        + v * sp.diff(v, y)
        + sp.diff(p, y)
        - nu * (sp.diff(v, x, 2) + sp.diff(v, y, 2)),
        sp.diff(u, x) + sp.diff(v, y),
    ]
    results = [str(sp.simplify(value)) for value in residuals]
    if results != ["0", "0", "0"]:
        raise AssertionError(results)
    return {
        "sympy": sp.__version__,
        "residuals": results,
        "viscosity_exact": "7/100",
        "passed": True,
    }


def independent_metrics(problem, flat):
    """Final acceptance through scalar nested AD on an additional point set."""
    count = problem.meta["validation_count"]
    x = torch.quasirandom.SobolEngine(
        2, scramble=True, seed=problem.meta["data_seed"] + 1000
    ).draw(count, dtype=torch.float64)
    x = (
        (x * torch.tensor([1.5, 2.0]) + torch.tensor([-0.5, -0.5]))
        .to(problem.x.device)
        .requires_grad_()
    )
    weights, biases = views(flat.detach())
    prediction = fields(x, weights, biases)

    def grad(value):
        return torch.autograd.grad(
            value.sum(), x, create_graph=True, retain_graph=True
        )[0]

    u, v, p = prediction.unbind(-1)
    gu, gv, gp = grad(u), grad(v), grad(p)
    ru = (
        u * gu[:, 0]
        + v * gu[:, 1]
        + gp[:, 0]
        - NU * (grad(gu[:, 0])[:, 0] + grad(gu[:, 1])[:, 1])
    )
    rv = (
        u * gv[:, 0]
        + v * gv[:, 1]
        + gp[:, 1]
        - NU * (grad(gv[:, 0])[:, 0] + grad(gv[:, 1])[:, 1])
    )
    divergence = gu[:, 0] + gv[:, 1]
    with torch.no_grad():
        target = exact(x)
        relative = (
            (prediction - target).square().sum(0) / target.square().sum(0)
        ).sqrt()
        boundary = fields(problem.validation_boundary, weights, biases) - exact(
            problem.validation_boundary
        )
        return {
            "pde_rms": float(
                (ru.square() + rv.square() + divergence.square()).mean().sqrt()
            ),
            "boundary_uv_rms": float(boundary[:, :2].square().mean().sqrt()),
            "outflow_p_rms": float(boundary[-257:, 2].square().mean().sqrt()),
            "relative_l2_u": float(relative[0]),
            "relative_l2_v": float(relative[1]),
            "relative_l2_p": float(relative[2]),
        }
