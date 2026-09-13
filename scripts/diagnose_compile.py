"""Isolate CUDA compiler parity failures without relaxing FP64 tolerance."""

import json
from pathlib import Path

import numpy as np
import torch

from flashns.euler import DOMAINS, load_points, load_surfaces, upstream_torch_checker
from flashns.provenance import run_metadata, write_report
from flashns.torch_spline import make_local_checker, vectorized_basis


def diff(a, b):
    d = (a - b).abs()
    return {"max_abs": float(d.max()), "dtype": str(a.dtype)}


def main():
    root = Path.cwd()
    torch.set_num_threads(4)
    meta, surfaces = load_surfaces(root / "sources/vendor/eulerRepo/splines")
    upstream = upstream_torch_checker(root, surfaces, meta, "cuda")
    points = load_points(root / "artifacts/euler_points.npz")
    points = np.concatenate([points[k] for k in DOMAINS])
    candidate = make_local_checker(upstream, meta, "cuda", mode="compiled")
    eager = make_local_checker(upstream, meta, "cuda", mode="vectorized")
    a, b = candidate(points), eager(points)
    errors = np.max(np.abs(a - b), axis=1)
    report = {
        "metadata": run_metadata(root),
        "worst_points": [
            {
                "point": points[i].tolist(),
                "compiled": a[i].tolist(),
                "eager": b[i].tolist(),
                "max_abs": float(errors[i]),
            }
            for i in np.argsort(errors)[-10:][::-1]
        ],
    }
    x = torch.tensor(points[:, 0], device="cuda", dtype=torch.float64)
    math_fn = lambda x: torch.stack([torch.cosh(x), torch.sinh(x), torch.tanh(x)])
    math_compiled = torch.compile(math_fn, fullgraph=True, dynamic=False)
    values, reference = math_compiled(x), math_fn(x)
    report["elementary_functions"] = {
        name: diff(values[i], reference[i])
        for i, name in enumerate(["cosh", "sinh", "tanh"])
    }
    surface = upstream.surfaces["Psi"]
    basis_fn = lambda x: vectorized_basis(
        x, surface.knots_x, surface.degree_x, surface.coeffs.shape[0]
    )
    basis_compiled = torch.compile(basis_fn, fullgraph=True, dynamic=False)
    aw, ai = basis_compiled(x)
    bw, bi = basis_fn(x)
    report["basis"] = {
        "weights": diff(aw, bw),
        "indices": diff(ai, bi),
        "weight_sum_max_error": float((aw.sum(1) - 1).abs().max()),
    }
    z = torch.tensor(points[:, 1], device="cuda", dtype=torch.float64)
    report["isolated_surface_jets"] = {}
    report["derivative_bases"] = {}
    for order in [1, 2]:
        trimmed = surface.knots_x[order:-order]
        fn = lambda x, t=trimmed, o=order: vectorized_basis(
            x, t, 9 - o, surface.coeffs.shape[0] - o
        )
        aw, ai = torch.compile(fn, fullgraph=True, dynamic=False)(x)
        bw, bi = fn(x)
        report["derivative_bases"][order] = {
            "weights": diff(aw, bw),
            "indices": diff(ai, bi),
        }
    for name in ["U", "Omega", "Psi"]:
        surface = upstream.surfaces[name]
        jet_fn = lambda x, z, s=surface, n=name: eager.tensor_jet(s, x, z, n == "Psi")
        compiled_jet = torch.compile(jet_fn, fullgraph=True, dynamic=False)
        expected, actual = jet_fn(x, z), compiled_jet(x, z)
        report["isolated_surface_jets"][name] = {
            k: diff(actual[k], expected[k]) for k in expected
        }
        for key in expected:
            i = int((actual[key] - expected[key]).abs().argmax())
            report["isolated_surface_jets"][name][key].update(
                {
                    "point": points[i].tolist(),
                    "actual": float(actual[key][i]),
                    "expected": float(expected[key][i]),
                }
            )
    write_report(root / "artifacts/gpu/compile_diagnostic.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "metadata"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
