"""Resolve actual backend disagreements using independent precision checks."""

from time import perf_counter

import mpmath as mp
import numpy as np

from .euler import (
    DOMAINS,
    load_surfaces,
    points_hash,
    scipy_residuals,
    upstream_torch_checker,
)
from .provenance import run_metadata, verify_sources
from .spline_reference import high_precision_residual


def diagnose_euler(root, points, *, worst_per_domain=2):
    if worst_per_domain < 1:
        raise ValueError("at least one worst point per domain required")
    start = perf_counter()
    sources = verify_sources(
        root, names=["euler_code", "euler_U", "euler_Omega", "euler_Psi", "euler_meta"]
    )
    meta, surfaces = load_surfaces(root / "sources/vendor/eulerRepo/splines")
    checker = upstream_torch_checker(root, surfaces, meta, "cpu")
    selections = []
    for domain in DOMAINS:
        direct = scipy_residuals(surfaces, meta, points[domain])
        upstream = checker(points[domain])
        local = scipy_residuals(
            surfaces, meta, points[domain], method="local_coefficients"
        )
        # Include the worst disagreements plus fixed samples, so point
        # selection cannot favor the new local method alone.
        scores = np.max(np.abs(direct - upstream), axis=1)
        selected = sorted(set(np.argsort(scores)[-worst_per_domain:].tolist() + [0, 1]))
        for index in selected:
            selections.append(
                (
                    domain,
                    index,
                    points[domain][index],
                    {
                        "scipy_direct": direct[index],
                        "upstream_torch": upstream[index],
                        "local_coefficients": local[index],
                    },
                )
            )
    # Fixed axis/near-axis probes are separate from upstream's random plan.
    for point in [[0, 0], [0, 0.1], [1e-10, 0.1], [1e-7, 0.1]]:
        point = np.asarray(point, dtype=float)
        selections.append(
            (
                "axis_probes",
                None,
                point,
                {
                    "scipy_direct": scipy_residuals(surfaces, meta, point[None, :])[0],
                    "upstream_torch": checker(point[None, :])[0],
                    "local_coefficients": scipy_residuals(
                        surfaces, meta, point[None, :], method="local_coefficients"
                    )[0],
                },
            )
        )
    rows = []
    atol = rtol = 2e-9
    with mp.workdps(100):
        for domain, index, point, outputs in selections:
            ref80 = high_precision_residual(surfaces, meta, point, dps=80)
            ref100 = high_precision_residual(surfaces, meta, point, dps=100)
            refinement = max(
                float(abs(a - b) / (1 + abs(b))) for a, b in zip(ref80, ref100)
            )
            errors = {}
            for engine, values in outputs.items():
                differences = [
                    abs(mp.mpf(float(a)) - b) for a, b in zip(values, ref100)
                ]
                ratio = max(
                    float(delta / (mp.mpf(atol) + mp.mpf(rtol) * abs(ref)))
                    for delta, ref in zip(differences, ref100)
                )
                errors[engine] = {
                    "residuals": values.tolist(),
                    "max_abs_error": float(max(differences)),
                    "max_tolerance_ratio": ratio,
                    "tolerance_met": ratio <= 1,
                }
            rows.append(
                {
                    "domain": domain,
                    "index_in_frozen_domain": index,
                    "point": point.tolist(),
                    "reference_residuals_decimal": [mp.nstr(v, 40) for v in ref100],
                    "reference_80_vs_100_scaled_delta": refinement,
                    "reference_precision_stable": refinement <= 1e-30,
                    "engines": errors,
                }
            )
    summary = {
        engine: {
            "max_abs_error": max(
                row["engines"][engine]["max_abs_error"] for row in rows
            ),
            "failed_points": sum(
                not row["engines"][engine]["tolerance_met"] for row in rows
            ),
        }
        for engine in ["scipy_direct", "upstream_torch", "local_coefficients"]
    }
    return {
        "schema_version": 1,
        "adapter": "euler_spline_precision_diagnostic",
        "metadata": run_metadata(root),
        "sources": sources,
        "points_sha256": points_hash(points),
        "checked_points": len(rows),
        "selection": "worst direct-vs-upstream disagreements per domain, first two points per domain, four axis probes",
        "rows": rows,
        "summary": summary,
        "acceptance": {
            "scope": "local coefficient FP64 residuals at listed diagnostic points only",
            "norm": "max abs(fp64-mpmath100)/(atol+rtol*abs(mpmath100)) <= 1; reference80/reference100 scaled difference <= 1e-30",
            "atol": atol,
            "rtol": rtol,
            "units": "dimensionless residuals",
            "criteria_met": all(
                row["reference_precision_stable"]
                and row["engines"]["local_coefficients"]["tolerance_met"]
                for row in rows
            ),
        },
        "timing_seconds": {"total_measured_workflow": perf_counter() - start},
        "status": {
            "continuous_certificate": "not_run",
            "gpu_benchmark": "not_run",
            "proof_replay": "not_run",
        },
    }
