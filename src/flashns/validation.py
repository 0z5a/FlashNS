import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .affine import PrecisionError, SineProfile, evaluate, integrate
from .provenance import run_metadata, verify_sources
from .reference import (
    differentiated_fields,
    field_error,
    high_precision_trajectory,
    reference_delta,
    trajectory_error,
)
from .symbolic import symbolic_report


def run_affine(root):
    root = Path(root)
    start = perf_counter()
    sources = verify_sources(root, names=["boussinesq_paper"])
    cases = json.loads((root / "cases/affine.json").read_text())
    contract = cases["acceptance"]
    stage = perf_counter()
    symbolic = symbolic_report()
    symbolic_seconds = perf_counter() - stage
    ode_seconds = reference_seconds = field_seconds = 0.0
    trajectories = []
    for case in cases["trajectories"]:
        args = [case["y0"], case["times"], case["D"], case["G"], case["lambda"]]
        stage = perf_counter()
        actual, solver = integrate(*args)
        ode_seconds += perf_counter() - stage
        stage = perf_counter()
        coarse = high_precision_trajectory(*args, substeps=64)
        fine = high_precision_trajectory(*args, substeps=128)
        delta = reference_delta(coarse, fine)
        error = trajectory_error(actual, fine)
        reference_seconds += perf_counter() - stage
        trajectories.append(
            {
                "name": case["name"],
                "solver": solver,
                "sample_count": len(case["times"]),
                "states": actual.tolist(),
                "mpmath_dps": 80,
                "reference_RK4_substeps_per_interval": [64, 128],
                "reference_refinement_delta": delta,
                "reference_tolerance_met": delta
                <= contract["reference_refinement"]["tolerance"],
                "fp64_vs_reference_error": error,
                "trajectory_tolerance_met": error
                <= contract["trajectory"]["tolerance"],
            }
        )
    fields = []
    D, G = [[0.2, -0.3], [0.4, -0.2]], [0.3, -1.2]
    stage = perf_counter()
    for case in cases["fields"]:
        try:
            evaluated = evaluate(
                case["points"],
                case["y"],
                case["lambda"],
                D=D,
                G=G,
                profile=SineProfile(tuple(case["coefficients"])),
            )
        except PrecisionError as exc:
            fields.append(
                {
                    "name": case["name"],
                    "outcome": "precision_rejected",
                    "reason": str(exc),
                    "expected_outcome_met": case.get("expected_outcome")
                    == "precision_rejected",
                }
            )
            continue
        errors = {}
        for i, point in enumerate(case["points"]):
            ref = differentiated_fields(
                point, case["y"], case["lambda"], case["coefficients"]
            )
            for name, value in ref.items():
                errors[name] = max(
                    errors.get(name, 0), field_error(evaluated[name][i], value)
                )
        normalized = {
            name: float(
                np.max(
                    np.abs(evaluated[f"{name}_residual_increment"])
                    / (1 + evaluated[f"{name}_term_scale"])
                )
            )
            for name in ["scalar", "vorticity"]
        }
        fields.append(
            {
                "name": case["name"],
                "point_count": len(case["points"]),
                "outcome": "evaluated",
                "expected_outcome_met": case.get("expected_outcome", "evaluated")
                == "evaluated",
                "field_errors": errors,
                "field_tolerance_met": max(errors.values())
                <= contract["field"]["tolerance"],
                "normalized_residual_increments": normalized,
                "residual_tolerance_met": max(normalized.values())
                <= contract["residual"]["tolerance"],
                "max_phase_roundoff_estimate": float(
                    np.max(evaluated["phase_roundoff_estimate"])
                ),
            }
        )
    field_seconds += perf_counter() - stage
    local_ok = (
        all(v["zero"] for v in symbolic["identities"].values())
        and all(
            v["reference_tolerance_met"] and v["trajectory_tolerance_met"]
            for v in trajectories
        )
        and all(
            v["expected_outcome_met"]
            and (
                v["outcome"] == "precision_rejected"
                or (v["field_tolerance_met"] and v["residual_tolerance_met"])
            )
            for v in fields
        )
    )
    elapsed = perf_counter() - start
    return {
        "schema_version": 1,
        "adapter": "boussinesq_exact_affine_wave",
        "metadata": run_metadata(root),
        "sources": sources,
        "contract": contract,
        "background": {"D": D, "G": G},
        "symbolic": symbolic,
        "trajectories": trajectories,
        "fields": fields,
        "acceptance": {
            "scope": "local lemma CPU regression at frozen inputs",
            "criteria_met": local_ok,
        },
        "status": {
            "reproduction": "local_checks_completed",
            "global_blowup": "not_attempted",
            "proof_replay": "not_run",
            "gpu_benchmark": "not_run",
            "interval_certificate": "not_run",
        },
        "timing_seconds": {
            "total_measured_workflow": elapsed,
            "fp64_ode": ode_seconds,
            "mpmath_ode_reference": reference_seconds,
            "field_and_derivative_checks": field_seconds,
            "symbolic_check": symbolic_seconds,
            "provenance_and_other": max(
                0,
                elapsed
                - ode_seconds
                - reference_seconds
                - field_seconds
                - symbolic_seconds,
            ),
        },
        "performance_scope": "one CPU regression workflow, no candidate search or acceleration claim",
    }
