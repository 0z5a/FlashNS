"""Paired SciPy/NumPy and PyTorch A/B on the original frozen Euler point set."""

import argparse
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import torch
from benchmark import telemetry

from flashns.euler import (
    DOMAINS,
    load_points,
    load_surfaces,
    points_hash,
    sampled_norms,
    scipy_residuals,
    upstream_torch_checker,
)
from flashns.provenance import run_metadata, verify_sources
from flashns.torch_spline import make_local_checker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN_4M = "101bc2adbc7bb58624f0685f7a5efdeb7fa172ffe9857b43f2f6b47cc59a8110"


def check(actual, reference):
    delta = np.abs(actual - reference)
    ratio = float(np.max(delta / (2e-9 + 2e-9 * np.abs(reference))))
    return {
        "max_absolute_error": float(np.max(delta)),
        "max_tolerance_ratio": ratio,
        "passed": bool(np.isfinite(ratio) and ratio <= 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--require-original-4m", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    workflow_start = perf_counter()
    points = load_points(args.points_file)
    digest = points_hash(points)
    if args.require_original_4m and digest != FROZEN_4M:
        raise RuntimeError(
            "the original 4M-point hash is required; regeneration is not equivalent"
        )
    started = perf_counter()
    sources = verify_sources(
        ROOT,
        names=[
            "euler_paper",
            "euler_code",
            "euler_U",
            "euler_Omega",
            "euler_Psi",
            "euler_meta",
        ],
    )
    checked = perf_counter()
    meta, surfaces = load_surfaces(ROOT / "sources/vendor/eulerRepo/splines")
    loaded = perf_counter()
    upstream = upstream_torch_checker(ROOT, surfaces, meta, "cuda")
    eager = make_local_checker(upstream, meta, "cuda", mode="eager")
    vectorized = make_local_checker(upstream, meta, "cuda", mode="vectorized")
    torch.cuda.synchronize()
    transferred = perf_counter()
    backend_functions = {
        "SciPy_direct_CPU": lambda sample, size: scipy_residuals(
            surfaces, meta, sample, batch_size=size
        ),
        "SciPy_NumPy_stable_CPU": lambda sample, size: scipy_residuals(
            surfaces, meta, sample, method="local_coefficients", batch_size=size
        ),
        "Torch_upstream_GPU": upstream,
        "Torch_stable_eager_GPU": eager,
        "Torch_stable_vectorized_GPU": vectorized,
    }
    warmup_started = perf_counter()
    for name, function in backend_functions.items():
        for domain in DOMAINS:
            function(
                points[domain][: min(args.batch_size, len(points[domain]))],
                args.batch_size,
            )
        torch.cuda.synchronize()
        print({"warmup": name}, flush=True)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "adapter": "official_euler_frozen_splines",
        "metadata": run_metadata(ROOT),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": sources,
        "points_sha256": digest,
        "point_counts": {name: len(value) for name, value in points.items()},
        "same_points_as_previous_A5000_4m": digest == FROZEN_4M,
        "points_file": str(args.points_file),
        "batch_size": args.batch_size,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "uuid": str(torch.cuda.get_device_properties().uuid),
        "torch_threads": torch.get_num_threads(),
        "telemetry_before": telemetry(),
        "precision": "FP64",
        "profile_parameters": meta,
        "timing_scope": "complete sampled residual evaluation over all domains; GPU paths include CPU input transfer and returning all outputs to CPU; separate per-repeat full-output comparison",
        "reference_scope": "local control-coefficient derivative CPU implementation; separate 80/100-digit diagnostics validate selected values; no proof or interval certificate",
        "initialization_seconds": {
            "point_load_and_hash": started - workflow_start,
            "source_hashes": checked - started,
            "JSON_and_SciPy_construction": loaded - checked,
            "adapter_and_GPU_transfer": transferred - loaded,
            "all_backend_small_warmups": perf_counter() - warmup_started,
        },
        "resident_spline_bytes": sum(
            surface.c.nbytes + sum(k.nbytes for k in surface.t)
            for surface in surfaces.values()
        ),
        "atol": 2e-9,
        "rtol": 2e-9,
        "paired_observations": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    names = list(backend_functions)
    for repeat in range(args.repeats):
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        if repeat % 2:
            order.reverse()
        row, outputs = {"repeat": repeat, "order": order, "measurements": {}}, {}
        report["in_progress_observation"] = row
        for name in order:
            row["running_backend"] = name
            save()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            outputs[name] = {
                domain: backend_functions[name](points[domain], args.batch_size)
                for domain in DOMAINS
            }
            torch.cuda.synchronize()
            row["measurements"][name] = {
                "synchronized_wall_seconds": perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                if name.endswith("GPU")
                else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
                if name.endswith("GPU")
                else None,
            }
            del row["running_backend"]
            save()
            print(
                {
                    "repeat": repeat,
                    "backend": name,
                    "seconds": row["measurements"][name]["synchronized_wall_seconds"],
                },
                flush=True,
            )
        started = perf_counter()
        reference = outputs["SciPy_NumPy_stable_CPU"]
        row["comparison_to_stable_CPU"] = {
            name: {
                domain: check(values[domain], reference[domain]) for domain in DOMAINS
            }
            for name, values in outputs.items()
        }
        row["full_output_comparison_seconds"] = perf_counter() - started
        if not repeat:
            report["sampled_residual_statistics"] = {
                name: {domain: sampled_norms(values[domain]) for domain in DOMAINS}
                for name, values in outputs.items()
            }
        report["paired_observations"].append(row)
        del report["in_progress_observation"]
        save()
        del outputs, reference
    report["median_seconds"] = {
        name: statistics.median(
            r["measurements"][name]["synchronized_wall_seconds"]
            for r in report["paired_observations"]
        )
        for name in names
    }
    report["passed_declared_contract"] = {
        name: all(
            check["passed"]
            for r in report["paired_observations"]
            for check in r["comparison_to_stable_CPU"][name].values()
        )
        for name in names
    }
    report["paired_speedup_over_stable_vectorized_median"] = {
        name: statistics.median(
            r["measurements"][name]["synchronized_wall_seconds"]
            / r["measurements"]["Torch_stable_vectorized_GPU"][
                "synchronized_wall_seconds"
            ]
            for r in report["paired_observations"]
        )
        if report["passed_declared_contract"][name]
        and report["passed_declared_contract"]["Torch_stable_vectorized_GPU"]
        else None
        for name in names
    }
    report["full_repeated_AB_workflow_seconds"] = perf_counter() - workflow_start
    report["telemetry_after"] = telemetry()
    report["completed"] = True
    report["science_scope"] = (
        "frozen profile sampled evaluation only; no training, proof replay, continuous-domain verification or convergence claim"
    )
    save()
    print(
        {
            "median_seconds": report["median_seconds"],
            "passed": report["passed_declared_contract"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
