"""Userspace CPU/CUDA event profile on the frozen 4,096-point input."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from flashns.euler import (
    DOMAINS,
    load_points,
    load_surfaces,
    points_hash,
    scipy_residuals,
    upstream_torch_checker,
)
from flashns.provenance import run_metadata, verify_sources, write_report
from flashns.torch_spline import make_local_checker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["upstream", "upstream_compiled", "eager", "vectorized", "compiled"],
        required=True,
    )
    parser.add_argument(
        "--points-file", type=Path, default=Path("artifacts/euler_points.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling")
    sources = verify_sources(
        root, names=["euler_code", "euler_U", "euler_Omega", "euler_Psi", "euler_meta"]
    )
    points = load_points(args.points_file)
    combined = np.concatenate([points[name] for name in DOMAINS])
    if len(combined) > 16384:
        raise ValueError("Use at most 16384 points for this event-profile command")
    meta, surfaces = load_surfaces(root / "sources/vendor/eulerRepo/splines")
    baseline = scipy_residuals(surfaces, meta, combined, method="local_coefficients")
    upstream = upstream_torch_checker(root, surfaces, meta, "cuda")
    report = {
        "variant": args.variant,
        "metadata": run_metadata(root),
        "sources": sources,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "points_sha256": points_hash(points),
        "points": len(combined),
        "batch_size": 4096,
        "scope": "userspace event profiling and sampled FP64 numerical agreement; timings below exclude source loading",
    }
    if args.variant == "upstream":
        run = upstream
    elif args.variant == "upstream_compiled":
        compiled = torch.compile(upstream.raw_residual, fullgraph=True, dynamic=False)

        def run(samples):
            results = []
            for start in range(0, len(samples), 4096):
                b = torch.tensor(
                    samples[start : start + 4096], dtype=torch.float64, device="cuda"
                )
                x = b[:, 0:1].contiguous().detach().requires_grad_(True)
                z = b[:, 1:2].contiguous().detach().requires_grad_(True)
                results.append(torch.cat(compiled(x, z), dim=1).cpu().numpy())
            return np.concatenate(results)
    else:
        run = make_local_checker(upstream, meta, "cuda", mode=args.variant)
    times = []
    try:
        for _ in range(3):
            torch.cuda.synchronize()
            start = perf_counter()
            output = run(combined)
            torch.cuda.synchronize()
            times.append(perf_counter() - start)
        ratio = float(
            np.max(np.abs(output - baseline) / (2e-9 + 2e-9 * np.abs(baseline)))
        )
        report.update(
            {
                "execution": "completed",
                "unprofiled_seconds": times,
                "first_run_includes_compilation": args.variant
                in {"compiled", "upstream_compiled"},
                "max_tolerance_ratio": ratio,
                "numerical_agreement": bool(np.isfinite(ratio) and ratio <= 1),
            }
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            run(combined)
            torch.cuda.synchronize()
        events = []
        for event in prof.key_averages():
            events.append(
                {
                    "operator": event.key,
                    "count": event.count,
                    "self_cpu_us": event.self_cpu_time_total,
                    "self_device_us": getattr(event, "self_device_time_total", 0.0),
                }
            )
        report["top_cpu_events"] = sorted(
            events, key=lambda e: e["self_cpu_us"], reverse=True
        )[:25]
        report["top_device_events"] = sorted(
            events, key=lambda e: e["self_device_us"], reverse=True
        )[:25]
        report["all_event_counts"] = {e["operator"]: e["count"] for e in events}
    except Exception as exc:  # noqa: BLE001 -- preserve any compiler/profiler failure as a failed experiment
        report.update(
            {
                "execution": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:6000],
                "unprofiled_seconds": times,
            }
        )
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    write_report(args.output, report)
    print(
        {
            "variant": args.variant,
            "execution": report["execution"],
            "seconds": times,
            "numerical_agreement": report.get("numerical_agreement"),
            "output": str(args.output),
        }
    )
    return 0 if report.get("numerical_agreement") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
