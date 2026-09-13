import argparse
import cProfile
import json
from pathlib import Path

from .provenance import fetch_sources, verify_sources, write_report


def main():
    parser = argparse.ArgumentParser(
        description="FlashNS source-traceable reproduction"
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="FlashNS project directory"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    affine = sub.add_parser(
        "affine", help="run the pinned local affine-wave CPU checks"
    )
    affine.add_argument(
        "--output", type=Path, default=Path("artifacts/affine_cpu.json")
    )
    affine.add_argument("--profile", type=Path)
    euler = sub.add_parser(
        "euler",
        help="compare official frozen-spline residuals with an independent CPU baseline",
    )
    euler.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    euler.add_argument("--points-per-domain", type=int, default=1024)
    euler.add_argument(
        "--points-file", type=Path, help="read previously frozen NPZ points"
    )
    euler.add_argument(
        "--save-points", type=Path, default=Path("artifacts/euler_points.npz")
    )
    euler.add_argument("--batch-size", type=int, default=4096)
    euler.add_argument("--repeats", type=int, default=3)
    euler.add_argument("--threads", type=int, default=4)
    euler.add_argument(
        "--skip-legacy",
        action="store_true",
        help="skip previously measured legacy baselines; retain full local CPU comparison",
    )
    euler.add_argument(
        "--torch-mode", choices=["eager", "vectorized", "compiled"], default="eager"
    )
    euler.add_argument("--output", type=Path, default=Path("artifacts/euler_cpu.json"))
    euler.add_argument("--profile", type=Path)
    diagnose = sub.add_parser(
        "diagnose-euler",
        help="check backend disagreements against 80/100-digit spline differentiation",
    )
    diagnose.add_argument(
        "--points-file", type=Path, default=Path("artifacts/euler_points.npz")
    )
    diagnose.add_argument("--worst-per-domain", type=int, default=2)
    diagnose.add_argument(
        "--output", type=Path, default=Path("artifacts/euler_precision.json")
    )
    diagnose.add_argument("--profile", type=Path)
    sub.add_parser("verify-sources")
    sub.add_parser("fetch-sources")
    sub.add_parser("symbolic")
    args = parser.parse_args()
    if args.command == "verify-sources":
        report = verify_sources(args.root)
    elif args.command == "fetch-sources":
        report = fetch_sources(args.root)
    elif args.command == "symbolic":
        from .symbolic import symbolic_report

        report = symbolic_report()
    else:
        if args.command == "affine":
            from .validation import run_affine

            run = lambda: run_affine(args.root)
        elif args.command == "diagnose-euler":
            from .diagnostics import diagnose_euler
            from .euler import load_points

            points = load_points(args.points_file)
            run = lambda: diagnose_euler(
                args.root, points, worst_per_domain=args.worst_per_domain
            )
        else:
            import numpy as np

            from .euler import load_points, run_euler, sample_points

            if args.points_file:
                points = load_points(args.points_file)
            else:
                points = sample_points(args.points_per_domain)
                args.save_points.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.save_points, **points)
            run = lambda: run_euler(
                args.root,
                points,
                device=args.device,
                batch_size=args.batch_size,
                repeats=args.repeats,
                threads=args.threads,
                torch_mode=args.torch_mode,
                include_legacy=not args.skip_legacy,
            )
        if args.profile:
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            profiler = cProfile.Profile()
            report = profiler.runcall(run)
            profiler.dump_stats(args.profile)
            report["profiling"] = {
                "enabled": True,
                "timings_include_profiler_overhead": True,
            }
        else:
            report = run()
        if args.command == "euler":
            report["points_file"] = str(args.points_file or args.save_points)
        write_report(args.output, report)
        print(
            json.dumps(
                {
                    "report": str(args.output.resolve()),
                    "acceptance": report["acceptance"],
                    "timing_seconds": report["timing_seconds"],
                },
                indent=2,
            )
        )
        return 0 if report["acceptance"]["criteria_met"] else 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
