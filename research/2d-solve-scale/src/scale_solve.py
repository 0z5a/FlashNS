"""Complete solves at a chosen interior point count for the H100 solve-scale study.

    python scale_solve.py --backend gl8_native --interior-count 16384 --seed 862101 --output OUT.json

bench_gpu.main() allows complete solves only at the frozen 2,048 points. This wrapper runs the same
path as bench_gpu.main for a solve -- one Torch thread, TF32 off, bench_gpu.evidence(), timed data
preparation, bench_gpu.complete_solve (unchanged solve.py optimizer program, 500-block extension and
acceptance) and the same report and exit status -- at another interior count. The only difference
from the frozen solve.py template is the recorded protocol field interior_count, which is set to the
actual count; the report says so.
"""
import argparse
import json
import traceback
from time import perf_counter

import torch

import bench_gpu
import problem
import solve as solve_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=bench_gpu.BACKENDS, required=True)
    parser.add_argument("--output", type=bench_gpu.Path, required=True)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--interior-count", type=int, required=True)
    parser.add_argument("--chunk-points", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists() or args.interior_count < 1 or args.chunk_points < 1:
        parser.error("require a new output and positive counts")
    args.solve = True
    template = solve_module.protocol

    def protocol():
        configuration = template()
        configuration["interior_count"] = args.interior_count
        return configuration
    solve_module.protocol = protocol
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {"status": "running", "arguments": vars(args) | {"output": str(args.output)},
              "environment": bench_gpu.evidence(), "progress": [],
              "scale_wrapper": {"file": "experiments/contracted_jets/scale_solve.py",
                                "difference_from_frozen_template": "protocol field interior_count set to the actual count",
                                "frozen_template_interior_count": template()["interior_count"]}}
    started = perf_counter()
    try:
        task = problem.Problem(interior_count=args.interior_count, device="cuda")
        initial = problem.parameters(args.seed)
        torch.cuda.synchronize()
        report["problem"] = task.meta
        report["data_preparation_seconds"] = perf_counter() - started
        bench_gpu.complete_solve(task, initial, args, report)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    report["execution_wall_seconds"] = perf_counter() - started
    bench_gpu.write_report(args.output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "execution_wall_seconds", "error")}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
