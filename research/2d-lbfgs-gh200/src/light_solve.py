"""One complete 2D solve with torch's L-BFGS as installed or with the device-side two-loop (GH200 study).

    python light_solve.py --optimizer stock|light --backend gl8_native --interior-count 2048 --seed 862101 --output OUT.json
    python light_solve.py --build-only --output OUT.json

Runs inside the pinned 2D source tree's experiments/contracted_jets directory, like the H100
solve-scale wrapper (scale_solve.py): bench_gpu.evidence(), timed data preparation and the unchanged
bench_gpu.complete_solve (solve.py optimizer program, 500-block extension and acceptance). With
--optimizer light, torch.optim.LBFGS is replaced for this process by lbfgs_light.LightLBFGS
(mode 'graph', form 'fma'): the installed LBFGS.step with only the two-loop recursion changed.
The report records the SHA-256 of the installed torch.optim.lbfgs module and, for the light
optimizer, its counters. --build-only compiles the native extension and exits.
"""
import argparse
import hashlib
import inspect
import json
import traceback
from pathlib import Path
from time import perf_counter

import torch
import torch.optim.lbfgs

import bench_gpu
import problem
import solve as solve_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=("stock", "light"))
    parser.add_argument("--backend", choices=bench_gpu.BACKENDS, default="gl8_native")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--interior-count", type=int)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("require a new output")
    if args.build_only:
        from flashns.vnext.cuda import load_extension
        started = perf_counter()
        module = load_extension()
        args.output.write_text(json.dumps({"built": True, "seconds": perf_counter() - started,
                                           "source_hash": module.source_hash, "build_flags": module.build_flags}) + "\n")
        return
    if args.optimizer is None or args.seed is None or args.interior_count is None or args.interior_count < 1:
        parser.error("require --optimizer, --seed and a positive --interior-count")
    args.solve, args.eager, args.chunk_points = True, False, 512
    template = solve_module.protocol

    def protocol():
        configuration = template()
        configuration["interior_count"] = args.interior_count
        return configuration
    solve_module.protocol = protocol
    created = []
    lbfgs_module_sha256 = hashlib.sha256(inspect.getsource(torch.optim.lbfgs).encode()).hexdigest()
    if args.optimizer == "light":
        import lbfgs_light

        def factory(params, **kwargs):
            created.append(lbfgs_light.LightLBFGS(params, mode="graph", form="fma", **kwargs))
            return created[-1]
        torch.optim.LBFGS = factory
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {"status": "running", "arguments": vars(args) | {"output": str(args.output)},
              "environment": bench_gpu.evidence(), "progress": [],
              "wrapper": {"optimizer": args.optimizer, "interior_count_override": args.interior_count,
                          "frozen_template_interior_count": template()["interior_count"],
                          "torch_optim_lbfgs_module_sha256": lbfgs_module_sha256}}
    started = perf_counter()
    try:
        task = problem.Problem(interior_count=args.interior_count, device="cuda")
        initial = problem.parameters(args.seed)
        torch.cuda.synchronize()
        report["problem"] = task.meta
        report["parameter_initial_sha256"] = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
        report["data_preparation_seconds"] = perf_counter() - started
        bench_gpu.complete_solve(task, initial, args, report)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    if created:
        o = created[0]
        report["light"] = {"optimizers_created": len(created), "graph_replays": o.graph_replays, "eager_two_loops": o.eager_two_loops,
                           "ring_refills": o.ring_refills, "history_unchanged": o.history_unchanged,
                           "graphs_captured": len(o._ring["graphs"]) if o._ring else 0,
                           "rewritten_step_sha256": hashlib.sha256(o.rewritten_source.encode()).hexdigest(),
                           "lbfgs_light_sha256": hashlib.sha256(Path(lbfgs_light.__file__).read_bytes()).hexdigest()}
    report["execution_wall_seconds"] = perf_counter() - started
    bench_gpu.write_report(args.output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "execution_wall_seconds", "error", "light")}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
