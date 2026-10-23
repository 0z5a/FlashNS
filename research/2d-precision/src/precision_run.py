"""Replay or complete solve of the 2D objective in FP64, FP32 or FP32 with TF32 matrix products.

    python precision_run.py --kind replay|solve|build --backend B --precision fp64|fp32|tf32 \
        --interior-count N --seed S --output OUT.json

Runs inside the pinned 2D source tree with src/fp32-dtype.patch applied (experiments/contracted_jets).
The problem data and formal initializations are built in FP64, as everywhere else, and then cast.

- replay: bench_gpu.replay_benchmark, unchanged for FP64. For FP32/TF32, `reduced_replay` below is
  that function with only its correctness block replaced: the loss and gradient are compared with
  the FP64 dense-Q10 oracle on the same parameters cast to FP64. The errors are reported, and the
  process fails only on a non-finite value or a relative L2 gradient error of at least 0.1.
- solve: bench_gpu.complete_solve (solve.py program, 500-block extension) with the loss/gradient
  engine, parameters and optimizer state in the arm's dtype. Every stop check and the final
  acceptance (solve.evaluate, solve.independent_metrics) evaluate the parameters cast to FP64 on the
  FP64 problem, with the unchanged thresholds.
- build: compile the FP64 and FP32 native extensions once, before the timed queue.
"""
import argparse
import json
import statistics
import traceback
from time import perf_counter

import torch

import bench_gpu
import problem
import solve as solve_module
from flashns.pinn_packed import PackedPINN
from gpu_engine import GraphEngine, flatten

FLOAT_FIELDS = ("x", "pde_weights", "boundary_weights", "target", "validation", "validation_boundary")
PRECISIONS = {"fp64": (torch.float64, False), "fp32": (torch.float32, False), "tf32": (torch.float32, True)}
SANITY_GRADIENT_RELATIVE_L2 = 0.1


def set_precision(tf32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def cast(task, dtype):
    for name in FLOAT_FIELDS:
        setattr(task, name, getattr(task, name).to(dtype))
    return task


def reduced_replay(task, task64, initial, args, report):
    """bench_gpu.replay_benchmark with the FP64 oracle comparison in place of its FP64 assert."""
    torch.cuda.reset_peak_memory_stats()
    engine = GraphEngine(task, initial, args.backend, compiled=True, chunk_points=512)
    report["engine_setup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["engine_setup_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["engine_ready_allocated_bytes"] = torch.cuda.memory_allocated()
    report["engine_ready_reserved_bytes"] = torch.cuda.memory_reserved()
    ni = task64.interior_count
    oracle = PackedPINN(task64.x, ni, task64.pde_weights[:ni], task64.boundary_weights[ni:], task64.target[ni:], layout="dense")
    expected_loss, expected_gradients = oracle(*problem.views(engine.flat.detach().double()))
    expected_gradient = flatten(expected_gradients)
    loss, gradient = engine.replay()
    loss, gradient = loss.double(), gradient.double()
    errors = {"loss_relative_error": float((loss - expected_loss).abs() / expected_loss.abs()),
              "gradient_relative_l2": float((gradient - expected_gradient).norm() / expected_gradient.norm()),
              "gradient_max_abs_error": float((gradient - expected_gradient).abs().max()),
              "gradient_max_over_max_abs": float((gradient - expected_gradient).abs().max() / expected_gradient.abs().max())}
    finite = bool(torch.isfinite(loss)) and bool(torch.isfinite(gradient).all())
    report["correctness_against_fp64_q10"] = {"passed": finite and errors["gradient_relative_l2"] < SANITY_GRADIENT_RELATIVE_L2,
                                              "sanity_gate_gradient_relative_l2": SANITY_GRADIENT_RELATIVE_L2, **errors}
    if not report["correctness_against_fp64_q10"]["passed"]:
        raise AssertionError(f"reduced-precision replay failed the sanity gate: {errors}")
    del oracle, expected_loss, expected_gradients, expected_gradient
    report["engine"] = engine.metadata
    for _ in range(30):
        engine.replay()
    torch.cuda.synchronize()
    report["device_before"] = bench_gpu.device_state()
    warm_started, warm_replays = perf_counter(), 0
    while perf_counter() - warm_started < args.warmup_seconds:
        for _ in range(100):
            engine.replay()
        torch.cuda.synchronize()
        warm_replays += 100
    report["sustained_warmup"] = {"requested_seconds": args.warmup_seconds, "wall_seconds": perf_counter() - warm_started,
                                  "replays": warm_replays, "included_in_measurement": False}
    report["device_warmed"] = bench_gpu.device_state()
    for _ in range(100):
        engine.replay()
    torch.cuda.synchronize()
    report["dynamo_before"] = bench_gpu.counters()
    torch.cuda.reset_peak_memory_stats()
    report["steady_allocated_bytes"] = torch.cuda.memory_allocated()
    report["steady_reserved_bytes"] = torch.cuda.memory_reserved()
    samples = []
    for _ in range(args.samples):
        for _ in range(10):
            engine.replay()
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = perf_counter()
        begin.record()
        for _ in range(args.replays):
            engine.replay()
        end.record()
        end.synchronize()
        samples.append({"gpu_ms_per_gradient": begin.elapsed_time(end) / args.replays,
                        "wall_ms_per_gradient": (perf_counter() - wall) * 1000 / args.replays})
    report["samples"] = samples
    report["median_ms"] = statistics.median(row["gpu_ms_per_gradient"] for row in samples)
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["device_after"] = bench_gpu.device_state()
    report["dynamo_after"] = bench_gpu.counters()
    if report["dynamo_before"] != report["dynamo_after"]:
        raise AssertionError("compilation/capture counters changed in measurement window")
    report["timing_scope"] = "full loss and all parameter gradients; excludes optimizer, stopping validation, setup, compilation and capture"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("replay", "solve", "build"), required=True)
    parser.add_argument("--backend", choices=bench_gpu.BACKENDS, default="gl8_native")
    parser.add_argument("--precision", choices=tuple(PRECISIONS), default="fp64")
    parser.add_argument("--interior-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=75101)
    parser.add_argument("--output", type=bench_gpu.Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.interior_count < 1:
        parser.error("require a new output and a positive point count")
    args.eager, args.chunk_points, args.solve = False, 512, args.kind == "solve"
    args.samples, args.replays, args.warmup_seconds, args.profile = 7, 100, 2.0, False
    dtype, tf32 = PRECISIONS[args.precision]
    torch.set_num_threads(1)
    report = {"status": "running", "arguments": vars(args) | {"output": str(args.output)},
              "environment": bench_gpu.evidence(), "progress": [],
              "precision": {"name": args.precision, "dtype": str(dtype), **set_precision(tf32)},
              "precision_wrapper": {"file": "experiments/contracted_jets/precision_run.py",
                                    "acceptance_precision": "float64", "data_cast_fields": list(FLOAT_FIELDS)}}
    started = perf_counter()
    try:
        if args.kind == "build":
            from flashns.vnext.cuda import load_extension, binary_evidence
            report["extensions"] = {str(d): binary_evidence(load_extension(d)) for d in (torch.float64, torch.float32)}
        else:
            task64 = problem.Problem(interior_count=args.interior_count, device="cuda")
            report["problem"] = task64.meta
            initial64 = problem.parameters(args.seed)
            report["initial_parameters_sha256"] = problem.digest(initial64)
            if dtype == torch.float64:
                task, initial = task64, initial64
            else:
                task, initial = cast(problem.Problem(interior_count=args.interior_count, device="cuda"), dtype), initial64.to(dtype)
            torch.cuda.synchronize()
            report["data_preparation_seconds"] = perf_counter() - started
            if args.kind == "replay":
                if dtype == torch.float64:
                    bench_gpu.replay_benchmark(task, initial, args, report)
                else:
                    reduced_replay(task, task64, initial, args, report)
            else:
                template = solve_module.protocol

                def protocol():
                    configuration = template()
                    configuration["interior_count"] = args.interior_count
                    return configuration
                solve_module.protocol = protocol
                if dtype != torch.float64:
                    evaluate, independent = solve_module.evaluate, solve_module.independent_metrics
                    solve_module.evaluate = lambda _, flat, evaluator: evaluate(task64, flat.detach().double(), evaluator)
                    solve_module.independent_metrics = lambda _, flat: independent(task64, flat.detach().double())
                report["scale_and_precision"] = {"frozen_template_interior_count": template()["interior_count"],
                                                 "stop_checks_and_acceptance": "FP64 problem, parameters cast to FP64"}
                bench_gpu.complete_solve(task, initial, args, report)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    report["execution_wall_seconds"] = perf_counter() - started
    bench_gpu.write_report(args.output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "median_ms", "execution_wall_seconds", "error")}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
