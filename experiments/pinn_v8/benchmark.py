"""Matched layout/seed Graph measurements and a separate diagnostic trace."""

import argparse
import json
import random
import statistics
from pathlib import Path
from time import perf_counter

import torch

from runtime import (Engine, NativeCUDA, Problem, compare, device_metadata, ensure_preflight,
                     make_step, parameters, snapshot, views)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interior", type=int, default=2048)
    parser.add_argument("--edge", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed-modes", nargs="+", choices=("autograd", "explicit", "compiled", "cuda"), default=("autograd", "cuda"))
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 3:
        raise RuntimeError("new output and at least three paired rounds required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    native = NativeCUDA(args.build)
    preflight = ensure_preflight(args.preflight, native)
    if preflight.get("quick"):
        raise RuntimeError("full preflight required for performance measurements")
    problem = Problem(interior_count=args.interior, edge_count=args.edge)
    initial = parameters(361902)
    report = {"completed": False, "device": device_metadata(), "problem": problem.meta,
              "source_hashes": snapshot(args.output), "rounds": [], "setup": {},
              "timing": "resident complete loss/parameter-gradient Graph replay; no optimizer; trace is separate"}
    engines = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for layout in ("dense", "split", "compact"):
        for seed in args.seed_modes:
            key = layout + "/" + seed
            engine = Engine(problem, initial, layout=layout, seed=seed, activation="cuda", native=native)
            for _ in range(5):
                engine.replay()
            engines[key] = engine
            report["setup"][key] = engine.metadata
    torch.cuda.synchronize()
    report["simultaneously_live_engines"] = len(engines)
    report["aggregate_pool_allocated_bytes"] = torch.cuda.memory_allocated()
    report["aggregate_pool_reserved_bytes"] = torch.cuda.memory_reserved()
    # Validate all candidates once more at this actual timing size, outside intervals.
    reference_key = "dense/autograd" if "dense/autograd" in engines else next(iter(engines))
    ref_loss, ref_grad = engines[reference_key].replay()
    torch.cuda.synchronize()
    report["size_validation"] = {}
    for key, engine in engines.items():
        value, gradient = engine.replay()
        report["size_validation"][key] = max(compare(value, ref_loss), compare(gradient, ref_grad))
    rng = random.Random(89241)
    for repeat in range(args.repeats):
        order = list(engines)
        rng.shuffle(order)
        row = {"round": repeat, "order": order, "observations": {}}
        for key in order:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = perf_counter()
            start.record()
            for _ in range(20):
                engines[key].replay()
            end.record()
            end.synchronize()
            row["observations"][key] = {"cuda_event_ms": start.elapsed_time(end)/20,
                                         "synchronized_wall_ms": (perf_counter()-wall)*1000/20}
        report["rounds"].append(row)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print("paired round", repeat+1, "of", args.repeats, flush=True)
    report["medians"] = {key: {metric: statistics.median(r["observations"][key][metric] for r in report["rounds"])
                               for metric in ("cuda_event_ms", "synchronized_wall_ms")} for key in engines}
    report["paired_reference_over_candidate"] = {key: statistics.median(r["observations"][reference_key]["cuda_event_ms"] /
                                                                         r["observations"][key]["cuda_event_ms"] for r in report["rounds"])
                                                    for key in engines}
    report["reference_key"] = reference_key
    report["trace"] = None
    if args.trace:
        try:
            trace_files = []
            for layout in ("dense", "compact"):
                step = make_step(problem, layout=layout, seed="cuda", activation="cuda", native=native, trace=True)
                ws, bs = views(initial.to("cuda"))
                trace_path = args.output.with_name(args.output.stem + "-" + layout + "-eager-trace.json")
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, profile_memory=True) as profiler:
                    for _ in range(3):
                        step(ws, bs)
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(str(trace_path))
                trace_files.append(str(trace_path))
            report["trace"] = {"files": trace_files, "scope": "separate instrumented eager diagnostic; not Graph benchmark timing"}
        except Exception as error:
            report["trace"] = {"unavailable": str(error), "hardware_permissions_changed": False}
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["medians"], indent=2))


if __name__ == "__main__":
    main()
