"""Four fixed independent jobs, scheduled over 1/2/4 persistent GPU workers."""

import argparse
import json
import multiprocessing as mp
import statistics
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch
from common import comparison, local_case, make_case, source_hashes, step
from gpu import Native


def worker(rank, world, batch, commands, responses):
    try:
        torch.cuda.set_device(rank)
        torch.set_num_threads(2)
        native = Native()
        jobs, checks, metadata = [], [], []
        for job_id in range(rank, 4, world):
            x, q, w, b, meta = local_case(
                make_case(batch, seed=7510 + job_id * 7), 1, 0, f"cuda:{rank}"
            )
            jobs.append((x, q, w, b, meta))
            expected_loss, expected_gradients = step(
                x, q, meta["global_denominator"], w, b, native, "B1"
            )
            loss, gradients = step(x, q, meta["global_denominator"], w, b, native, "F3")
            checks.append(
                {
                    "job_id": job_id,
                    "loss": comparison(loss, expected_loss),
                    "gradient": comparison(
                        torch.cat([g.flatten() for g in gradients]),
                        torch.cat([g.flatten() for g in expected_gradients]),
                    ),
                }
            )
            metadata.append({"job_id": job_id, "case": meta})
        for backend in ("B1", "F3"):
            for _ in range(3):
                for x, q, w, b, meta in jobs:
                    step(x, q, meta["global_denominator"], w, b, native, backend)
        torch.cuda.synchronize()
        responses.put(
            {
                "kind": "ready",
                "rank": rank,
                "gpu_uuid": str(torch.cuda.get_device_properties(rank).uuid),
                "jobs": metadata,
                "checks": checks,
                "build": native.build,
            }
        )
        while True:
            backend = commands.get()
            if backend is None:
                break
            start = perf_counter()
            for x, q, w, b, meta in jobs:
                loss, gradients = step(
                    x, q, meta["global_denominator"], w, b, native, backend
                )
                del loss, gradients
            torch.cuda.synchronize()
            responses.put(
                {
                    "kind": "result",
                    "rank": rank,
                    "backend": backend,
                    "compute_ms": (perf_counter() - start) * 1000,
                }
            )
    except Exception:
        responses.put({"kind": "error", "rank": rank, "error": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    context = mp.get_context("spawn")
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes(),
        "fixed_jobs": 4,
        "points_per_job": args.batch,
        "includes_optimizer": False,
        "timing_scope": "persistent workers; wall includes parent dispatch, four complete independent gradients, CUDA synchronize and response IPC; excludes process startup, case creation, warmup and correctness checks",
        "worlds": [],
    }
    for world in (1, 2, 4):
        responses = context.Queue()
        commands = [context.Queue() for _ in range(world)]
        processes = [
            context.Process(
                target=worker, args=(rank, world, args.batch, commands[rank], responses)
            )
            for rank in range(world)
        ]
        startup = perf_counter()
        try:
            for process in processes:
                process.start()
            ready = [responses.get(timeout=180) for _ in range(world)]
            if any(row["kind"] != "ready" for row in ready):
                raise RuntimeError(ready)
            startup_seconds = perf_counter() - startup
            observations = []
            for repeat in range(args.repeats):
                order = ("B1", "F3") if repeat % 2 == 0 else ("F3", "B1")
                row = {"repeat": repeat, "order": order}
                for backend in order:
                    start = perf_counter()
                    for command in commands:
                        command.put(backend)
                    results = [responses.get(timeout=120) for _ in range(world)]
                    elapsed = (perf_counter() - start) * 1000
                    if any(
                        r["kind"] != "result" or r["backend"] != backend
                        for r in results
                    ):
                        raise RuntimeError(results)
                    row[backend] = {
                        "four_job_wall_ms": elapsed,
                        "workers": sorted(results, key=lambda r: r["rank"]),
                    }
                observations.append(row)
            result = {
                "world_size": world,
                "startup_warmup_validation_seconds": startup_seconds,
                "workers": sorted(ready, key=lambda r: r["rank"]),
                "observations": observations,
                "median_four_job_wall_ms": {
                    backend: statistics.median(
                        row[backend]["four_job_wall_ms"] for row in observations
                    )
                    for backend in ("B1", "F3")
                },
            }
            report["worlds"].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                {
                    "independent_workers": world,
                    "four_job_ms": result["median_four_job_wall_ms"],
                },
                flush=True,
            )
        finally:
            for command in commands:
                command.put(None)
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join()
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
