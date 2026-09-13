"""Un-timed attribution pass; profiler overhead is excluded from speed claims."""

import argparse
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import torch
from common import local_case, make_case, source_hashes, step
from gpu import Native


def profile(function):
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        function()
        torch.cuda.synchronize()
    rows = [
        {
            "operator": e.key,
            "count": e.count,
            "self_cpu_us": e.self_cpu_time_total,
            "self_device_us": getattr(e, "self_device_time_total", 0),
        }
        for e in profiler.key_averages()
    ]
    return {
        "cuda_event_count": sum(
            str(e.device_type).endswith("CUDA") for e in profiler.events()
        ),
        "self_device_sum_us": sum(row["self_device_us"] for row in rows),
        "top_device_events": sorted(
            rows, key=lambda r: r["self_device_us"], reverse=True
        )[:18],
        "top_cpu_events": sorted(rows, key=lambda r: r["self_cpu_us"], reverse=True)[
            :10
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    native = Native()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes(),
        "build": native.build,
        "performance_measurement": False,
        "profiles": [],
    }
    for batch in (4096, 16384):
        x, q, w, b, meta = local_case(make_case(batch), 1, 0, "cuda:0")
        row = {"case": meta, "backends": {}}
        for backend in ("B1", "U3", "F3"):
            row["backends"][backend] = profile(
                partial(step, x, q, meta["global_denominator"], w, b, native, backend)
            )
        report["profiles"].append(row)
        print(
            {
                "profile": batch,
                "events": {
                    name: data["cuda_event_count"]
                    for name, data in row["backends"].items()
                },
            },
            flush=True,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
