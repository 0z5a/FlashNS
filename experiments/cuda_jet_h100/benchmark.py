"""Same-device paired local dgrad/VJP and complete-gradient measurements."""

import argparse
import json
import statistics
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
from common import comparison, local_case, make_case, source_hashes, step
from gpu import BACKENDS, Native


def telemetry():
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used",
        "--format=csv",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "output": result.stdout + result.stderr,
    }


def measure(functions, repeats, inner):
    for function in functions.values():
        for _ in range(4):
            function()
    torch.cuda.synchronize()
    rows = []
    names = list(functions)
    for repeat in range(repeats):
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        if repeat % 2:
            order.reverse()
        row = {"repeat": repeat, "order": order}
        for name in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated()
            start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            wall = perf_counter()
            start.record()
            for _ in range(inner):
                output = functions[name]()
            stop.record()
            stop.synchronize()
            row[name] = {
                "cuda_event_ms": start.elapsed_time(stop) / inner,
                "synchronized_wall_ms": (perf_counter() - wall) * 1000 / inner,
                "additional_peak_bytes": torch.cuda.max_memory_allocated() - allocated,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            del output
        rows.append(row)
    return {
        "inner": inner,
        "paired_observations": rows,
        "medians": {
            name: {
                key: statistics.median(row[name][key] for row in rows)
                for key in rows[0][name]
            }
            for name in names
        },
        "paired_speedup_medians": {
            f"{reference}_over_{candidate}": statistics.median(
                row[reference]["cuda_event_ms"] / row[candidate]["cuda_event_ms"]
                for row in rows
            )
            for reference, candidate in (
                ("B1", "F"),
                ("U", "F"),
                ("U_n32", "F_n32"),
                ("B1", "F3"),
                ("U3", "F3"),
            )
            if reference in names and candidate in names
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.cuda.set_device(args.device)
    torch.set_num_threads(2)
    native = Native()
    device = f"cuda:{args.device}"
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": args.device,
        "gpu_uuid": str(torch.cuda.get_device_properties(args.device).uuid),
        "build": native.build,
        "source_hashes": source_hashes(),
        "telemetry_before": telemetry(),
        "baseline": "B1: current Torch FP64 matmul plus stable native VJP; no cuBLASLt plan search",
        "local": [],
        "complete_gradient": [],
        "includes_optimizer": False,
        "includes_independent_validation_in_timing": False,
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for dim in (2, 3):
        q = 10 if dim == 2 else 20
        for batch in (4096, 16384):
            for cin, cout in ((32, 32), (64, 64), (32, 3), (64, 3)):
                torch.manual_seed(8000 + dim + batch + cin + cout)
                z = torch.randn(batch, q, cin, device=device, dtype=torch.float64) * 0.2
                h, aux = native.forward(z, dim)
                d = (
                    torch.randn(batch, q, cout, device=device, dtype=torch.float64)
                    * 0.2
                )
                w = torch.randn(cout, cin, device=device, dtype=torch.float64) * 0.2
                names = BACKENDS if cout == 3 else BACKENDS[:5]
                functions = {
                    name: partial(native.dgrad, d, w, h, aux, dim, name)
                    for name in names
                }
                expected = functions["B1"]()
                checks = {
                    name: comparison(function(), expected)
                    for name, function in functions.items()
                }
                del expected, z
                row = {
                    "dimension": dim,
                    "batch": batch,
                    "cin": cin,
                    "cout": cout,
                    "correctness": checks,
                    **measure(functions, args.repeats, 5),
                }
                report["local"].append(row)
                print(
                    {
                        "local": [dim, batch, cin, cout],
                        "event_ms": {
                            name: round(data["cuda_event_ms"], 4)
                            for name, data in row["medians"].items()
                        },
                    },
                    flush=True,
                )
                save()
                del functions, h, aux, d, w
    for batch in (4096, 16384, 65536):
        case = make_case(batch)
        x, quadrature, weights, biases, meta = local_case(case, 1, 0, device)
        functions = {
            name: partial(
                step,
                x,
                quadrature,
                meta["global_denominator"],
                weights,
                biases,
                native,
                name,
            )
            for name in BACKENDS
        }
        expected_loss, expected_gradients = functions["B1"]()
        checks = {}
        for name, function in functions.items():
            loss, gradients = function()
            checks[name] = {
                "loss": comparison(loss, expected_loss),
                "gradient": comparison(
                    torch.cat([g.flatten() for g in gradients]),
                    torch.cat([g.flatten() for g in expected_gradients]),
                ),
            }
        del expected_loss, expected_gradients, loss, gradients
        row = {
            "case": meta,
            "correctness": checks,
            **measure(functions, args.repeats, 3),
        }
        report["complete_gradient"].append(row)
        print(
            {
                "complete_gradient": batch,
                "event_ms": {
                    name: round(data["cuda_event_ms"], 4)
                    for name, data in row["medians"].items()
                },
            },
            flush=True,
        )
        save()
    report["telemetry_after"] = telemetry()
    report["passed"] = True
    save()


if __name__ == "__main__":
    main()
