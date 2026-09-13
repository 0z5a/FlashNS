"""Accuracy and resident/transfer-inclusive timing of the local formal adapter."""

import argparse
import faulthandler
import hashlib
import json
import os
import random
import resource
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from projected_amplitude import (
    compare,
    contract,
    make_case,
    numpy_operator,
    saddle_reference,
    source_hashes,
    torch_operator,
    trajectory_validation,
)


def derivative_validation():
    import mpmath as mp

    data, _, geometry = make_case(12, seed=761813)
    results = []
    with mp.workdps(100):
        for index in range(12):
            values = [mp.mpf(float(value)) for value in data[index]]
            initial = [mp.mpf(float(value)) for value in geometry["nzero"][index]]
            prime = [mp.mpf(float(value)) for value in geometry["nprime"][index]]
            scale = mp.mpf(float(geometry["damping_scale"][index]))
            pulse = float(geometry["pulse"][index])

            def reference(time):
                n = [initial[i] + time * prime[i] for i in range(3)]
                delta = scale * sum(value * value for value in n)
                matrix = mp.eye(4)
                for i in range(3):
                    matrix[i, 3], matrix[3, i] = n[i], n[i]
                matrix[3, 3] = 0
                derivative, pressure = [], []
                for part in range(2):
                    t, source = (
                        values[12 + 3 * part : 15 + 3 * part],
                        values[18 + 3 * part : 21 + 3 * part],
                    )
                    kt = [
                        -2 * values[6] * t[1],
                        (2 * values[6] + values[7]) * t[0],
                        values[8] * t[0],
                    ]
                    right = mp.matrix(
                        [-kt[i] - delta * t[i] - source[i] for i in range(3)]
                        + [-sum((prime[i] + delta * n[i]) * t[i] for i in range(3))]
                    )
                    solution = mp.lu_solve(matrix, right)
                    derivative.extend(solution[:3])
                    pressure.append(-solution[3] / (values[10] * values[11]))
                return derivative + [-pressure[1], pressure[0]]

            gpurow = torch.tensor(data[index], dtype=torch.float64, device="cuda")
            nzero = torch.tensor(
                geometry["nzero"][index], dtype=torch.float64, device="cuda"
            )
            nprime = torch.tensor(
                geometry["nprime"][index], dtype=torch.float64, device="cuda"
            )
            gscale = float(geometry["damping_scale"][index])

            def function(time):
                n = nzero + time * nprime
                delta = gscale * (n * n).sum()
                row = torch.cat((n, gpurow[3:9], delta.reshape(1), gpurow[10:]))
                return torch_operator(row.reshape(1, 24)).reshape(8)

            time = torch.tensor(pulse, device="cuda", dtype=torch.float64)
            derivative = function
            checks = []
            for order in (1, 2, 3):
                derivative = torch.func.jacfwd(derivative)
                actual = derivative(time).detach().cpu().numpy()
                expected = np.array(
                    [
                        float(
                            mp.diff(
                                lambda point: reference(point)[component],
                                mp.mpf(pulse),
                                order,
                            )
                        )
                        for component in range(8)
                    ]
                )
                checks.append(
                    {
                        "order": order,
                        **compare(actual, expected, atol=1e-10, rtol=2e-10),
                    }
                )
            results.append(
                {
                    "label": index,
                    "partial_derivative": "pulse coordinate, fixed complex amplitude/source and discrete frequencies",
                    "checks": checks,
                }
            )
    return results


def validation(compiled):
    data, metadata, _ = make_case(4096)
    expected = numpy_operator(data)
    device = torch.tensor(data, device="cuda")
    ids = np.linspace(0, len(data) - 1, 96, dtype=int)
    high = saddle_reference(data[ids], 100)
    high2 = saddle_reference(data[ids], 120)
    result = {
        "case": metadata,
        "independent_100_vs_120_digits": compare(high, high2, atol=1e-25, rtol=1e-25),
        "numpy_vs_saddle": compare(expected[ids], high),
        "gpu_eager": compare(torch_operator(device).cpu().numpy(), expected),
        "gpu_compiled": compare(compiled(device).cpu().numpy(), expected),
        "derivatives": derivative_validation(),
        "trajectories": trajectory_validation(),
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        moved = device * 1.0
        actual = compiled(moved)
    stream.synchronize()
    result["nondefault_stream"] = compare(actual.cpu().numpy(), expected)
    result["passed"] = True
    return result


def numpy_tiled(data, tile=65536):
    output = np.empty((len(data), 8), dtype=data.dtype)
    for offset in range(0, len(data), tile):
        output[offset : offset + tile] = numpy_operator(data[offset : offset + tile])
    return output


def benchmark(compiled, batches, repeats, records, save):
    rng = random.Random(90731)
    for batch in batches:
        print({"batch": batch, "stage": "prepare"}, flush=True)
        start = perf_counter()
        data, meta, _ = make_case(batch, seed=78003)
        record = {
            "case": meta,
            "preparation_seconds": perf_counter() - start,
            "runs": [],
            "completed": False,
        }
        records.append(record)
        save()
        device = torch.tensor(data, device="cuda")
        print({"batch": batch, "stage": "numpy_reference"}, flush=True)
        start = perf_counter()
        reference = numpy_operator(data)
        record["reference_seconds"] = perf_counter() - start
        print(
            {
                "batch": batch,
                "stage": "warmup",
                "reference_seconds": record["reference_seconds"],
            },
            flush=True,
        )
        for _ in range(3):
            compiled(device)
            torch_operator(device)
        torch.cuda.synchronize()
        checks = {
            "compiled": compare(compiled(device).cpu().numpy(), reference),
            "eager": compare(torch_operator(device).cpu().numpy(), reference),
            "cpu_tiled": compare(numpy_tiled(data), reference),
        }
        record["validation"] = checks
        rows = record["runs"]
        save()
        for repeat in range(repeats):
            order = [
                "cpu_numpy",
                "cpu_numpy_tiled",
                "gpu_eager_resident",
                "gpu_compiled_resident",
                "gpu_compiled_with_transfers",
            ]
            rng.shuffle(order)
            row = {"repeat": repeat, "order": order}
            rows.append(row)
            for backend in order:
                print(
                    {
                        "batch": batch,
                        "repeat": repeat,
                        "backend": backend,
                        "stage": "start",
                    },
                    flush=True,
                )
                cpu_before = resource.getrusage(resource.RUSAGE_SELF)
                if backend.startswith("cpu_numpy"):
                    start = perf_counter()
                    result = (
                        numpy_tiled if backend.endswith("tiled") else numpy_operator
                    )(data)
                    row[backend] = perf_counter() - start
                elif backend == "gpu_compiled_with_transfers":
                    torch.cuda.synchronize()
                    start = perf_counter()
                    result = compiled(torch.from_numpy(data).cuda()).cpu().numpy()
                    row[backend] = perf_counter() - start
                else:
                    function = compiled if "compiled" in backend else torch_operator
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    output = function(device)
                    end.record()
                    end.synchronize()
                    row[backend] = start.elapsed_time(end) / 1000
                    result = output.cpu().numpy()
                cpu_after = resource.getrusage(resource.RUSAGE_SELF)
                row.setdefault("process_cpu_seconds", {})[backend] = {
                    "user": cpu_after.ru_utime - cpu_before.ru_utime,
                    "system": cpu_after.ru_stime - cpu_before.ru_stime,
                }
                # Validation is outside every measured interval.
                compare(result, reference)
                save()
                print(
                    {
                        "batch": batch,
                        "repeat": repeat,
                        "backend": backend,
                        "seconds": row[backend],
                    },
                    flush=True,
                )
        medians = {
            backend: statistics.median(row[backend] for row in rows)
            for backend in order
        }
        speedups = {
            backend: statistics.median(
                min(row["cpu_numpy"], row["cpu_numpy_tiled"]) / row[backend]
                for row in rows
            )
            for backend in order
            if not backend.startswith("cpu_numpy")
        }
        record.update(
            {
                "seconds": medians,
                "paired_best_cpu_speedups": speedups,
                "completed": True,
            }
        )
        save()
        print(
            {"batch": batch, "seconds": medians, "cpu_speedups": speedups}, flush=True
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--validation-artifact", type=Path)
    parser.add_argument("--batches", default="65536,1048576")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite a numerical run")
    torch.set_num_threads(1)
    faulthandler.dump_traceback_later(60, repeat=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "contract": contract(),
        "source_hashes": source_hashes(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NUMPY_MADVISE_HUGEPAGE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "device": torch.cuda.get_device_name(),
        "hardware": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,power.limit,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "timing_contract": {
            "repeats": args.repeats,
            "batches": [int(value) for value in args.batches.split(",")],
            "order": "seeded random interleaving",
            "resident_gpu": "CUDA event intervals; device inputs resident; output validation excluded",
            "transfer_inclusive": "wall clock from pageable CPU input through H2D, compute, D2H, synchronization",
            "cpu": "NumPy allocation and arithmetic on fixed CPU input; no mpmath or SciPy reference work",
            "cpu_tiled": "the same NumPy formula in blocks of 65536 with full output allocation and assembly included",
            "speedup_baseline": "the faster of untiled and tiled NumPy in each paired repetition",
        },
        "benchmarks": [],
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    start = perf_counter()
    compiled = torch.compile(torch_operator, fullgraph=True, dynamic=True)
    data, _, _ = make_case(4096)
    compiled(torch.tensor(data, device="cuda"))
    torch.cuda.synchronize()
    report["compile_and_first_call_seconds"] = perf_counter() - start
    if args.validation_artifact:
        previous = json.loads(args.validation_artifact.read_text())
        key = "experiments/openai_ns_formal/projected_amplitude.py"
        if (
            not previous["passed"]
            or previous["source_hashes"][key] != report["source_hashes"][key]
        ):
            raise RuntimeError(
                "reused validation must pass with an identical operator source"
            )
        report["validation"] = previous["validation"]
        report["reused_validation"] = {
            "path": str(args.validation_artifact),
            "sha256": hashlib.sha256(args.validation_artifact.read_bytes()).hexdigest(),
            "numerical_validation_source_hashes": previous["source_hashes"],
        }
    else:
        report["validation"] = validation(compiled)
    save()
    if not args.validation_only:
        benchmark(
            compiled,
            report["timing_contract"]["batches"],
            args.repeats,
            report["benchmarks"],
            save,
        )
    report["passed"] = True
    report["finished_utc"] = datetime.now(UTC).isoformat()
    save()
    faulthandler.cancel_dump_traceback_later()
    print({"passed": True, "validation_only": args.validation_only}, flush=True)


if __name__ == "__main__":
    main()
