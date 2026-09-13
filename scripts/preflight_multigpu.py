"""Read-only inventory plus separately labeled FP64/copy smoke tests."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from time import perf_counter

import torch
from doctor import command, snapshot


def collect(copy_smoke=False):
    report = snapshot()
    report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    for name, argv in {
        "gpu_uuid": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total",
            "--format=csv",
        ],
        "topology": ["nvidia-smi", "topo", "-m"],
        "p2p_read": ["nvidia-smi", "topo", "-p2p", "r"],
        "p2p_write": ["nvidia-smi", "topo", "-p2p", "w"],
        "gpu_metrics": [
            "nvidia-smi",
            "--query-gpu=index,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,memory.used",
            "--format=csv",
        ],
        "cpu": ["lscpu", "-J"],
        "pip_check": [os.sys.executable, "-m", "pip", "check"],
    }.items():
        report["commands"][name] = command(argv)
    count = torch.cuda.device_count()
    report["distributed"] = {
        "available": torch.distributed.is_available(),
        "nccl_available": torch.distributed.is_nccl_available(),
        "nccl_version": torch.cuda.nccl.version()
        if torch.distributed.is_nccl_available()
        else None,
    }
    report["p2p_capability"] = [
        [
            torch.cuda.can_device_access_peer(i, j) if i != j else None
            for j in range(count)
        ]
        for i in range(count)
    ]
    report["fp64_smoke"] = []
    for index in range(count):
        with torch.cuda.device(index):
            x = torch.arange(12, device=index, dtype=torch.float64).reshape(3, 4)
            expected = x.cpu() @ x.cpu().T
            value = x @ x.T
            torch.cuda.synchronize(index)
            error = float((value.cpu() - expected).abs().max())
            report["fp64_smoke"].append(
                {"device": index, "max_abs": error, "passed": error == 0}
            )
    maps = Path("/proc/self/maps")
    report["loaded_cuda_libraries"] = sorted(
        {
            line.split()[-1]
            for line in maps.read_text().splitlines()
            if any(
                name in line
                for name in ["libcudart", "libcublas", "libnccl", "libcuda.so"]
            )
        }
    )
    report["copy_smoke"] = None
    if copy_smoke:
        rows = []
        elements = 2**23  # 64 MiB per buffer, one ordered pair at a time.
        for source in range(count):
            for target in range(count):
                if source == target:
                    continue
                src = torch.full(
                    (elements,), source + 0.125, device=source, dtype=torch.float64
                )
                dst = torch.empty(elements, device=target, dtype=torch.float64)
                torch.cuda.synchronize(source)
                dst.copy_(src, non_blocking=True)
                torch.cuda.synchronize(target)
                wall = []
                for _ in range(5):
                    start = perf_counter()
                    dst.copy_(src, non_blocking=True)
                    torch.cuda.synchronize(target)
                    wall.append(perf_counter() - start)
                passed = bool((dst == source + 0.125).all())
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "bytes": elements * 8,
                        "wall_seconds": wall,
                        "median_GB_per_second": elements
                        * 8
                        / statistics.median(wall)
                        / 1e9,
                        "passed": passed,
                    }
                )
                del src, dst
        report["copy_smoke"] = {
            "method": "Torch CUDA copy_ and destination synchronize; actual transport not inferred from capability alone",
            "ordered_pairs": rows,
        }
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-gpus", type=int, default=4)
    parser.add_argument("--copy-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.copy_smoke)
    report["passed"] = torch.cuda.device_count() == args.expect_gpus and all(
        r["passed"] for r in report["fp64_smoke"]
    )
    if report["copy_smoke"]:
        report["passed"] &= all(
            r["passed"] for r in report["copy_smoke"]["ordered_pairs"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(
        {
            "output": str(args.output),
            "passed": report["passed"],
            "gpu_count": torch.cuda.device_count(),
        },
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
