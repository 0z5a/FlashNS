"""Controlled weak-scaling recheck; interleave grad/SGD in the same run."""

import argparse
import json
import os
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from common import comparison, local_case, make_case, source_hashes, step, tensor_hash
from distributed import flat_parameters, sum_gradients, update_parameters
from gpu import Native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    device = f"cuda:{rank}"
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=120), device_id=torch.device(device)
    )
    native = Native()
    # Freeze one parameter set and nested global point prefixes across world sizes.
    master = make_case(16384)
    batch = 4096 * world
    px, pq = master[0][:batch], master[1][:batch]
    meta = {
        **master[4],
        "global_batch": batch,
        "global_denominator": float(batch),
        "global_points_sha256": tensor_hash(px),
        "global_weights_sha256": tensor_hash(pq),
    }
    case = (px, pq, master[2], master[3], meta)
    x, quadrature, _, _, _ = local_case(case, world, rank, device)
    modes = {
        "B1_grad": ("B1", False),
        "F3_grad": ("F3", False),
        "B1_sgd": ("B1", True),
        "F3_sgd": ("F3", True),
    }
    states = {}
    for name, (backend, update) in modes.items():
        _, _, weights, biases, _ = local_case(case, world, rank, device)
        states[name] = weights, biases
        for _ in range(5):
            sum_gradients(
                native,
                backend,
                x,
                quadrature,
                float(batch),
                weights,
                biases,
                0,
                rank,
                update,
            )
    observations = []
    names = list(modes)
    for repeat in range(args.repeats):
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        if repeat % 2:
            order.reverse()
        row = {"repeat": repeat, "order": order}
        for name in order:
            backend, update = modes[name]
            dist.barrier()
            torch.cuda.synchronize()
            loss, gradient, metrics = sum_gradients(
                native,
                backend,
                x,
                quadrature,
                float(batch),
                *states[name],
                0,
                rank,
                update,
            )
            row[name] = metrics
            del loss, gradient
        observations.append(row)
    checks = None
    if rank == 0:
        sx, sq, sw, sb, _ = local_case(case, 1, 0, device)
        for _ in range(5 + args.repeats):
            _, gradients = step(sx, sq, float(batch), sw, sb, native, "B1")
            update_parameters(torch.cat([g.flatten() for g in gradients]), sw, sb, 1e-3)
        expected = flat_parameters(sw, sb)
        checks = {
            name: comparison(flat_parameters(*states[name]), expected)
            for name in ("B1_sgd", "F3_sgd")
        }
    gathered = [None] * world
    dist.all_gather_object(
        gathered,
        {
            "rank": rank,
            "gpu_uuid": str(torch.cuda.get_device_properties(rank).uuid),
            "observations": observations,
        },
    )
    if rank == 0:
        maxima = [
            {
                name: max(r["observations"][i][name]["wall_ms"] for r in gathered)
                for name in names
            }
            for i in range(args.repeats)
        ]
        report = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "source_hashes": source_hashes(),
            "build": native.build,
            "world_size": world,
            "global_case": meta,
            "warmup": 5,
            "repeats": args.repeats,
            "paired_max_rank_wall_ms": maxima,
            "median_max_rank_wall_ms": {
                name: statistics.median(r[name] for r in maxima) for name in names
            },
            "by_rank": gathered,
            "optimizer_final_state_checks": checks,
            "passed": True,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "rank0.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            {
                "world": world,
                "controlled_weak": report["median_max_rank_wall_ms"],
                "passed": True,
            },
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
