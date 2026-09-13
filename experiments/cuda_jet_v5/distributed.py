"""Actual native-jet sample DP: globally normalized local sums, FP64 SUM.

Every rank computes full jets and manual parameter gradients. Collectives only
carry a finite-status flag, one flat parameter buffer and one loss scalar.
"""

import argparse
import json
import os
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from common import (
    comparison,
    local_case,
    make_case,
    nested_reference,
    source_hashes,
    step,
)
from gpu import BACKENDS, Native


def flat_parameters(weights, biases):
    return torch.cat([p.flatten() for p in weights + biases])


def copy_parameters(flat, weights, biases):
    offset = 0
    with torch.no_grad():
        for parameter in weights + biases:
            parameter.copy_(
                flat[offset : offset + parameter.numel()].view_as(parameter)
            )
            offset += parameter.numel()


def update_parameters(flat_gradient, weights, biases, learning_rate):
    offset = 0
    with torch.no_grad():
        for parameter in weights + biases:
            parameter.add_(
                flat_gradient[offset : offset + parameter.numel()].view_as(parameter),
                alpha=-learning_rate,
            )
            offset += parameter.numel()


def sum_gradients(
    native,
    backend,
    x,
    quadrature,
    denominator,
    weights,
    biases,
    regularization,
    rank,
    update=False,
    learning_rate=1e-3,
):
    start = perf_counter()
    loss, gradients = step(x, quadrature, denominator, weights, biases, native, backend)
    flat = torch.cat([g.flatten() for g in gradients])
    if rank == 0 and regularization:
        parameters = flat_parameters(weights, biases)
        flat.add_(parameters, alpha=regularization)
        loss.add_(parameters.square().sum(), alpha=0.5 * regularization)
    torch.cuda.synchronize()
    local_end = perf_counter()
    # All ranks reach the same decision before gradient collectives or updates.
    finite = (torch.isfinite(flat).all() & torch.isfinite(loss)).to(torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if int(finite) != 1:
        raise FloatingPointError(
            "at least one rank produced nonfinite loss/gradient; every rank skips update"
        )
    flag_end = perf_counter()
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    reduced = perf_counter()
    if update:
        update_parameters(flat, weights, biases, learning_rate)
    torch.cuda.synchronize()
    finished = perf_counter()
    return (
        loss,
        flat,
        {
            "wall_ms": (finished - start) * 1000,
            "local_compute_and_pack_ms": (local_end - start) * 1000,
            "finite_status_collective_ms": (flag_end - local_end) * 1000,
            "gradient_and_loss_collective_ms": (reduced - flag_end) * 1000,
            "optimizer_ms": (finished - reduced) * 1000,
        },
    )


def serial_reference(
    native, case, device, backend="B1", nested=False, regularization=0
):
    x, q, weights, biases, meta = local_case(case, 1, 0, device)
    if nested:
        loss, gradients = nested_reference(
            x, q, meta["global_denominator"], weights, biases
        )
    else:
        loss, gradients = step(
            x, q, meta["global_denominator"], weights, biases, native, backend
        )
    flat = torch.cat([g.flatten() for g in gradients])
    if regularization:
        parameters = flat_parameters(weights, biases)
        loss.add_(parameters.square().sum(), alpha=0.5 * regularization)
        flat.add_(parameters, alpha=regularization)
    return loss, flat


def validate_case(
    native, case, backends, world, rank, device, regularization, updates=3
):
    x, quadrature, _, _, meta = local_case(case, world, rank, device)
    results = {}
    # Serial independent nested AD is used for protocol cases. Native B1 is the
    # large-batch reference after the separately validated local CUDA contract.
    use_nested = meta["global_batch"] <= 17
    for backend in backends:
        _, _, weights, biases, _ = local_case(case, world, rank, device)
        init = flat_parameters(weights, biases)
        dist.broadcast(init, src=0)
        copy_parameters(init, weights, biases)
        reference_weights = [p.clone() for p in case[2]]
        reference_biases = [p.clone() for p in case[3]]
        checks = []
        for iteration in range(updates):
            dist.barrier()
            loss, flat, _ = sum_gradients(
                native,
                backend,
                x,
                quadrature,
                meta["global_denominator"],
                weights,
                biases,
                regularization,
                rank,
            )
            if rank == 0:
                current = (
                    case[0],
                    case[1],
                    reference_weights,
                    reference_biases,
                    case[4],
                )
                expected_loss, expected_gradient = serial_reference(
                    native,
                    current,
                    device,
                    nested=use_nested,
                    regularization=regularization,
                )
                try:
                    check = {
                        "iteration": iteration,
                        "loss": comparison(loss, expected_loss),
                        "all_parameters": comparison(flat, expected_gradient),
                    }
                    update_parameters(
                        expected_gradient.cpu(),
                        reference_weights,
                        reference_biases,
                        1e-3,
                    )
                except Exception as error:  # noqa: BLE001 -- broadcast failure before all ranks proceed
                    check = {"failure": str(error)}
            else:
                check = None
            decision = [check]
            dist.broadcast_object_list(decision, src=0)
            if "failure" in decision[0]:
                raise AssertionError(decision[0])
            update_parameters(flat, weights, biases, 1e-3)
            checks.append(decision[0])
        if rank == 0:
            try:
                expected_parameters = flat_parameters(
                    reference_weights, reference_biases
                ).to(device)
                final_check = comparison(
                    flat_parameters(weights, biases), expected_parameters
                )
            except Exception as error:  # noqa: BLE001 -- keep rank decisions aligned
                final_check = {"failure": str(error)}
        else:
            final_check = None
        decision = [final_check]
        dist.broadcast_object_list(decision, src=0)
        if "failure" in decision[0]:
            raise AssertionError(decision[0])
        final_check = decision[0]
        results[backend] = {"gradient_steps": checks, "final_parameters": final_check}
    return {
        "case": meta,
        "regularization_added_once": regularization,
        "updates": updates,
        "reference": "independent nested stable AD"
        if use_nested
        else "serial full-global native B1",
        "backends": results,
        "passed": True,
    }


def benchmark_case(native, case, backends, world, rank, device, update, repeats):
    x, q, _, _, meta = local_case(case, world, rank, device)
    state = {}
    for backend in backends:
        _, _, weights, biases, _ = local_case(case, world, rank, device)
        state[backend] = (weights, biases)
        for _ in range(3):
            sum_gradients(
                native,
                backend,
                x,
                q,
                meta["global_denominator"],
                weights,
                biases,
                0,
                rank,
                update,
            )
    torch.cuda.synchronize()
    observations = []
    for repeat in range(repeats):
        order = backends[repeat % len(backends) :] + backends[: repeat % len(backends)]
        if repeat % 2:
            order = list(reversed(order))
        row = {"repeat": repeat, "order": order}
        for backend in order:
            weights, biases = state[backend]
            dist.barrier()  # Synchronize start; explicitly outside the timer.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated_before = torch.cuda.memory_allocated()
            loss, flat, metrics = sum_gradients(
                native,
                backend,
                x,
                q,
                meta["global_denominator"],
                weights,
                biases,
                0,
                rank,
                update,
            )
            metrics.update(
                {
                    "allocated_before_bytes": allocated_before,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "additional_peak_bytes": torch.cuda.max_memory_allocated()
                    - allocated_before,
                }
            )
            row[backend] = metrics
            del loss, flat
        observations.append(row)
    # Check all final optimizer states against the serial global B1 trajectory.
    # The validation time is deliberately outside the benchmark.
    final_checks = None
    if update:
        if rank == 0:
            sx, sq, sw, sb, sm = local_case(case, 1, 0, device)
            for _ in range(3 + repeats):
                _, gradients = step(
                    sx, sq, sm["global_denominator"], sw, sb, native, "B1"
                )
                update_parameters(
                    torch.cat([g.flatten() for g in gradients]), sw, sb, 1e-3
                )
            expected = flat_parameters(sw, sb)
            try:
                final_checks = {
                    backend: comparison(flat_parameters(*state[backend]), expected)
                    for backend in backends
                }
            except Exception as error:  # noqa: BLE001 -- all ranks receive the same failure
                final_checks = {"failure": str(error)}
        decision = [final_checks]
        dist.broadcast_object_list(decision, src=0)
        if "failure" in decision[0]:
            raise AssertionError(decision[0])
        final_checks = decision[0]
    gathered = [None] * world
    dist.all_gather_object(
        gathered,
        {
            "rank": rank,
            "physical_device": device,
            "local_batch": x.shape[0],
            "observations": observations,
        },
    )
    if rank != 0:
        return None
    maxima = [
        {
            backend: max(g["observations"][i][backend]["wall_ms"] for g in gathered)
            for backend in backends
        }
        for i in range(repeats)
    ]
    return {
        "global_case": case[4],
        "world_size": world,
        "includes_optimizer": update,
        "initial_barrier_in_timing": False,
        "warmup_steps": 3,
        "paired_max_rank_wall_ms": maxima,
        "median_max_rank_wall_ms": {
            backend: statistics.median(row[backend] for row in maxima)
            for backend in backends
        },
        "by_rank": gathered,
        "optimizer_final_state_checks": final_checks,
        "gradient_collective_bytes": sum(p.numel() for p in case[2] + case[3]) * 8,
        "other_per_step_collectives": {
            "finite_status_MIN_bytes": 4,
            "loss_SUM_bytes": 8,
        },
        "timing_scope": "native local full gradient + pack + finite-status MIN + FP64 parameter SUM + loss SUM + optional SGD; initialization, warmup, barriers and independent validation excluded",
    }


def collective_probe(world, rank, device):
    rows = []
    for elements in (4547, 131072):
        values = torch.empty(elements, device=device, dtype=torch.float64)
        wall = []
        for iteration in range(25):
            values.fill_(rank + 1)
            torch.cuda.synchronize()
            dist.barrier()
            start = perf_counter()
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            if iteration >= 5:
                wall.append((perf_counter() - start) * 1000)
        assert bool((values == world * (world + 1) / 2).all())
        gathered = [None] * world
        dist.all_gather_object(gathered, wall)
        rows.append(
            {
                "bytes": elements * 8,
                "by_rank_wall_ms": gathered,
                "median_max_rank_ms": statistics.median(
                    max(r[i] for r in gathered) for i in range(20)
                ),
                "passed": True,
            }
        )
    # Inject one nonfinite flag and confirm the update decision is unanimous.
    finite = torch.tensor(
        0 if rank == world - 1 else 1, device=device, dtype=torch.int32
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    assert int(finite) == 0
    return {"payloads": rows, "global_failure_decision_checked": True, "passed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backends", default="B1,U,F,F3")
    parser.add_argument("--suite", choices=["smoke", "strong", "full"], default="smoke")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    rank, local_rank, world = (
        int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    device = f"cuda:{local_rank}"
    backends = args.backends.split(",")
    if (
        not backends
        or any(b not in BACKENDS for b in backends)
        or len(set(backends)) != len(backends)
    ):
        raise ValueError("backends must be unique supported names")
    initialization = perf_counter()
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=120), device_id=torch.device(device)
    )
    native = Native()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "world_size": world,
        "rank": rank,
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(
            getattr(torch.cuda.get_device_properties(local_rank), "uuid", None)
        ),
        "compute_capability": torch.cuda.get_device_capability(local_rank),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nccl_version": torch.cuda.nccl.version(),
        "source_hashes": source_hashes(),
        "build": native.build,
        "gradient_reduction": "SUM_after_global_normalization",
        "control_env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("NCCL_", "TORCH_NCCL_")) or k == "CUDA_VISIBLE_DEVICES"
        },
        "initialization_seconds": perf_counter() - initialization,
        "correctness": [],
        "benchmarks": [],
    }
    try:
        report["collective_probe"] = collective_probe(world, rank, device)
        for batch in (17, 3):
            report["correctness"].append(
                validate_case(
                    native,
                    make_case(batch, 7, weighted=True),
                    backends,
                    world,
                    rank,
                    device,
                    1e-4,
                )
            )
            if rank == 0:
                print(
                    {"world": world, "protocol_global_batch": batch, "passed": True},
                    flush=True,
                )
        if args.suite != "smoke":
            cases = [("strong_16384", make_case(16384, 64))]
            if args.suite == "full":
                cases += [
                    ("strong_65536", make_case(65536, 64)),
                    ("weak_local4096", make_case(4096 * world, 64)),
                ]
            for name, case in cases:
                report["correctness"].append(
                    validate_case(
                        native, case, backends, world, rank, device, 0, updates=1
                    )
                )
                for optimizer in (False, True):
                    row = benchmark_case(
                        native,
                        case,
                        backends,
                        world,
                        rank,
                        device,
                        optimizer,
                        args.repeats,
                    )
                    if rank == 0:
                        row["case_id"] = name
                        report["benchmarks"].append(row)
                        print(
                            {
                                "world": world,
                                "case": name,
                                "optimizer": optimizer,
                                "times_ms": row["median_max_rank_wall_ms"],
                            },
                            flush=True,
                        )
        report["passed"] = True
    except Exception as error:  # noqa: BLE001 -- retain failure; torchrun terminates failed process groups
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        report["passed"] = False
    finally:
        maps = Path("/proc/self/maps")
        report["loaded_cuda_libraries"] = sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if any(name in line for name in ("libnccl", "libcublas", "libcudart"))
            }
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / f"rank{rank}.json").open("x") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
        dist.destroy_process_group()
    print(
        {
            "rank": rank,
            "world": world,
            "passed": report["passed"],
            "failure": report.get("failure"),
        },
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
