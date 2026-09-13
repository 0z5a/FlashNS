"""A fixed full-gradient ablation with bounded FP64 library plan search."""

import argparse
import gc
import json
import math
import statistics
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import torch
from benchmark import measure, telemetry
from blaslt import Gemm
from common import (
    comparison,
    local_case,
    make_case,
    nested_reference,
    residual_per_point,
    seed_coordinates,
    source_hashes,
)
from gpu import Native
from graph_experiment import CapturedStep, check_outputs
from validate import tail_tensor_comparison

BACKENDS = ("B1", "U", "F", "F3")


def step_with_gemm(
    x, quadrature, denominator, weights, biases, native, backend="B1", *, gemm
):
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("positive global denominator required")
    if x.shape[0] == 0:
        return x.new_zeros(()), [torch.zeros_like(p) for p in weights + biases]
    if backend not in BACKENDS:
        raise ValueError(backend)
    with torch.no_grad():
        hidden = seed_coordinates(x)
        checkpoints, auxiliaries = [hidden], [None]
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            z = gemm(
                hidden.flatten(0, 1), weight, transpose_b=True, role=f"forward_{index}"
            )
            z = z.view(x.shape[0], 10, weight.shape[0])
            z[:, 0] += bias
            if index + 1 < len(weights):
                hidden, aux = native.forward(z, 2)
            else:
                hidden, aux = z, None
            checkpoints.append(hidden)
            auxiliaries.append(aux)
    leaf = hidden.detach().requires_grad_()
    loss = (quadrature * residual_per_point(leaf)).sum() / denominator
    (d,) = torch.autograd.grad(loss, leaf)
    dws, dbs = [None] * len(weights), [None] * len(weights)
    with torch.no_grad():
        for index in reversed(range(len(weights))):
            dws[index] = gemm(
                d.flatten(0, 1),
                checkpoints[index].flatten(0, 1),
                transpose_a=True,
                role=f"wgrad_{index}",
            )
            dbs[index] = d[:, 0].sum(0)
            if index:
                if backend == "B1" or (
                    backend == "F3" and weights[index].shape[0] != 3
                ):
                    adjoint = gemm(
                        d.flatten(0, 1), weights[index], role=f"dgrad_{index}"
                    )
                    d = native.vjp(
                        checkpoints[index],
                        auxiliaries[index],
                        adjoint.view_as(checkpoints[index]),
                        2,
                    )
                else:
                    d = native.dgrad(
                        d,
                        weights[index],
                        checkpoints[index],
                        auxiliaries[index],
                        2,
                        backend,
                    )
    return loss.detach(), dws + dbs


def validate(native):
    gemm = Gemm()
    function = partial(step_with_gemm, gemm=gemm)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    matrices = []
    with torch.cuda.stream(stream), torch.no_grad():
        for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
            m, n, k = 7, 9, 5
            a_shape, b_shape = ((k, m) if ta else (m, k)), ((n, k) if tb else (k, n))
            # Deliberately offset both storage pointers: exercise alignment=8.
            a = torch.randn(math.prod(a_shape) + 1, device="cuda", dtype=torch.float64)[
                1:
            ].view(a_shape)
            b = torch.randn(math.prod(b_shape) + 1, device="cuda", dtype=torch.float64)[
                1:
            ].view(b_shape)
            matrices.append(
                {
                    "transpose_a": ta,
                    "transpose_b": tb,
                    "stream": "nondefault",
                    "check": comparison(
                        gemm(a, b, ta, tb, "matrix_smoke"),
                        (a.T if ta else a) @ (b.T if tb else b),
                    ),
                }
            )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    networks = []
    for scale in (0.01, 0.4, 3.0):
        case = make_case(5, 7, weighted=True, coordinate_scale=scale)
        x, q, ws, bs, meta = local_case(case, 1, 0, "cuda:0")
        expected = nested_reference(x, q, meta["global_denominator"], ws, bs)
        checks = {
            name: check_outputs(
                function(x, q, meta["global_denominator"], ws, bs, native, name),
                expected,
            )
            for name in BACKENDS
        }
        networks.append({"coordinate_scale": scale, "checks": checks})
    tails = []
    for shift in (20.0, -20.0):
        gen = torch.Generator().manual_seed(7513)
        x = (torch.randn(3, 2, generator=gen, dtype=torch.float64) * 0.2).cuda()
        ws = [
            (torch.randn(o, 2, generator=gen, dtype=torch.float64) * 0.2).cuda()
            for o in (2, 3)
        ]
        bs = [
            torch.full((2,), shift, dtype=torch.float64, device="cuda"),
            torch.zeros(3, dtype=torch.float64, device="cuda"),
        ]
        q = torch.tensor([0.5, 1, 1.5], dtype=torch.float64, device="cuda")
        expected = nested_reference(x, q, 3.0, ws, bs)
        checks = {}
        for name in BACKENDS:
            actual = function(x, q, 3.0, ws, bs, native, name)
            checks[name] = {
                "loss": tail_tensor_comparison(actual[0], expected[0]),
                "gradients": [
                    tail_tensor_comparison(a, b) for a, b in zip(actual[1], expected[1])
                ],
            }
        tails.append({"bias_shift": shift, "checks": checks})
    updates = []
    for name in BACKENDS:
        state = CapturedStep(native, make_case(5, 7, weighted=True), name, function)
        rows = []
        for index in range(4):
            with torch.no_grad():
                state.x.add_(0.001)
            actual, expected = state.replay(), state.reference(nested=True)
            checks = check_outputs(actual, expected)
            old = [p.clone() for p in state.weights + state.biases]
            with torch.no_grad():
                for parameter, gradient in zip(state.weights + state.biases, actual[1]):
                    parameter.add_(gradient, alpha=-1e-3)
            checks["parameters"] = comparison(
                torch.cat([p.flatten() for p in state.weights + state.biases]),
                torch.cat([(p - 1e-3 * g).flatten() for p, g in zip(old, expected[1])]),
            )
            rows.append({"iteration": index, **checks})
        updates.append({"backend": name, "checks": rows})
        torch.cuda.synchronize()
        del state
    x, q, ws, bs, meta = local_case(make_case(3, 7), 1, 0, "cuda:0")
    empty = function(x[:0], q[:0], meta["global_denominator"], ws, bs, native)
    assert float(empty[0]) == 0 and all(torch.count_nonzero(g) == 0 for g in empty[1])
    return {
        "passed": True,
        "matrices": matrices,
        "networks": networks,
        "tails": tails,
        "input_and_sgd_updates": updates,
        "empty_batch": True,
        "planning": gemm.report(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="4096,16384,65536,262144")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    native = Native()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes(),
        "native_build": native.build,
        "telemetry_before": telemetry(),
        "includes_optimizer": False,
        "includes_independent_validation_in_timing": False,
        "includes_plan_search_in_timing": False,
        "includes_graph_capture_in_timing": False,
        "graph_pool_sharing": False,
        "validation": validate(native),
        "cases": [],
        "memory_scope": "peak allocator values include all live states/caches; additional peak is per call; graph reservations are not the allocated peak",
    }

    def save():
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    print(
        {
            "validation": "passed",
            "plans": len(report["validation"]["planning"]["plans"]),
        },
        flush=True,
    )
    for batch in [] if args.validate_only else map(int, args.batches.split(",")):
        gemm = Gemm()
        function = partial(step_with_gemm, gemm=gemm)
        case = make_case(batch)
        states = {
            "B1": CapturedStep(native, case, "B1"),
            "F": CapturedStep(native, case, "F"),
        }
        for name in BACKENDS:
            states[name + "+"] = CapturedStep(native, case, name, function)
        gemm.freeze()
        expected = states["B1"].reference()
        checks = {
            name: {
                "eager": check_outputs(state.eager(), expected),
                "replay": check_outputs(state.replay(), expected),
            }
            for name, state in states.items()
        }
        del expected
        functions = {
            f"{name}_{mode}": getattr(state, mode)
            for name, state in states.items()
            for mode in ("eager", "replay")
        }
        row = {
            "case": case[4],
            "correctness": checks,
            "planning": gemm.report(),
            "captures": {name: state.capture for name, state in states.items()},
            **measure(functions, args.repeats, 5),
        }
        ratios = [
            (f"{a}_{mode}", f"{b}_{mode}")
            for a, b in (
                ("B1", "B1+"),
                ("F", "F+"),
                ("B1+", "F+"),
                ("U+", "F+"),
                ("B1+", "F3+"),
            )
            for mode in ("eager", "replay")
        ]
        ratios.extend((f"{name}_eager", f"{name}_replay") for name in states)
        row["paired_speedup_medians"] = {
            f"{a}_over_{b}": statistics.median(
                r[a]["cuda_event_ms"] / r[b]["cuda_event_ms"]
                for r in row["paired_observations"]
            )
            for a, b in ratios
        }
        report["cases"].append(row)
        save()
        print(
            {
                "batch": batch,
                "selected_plans": [p["selected"] for p in row["planning"]["plans"]],
                "event_ms": {n: v["cuda_event_ms"] for n, v in row["medians"].items()},
            },
            flush=True,
        )
        torch.cuda.synchronize()
        del functions, states, function, gemm
        gc.collect()
        torch.cuda.empty_cache()
    report["telemetry_after"] = telemetry()
    report["passed"] = True
    save()


if __name__ == "__main__":
    main()
