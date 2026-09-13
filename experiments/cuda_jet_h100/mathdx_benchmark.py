"""Finite MathDx tile A/B against Torch/Lt and the existing native FP64 kernels."""

import argparse
import gc
import json
import statistics
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import torch
from baseline_plus import step_with_gemm
from benchmark import measure, telemetry
from blaslt import Gemm
from common import comparison, local_case, make_case, nested_reference, source_hashes
from gpu import Native
from graph_experiment import CapturedStep, check_outputs
from mathdx import SHAPES, MathDx
from scientific_benchmark import CapturedCall
from validate import tail_tensor_comparison


def validate(engine, native):
    rows = []
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    for tile in (32, 64):
        for n, k in SHAPES:
            for transpose_b in (False, True):
                with torch.cuda.stream(producer):
                    a = torch.randn(137, k, device="cuda", dtype=torch.float64)
                    b = torch.randn(
                        (n, k) if transpose_b else (k, n),
                        device="cuda",
                        dtype=torch.float64,
                    )
                consumer.wait_stream(producer)
                with torch.cuda.stream(consumer):
                    output = engine.mm(a, b, transpose_b=transpose_b, tile=tile)
                    reference = a @ (b.T if transpose_b else b)
                torch.cuda.current_stream().wait_stream(consumer)
                rows.append(
                    {
                        "tile_m": tile,
                        "m": 137,
                        "n": n,
                        "k": k,
                        "transpose_b": transpose_b,
                        "producer_consumer_streams": True,
                        **comparison(output, reference),
                    }
                )
    networks, tails, updates = [], [], []
    for tile in (32, 64):
        gemm = partial(engine.gemm, tile=tile)
        function = partial(step_with_gemm, gemm=gemm)
        x, q, ws, bs, meta = local_case(make_case(5, 64, weighted=True), 1, 0, "cuda:0")
        networks.append(
            {
                "tile_m": tile,
                **check_outputs(
                    function(x, q, meta["global_denominator"], ws, bs, native),
                    nested_reference(x, q, meta["global_denominator"], ws, bs),
                ),
            }
        )
        for shift in (20.0, -20.0):
            gen = torch.Generator().manual_seed(7613)
            x = (torch.randn(3, 2, generator=gen, dtype=torch.float64) * 0.2).cuda()
            ws = [
                (torch.randn(64, 2, generator=gen, dtype=torch.float64) * 0.2).cuda(),
                (torch.randn(3, 64, generator=gen, dtype=torch.float64) * 0.1).cuda(),
            ]
            bs = [
                torch.full((64,), shift, dtype=torch.float64, device="cuda"),
                torch.zeros(3, dtype=torch.float64, device="cuda"),
            ]
            q = torch.tensor([0.5, 1, 1.5], dtype=torch.float64, device="cuda")
            actual, expected = (
                function(x, q, 3.0, ws, bs, native),
                nested_reference(x, q, 3.0, ws, bs),
            )
            tails.append(
                {
                    "tile_m": tile,
                    "bias_shift": shift,
                    "loss": tail_tensor_comparison(actual[0], expected[0]),
                    "gradients": [
                        tail_tensor_comparison(a, b)
                        for a, b in zip(actual[1], expected[1])
                    ],
                }
            )
        state = CapturedStep(native, make_case(5, 64, weighted=True), "B1", function)
        checks = []
        for index in range(4):
            with torch.no_grad():
                state.x.add_(0.001)
            actual, expected = state.replay(), state.reference(nested=True)
            check = check_outputs(actual, expected)
            old = [p.clone() for p in state.weights + state.biases]
            with torch.no_grad():
                for parameter, gradient in zip(state.weights + state.biases, actual[1]):
                    parameter.add_(gradient, alpha=-1e-3)
            check["parameters"] = comparison(
                torch.cat([p.flatten() for p in state.weights + state.biases]),
                torch.cat([(p - 1e-3 * g).flatten() for p, g in zip(old, expected[1])]),
            )
            checks.append({"iteration": index, **check})
        updates.append({"tile_m": tile, "checks": checks})
        torch.cuda.synchronize()
        del state
    return {
        "passed": True,
        "matrices": rows,
        "full_gradient": networks,
        "tail_networks": tails,
        "changed_inputs_and_four_SGD_updates": updates,
    }


def dgrad_gemm(d, w, h, aux, native, gemm):
    adjoint = gemm(d.flatten(0, 1), w, role="dgrad")
    return native.vjp(h, aux, adjoint.view_as(h), 2 if d.shape[1] == 10 else 3)


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
    engine, native = MathDx(), Native()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes(),
        "mathdx": engine.report(),
        "native_build": native.build,
        "telemetry_before": telemetry(),
        "scope": "same FP64 stable full-jet semantics; explicit MathDx tiles 32 and 64; all global wgrad reductions remain Torch",
        "toolchain_scope": "MathDx 0.7.1 requires isolated CUDA 13; Torch/Lt/native originals remain CUDA 12.8; this is not a compiler-only ablation",
        "includes_optimizer": False,
        "includes_validation_in_timing": False,
        "includes_plan_search_or_capture_in_timing": False,
        "validation": validate(engine, native),
        "local": [],
        "complete_gradient": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    print({"validation": "passed"}, flush=True)
    if not args.validate_only:
        for dim, q in ((2, 10), (3, 20)):
            for batch in (4096, 16384):
                for channels in (32, 64):
                    for cout in (3, channels):
                        torch.manual_seed(8060 + dim + batch + channels + cout)
                        z = (
                            torch.randn(
                                batch, q, channels, device="cuda", dtype=torch.float64
                            )
                            * 0.2
                        )
                        h, aux = native.forward(z, dim)
                        d = (
                            torch.randn(
                                batch, q, cout, device="cuda", dtype=torch.float64
                            )
                            * 0.2
                        )
                        w = (
                            torch.randn(
                                cout, channels, device="cuda", dtype=torch.float64
                            )
                            * 0.2
                        )
                        planned = Gemm()
                        functions = {
                            name: partial(native.dgrad, d, w, h, aux, dim, name)
                            for name in (
                                ("B1", "U", "F", "F_n32", "F3")
                                if cout == 3
                                else ("B1", "U", "F", "F_n32")
                            )
                        }
                        functions["B1+"] = partial(
                            dgrad_gemm, d, w, h, aux, native, planned
                        )
                        for tile in (32, 64):
                            functions[f"MathDx{tile}"] = partial(
                                dgrad_gemm,
                                d,
                                w,
                                h,
                                aux,
                                native,
                                partial(engine.gemm, tile=tile),
                            )
                        reference = functions["B1"]()
                        checks = {
                            name: comparison(function(), reference)
                            for name, function in functions.items()
                        }
                        planned.freeze()
                        captures = {
                            name: CapturedCall(function)
                            for name, function in functions.items()
                        }
                        measured = {
                            **{name + "_eager": f for name, f in functions.items()},
                            **{
                                name + "_replay": c.replay
                                for name, c in captures.items()
                            },
                        }
                        row = {
                            "dimension": dim,
                            "batch": batch,
                            "cin": channels,
                            "cout": cout,
                            "correctness": checks,
                            "baseline_plans": planned.report(),
                            **measure(measured, args.repeats, 5),
                        }
                        report["local"].append(row)
                        save()
                        print(
                            {
                                "local": [dim, batch, channels, cout],
                                "event_ms": {
                                    n: v["cuda_event_ms"]
                                    for n, v in row["medians"].items()
                                    if n.endswith("replay")
                                },
                            },
                            flush=True,
                        )
                        torch.cuda.synchronize()
                        del (
                            functions,
                            measured,
                            captures,
                            planned,
                            z,
                            h,
                            aux,
                            d,
                            w,
                            reference,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
    for batch in [] if args.validate_only else map(int, args.batches.split(",")):
        case, planned = make_case(batch), Gemm()
        tuned_step = partial(step_with_gemm, gemm=planned)
        states = {
            "B1": CapturedStep(native, case, "B1"),
            "F": CapturedStep(native, case, "F"),
            "B1+": CapturedStep(native, case, "B1", tuned_step),
            "F+": CapturedStep(native, case, "F", tuned_step),
        }
        for tile in (32, 64):
            states[f"MathDx{tile}"] = CapturedStep(
                native,
                case,
                "B1",
                partial(step_with_gemm, gemm=partial(engine.gemm, tile=tile)),
            )
        planned.freeze()
        reference = states["B1"].reference()
        checks = {
            name: {
                "eager": check_outputs(state.eager(), reference),
                "replay": check_outputs(state.replay(), reference),
            }
            for name, state in states.items()
        }
        del reference
        functions = {
            f"{name}_{mode}": getattr(state, mode)
            for name, state in states.items()
            for mode in ("eager", "replay")
        }
        row = {
            "case": case[4],
            "correctness": checks,
            "baseline_plans": planned.report(),
            "captures": {name: state.capture for name, state in states.items()},
            **measure(functions, args.repeats, 5),
        }
        row["paired_speedup_medians"] = {
            f"{name}_over_F+_{mode}": statistics.median(
                r[f"{name}_{mode}"]["cuda_event_ms"] / r[f"F+_{mode}"]["cuda_event_ms"]
                for r in row["paired_observations"]
            )
            for name in states
            if name != "F+"
            for mode in ("eager", "replay")
        }
        report["complete_gradient"].append(row)
        save()
        print(
            {
                "batch": batch,
                "event_ms": {n: v["cuda_event_ms"] for n, v in row["medians"].items()},
            },
            flush=True,
        )
        torch.cuda.synchronize()
        del states, functions, planned, tuned_step
        gc.collect()
        torch.cuda.empty_cache()
    report["telemetry_after"] = telemetry()
    report["passed"] = True
    save()


if __name__ == "__main__":
    main()
