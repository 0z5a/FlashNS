"""Paired scientific-library A/B on the identical full FP64 PINN gradient."""

import argparse
import gc
import hashlib
import json
import statistics
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
from benchmark import measure, telemetry
from common import (
    comparison,
    local_case,
    make_case,
    nested_reference,
    source_hashes,
    step,
)
from gpu import Native
from graph_experiment import CapturedStep, check_outputs
from scientific_backends import CuEqNative, PhysicsNeMoStep, TorchJetNative, versions
from validate import tail_diagnostics, tail_tensor_comparison

HERE = Path(__file__).resolve().parent


def nested_step(x, q, denominator, weights, biases, native=None, backend=None):
    return nested_reference(x, q, denominator, weights, biases)


class CapturedCall:
    def __init__(self, function):
        self.function = function
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(5):
                function()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        start = perf_counter()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.output = function()
        self.capture_seconds = perf_counter() - start

    def replay(self):
        self.graph.replay()
        return self.output


def validate(name, native, backend, function):
    start = perf_counter()
    networks = []
    for scale in (0.01, 0.4, 3.0):
        case = make_case(5, 7, weighted=True, coordinate_scale=scale)
        x, q, ws, bs, meta = local_case(case, 1, 0, "cuda:0")
        expected = step(x, q, meta["global_denominator"], ws, bs, Native())
        actual = function(x, q, meta["global_denominator"], ws, bs, native, backend)
        networks.append({"coordinate_scale": scale, **check_outputs(actual, expected)})
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
        actual = function(x, q, 3.0, ws, bs, native, backend)
        tails.append(
            {
                "bias_shift": shift,
                "loss": tail_tensor_comparison(actual[0], expected[0]),
                "gradients": [
                    tail_tensor_comparison(a, b) for a, b in zip(actual[1], expected[1])
                ],
            }
        )
    state = CapturedStep(native, make_case(5, 7, weighted=True), backend, function)
    updates = []
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
        updates.append({"iteration": index, **checks})
    torch.cuda.synchronize()
    del state
    return {
        "backend": name,
        "networks": networks,
        "tails": tails,
        "changed_input_sgd": updates,
        "seconds_including_first_compilation_and_graph_warmup": perf_counter() - start,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="4096,16384")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    native, cue, eager, compiled = (
        Native(),
        CuEqNative(),
        TorchJetNative(),
        TorchJetNative(compiled=True),
    )
    physics = PhysicsNeMoStep()
    specifications = {
        "B1": (native, "B1", step),
        "F": (native, "F", step),
        "Torch_nested_AD": (native, "B1", nested_step),
        "PhysicsNeMo_AD": (native, "B1", physics),
        "cuEquivariance": (cue, "B1", step),
        "Torch_jet_eager": (eager, "B1", step),
        "Torch_jet_compiled": (compiled, "B1", step),
    }
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes(),
        "native_build": native.build,
        "versions": versions(),
        "telemetry_before": telemetry(),
        "includes_optimizer_in_timing": False,
        "includes_independent_validation_in_timing": False,
        "includes_compilation_in_timing": False,
        "includes_graph_capture_in_timing": False,
        "same_precision_policy": "FP64 stable_aux_a1 / stable scalar tanh; third-order spatial loss, first parameter gradient",
        "workload": "2-D steady NS residual plus first spatial derivatives of residual; 2-C-C-3 network",
        "PhysicsNeMo_scope": "public PhysicsInformer autodiff supplies residuals through second scalar derivatives; Torch AD supplies residual spatial gradient and parameter gradient; shared stable scalar activation",
        "cuEquivariance_scope": "public uniform_1d FP64 JIT polynomial kernels for full jet forward and explicit VJP, shared Torch GEMM/loss/seed",
        "Torch_compile_scope": "Inductor compiles the stable tensor-only jet forward and VJP; common GEMM/loss/seed; dynamic=True, fullgraph=True",
        "graph_pool_sharing": False,
        "memory_scope": "all simultaneously live graph pools and warmup caches; do not interpret graph allocated peak as total graph reservation",
        "validation": {},
        "high_precision_tails": {},
        "failed_backends": [],
        "local": [],
        "complete_gradient": [],
        "dependency_wheel_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (HERE.parents[1] / "sources/h100_library_wheels").glob("*.whl")
        },
    }

    def save():
        report["cuEquivariance_descriptors"] = cue.descriptors
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for name, (target, backend, function) in list(specifications.items()):
        try:
            report["validation"][name] = validate(name, target, backend, function)
            if name in ("cuEquivariance", "Torch_jet_eager", "Torch_jet_compiled"):
                report["high_precision_tails"][name] = tail_diagnostics(target)
        except Exception as error:  # noqa: BLE001 -- preserve failures as experimental outcomes
            report["failed_backends"].append(
                {
                    "backend": name,
                    "stage": "validation",
                    "type": type(error).__name__,
                    "error": str(error),
                }
            )
            specifications.pop(name)
        save()
        print({"validated": name, "accepted": name in specifications}, flush=True)
    if not args.validate_only and not args.skip_local:
        local_targets = {
            name: target
            for name, (target, _, _) in specifications.items()
            if name in ("B1", "cuEquivariance", "Torch_jet_eager", "Torch_jet_compiled")
        }
        for dim, q in ((2, 10), (3, 20)):
            for batch in (4096, 16384):
                channels = 64
                torch.manual_seed(8020 + dim + batch)
                z = (
                    torch.randn(batch, q, channels, device="cuda", dtype=torch.float64)
                    * 0.2
                )
                h, aux = native.forward(z, dim)
                d = torch.randn_like(h) * 0.2
                for operation in ("forward", "vjp"):
                    preparation_started = perf_counter()
                    functions = {
                        name: partial(target.forward, z, dim)
                        if operation == "forward"
                        else partial(target.vjp, h, aux, d, dim)
                        for name, target in local_targets.items()
                    }
                    reference = functions["B1"]()
                    checks = {}
                    for name, function in functions.items():
                        output = function()
                        checks[name] = (
                            [comparison(a, b) for a, b in zip(output, reference)]
                            if operation == "forward"
                            else comparison(output, reference)
                        )
                    captures = {
                        name: CapturedCall(function)
                        for name, function in functions.items()
                    }
                    measured = {
                        **{
                            name + "_eager": function
                            for name, function in functions.items()
                        },
                        **{
                            name + "_replay": capture.replay
                            for name, capture in captures.items()
                        },
                    }
                    row = {
                        "dimension": dim,
                        "batch": batch,
                        "channels": channels,
                        "operation": operation,
                        "correctness": checks,
                        "preparation_seconds_including_warmup_capture_and_validation": perf_counter()
                        - preparation_started,
                        "capture_seconds": {
                            name: c.capture_seconds for name, c in captures.items()
                        },
                        **measure(measured, args.repeats, 5),
                    }
                    report["local"].append(row)
                    save()
                    print(
                        {
                            "local": [dim, batch, operation],
                            "event_ms": {
                                n: v["cuda_event_ms"] for n, v in row["medians"].items()
                            },
                        },
                        flush=True,
                    )
                    torch.cuda.synchronize()
                    del functions, captures, measured, output, reference
                    gc.collect()
                    torch.cuda.empty_cache()
    for batch in [] if args.validate_only else map(int, args.batches.split(",")):
        preparation_started = perf_counter()
        case = make_case(batch)
        states = {
            name: CapturedStep(target, case, backend, function)
            for name, (target, backend, function) in specifications.items()
        }
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
            "preparation_seconds_including_warmup_capture_and_validation": perf_counter()
            - preparation_started,
            "captures": {name: state.capture for name, state in states.items()},
            **measure(functions, args.repeats, 3),
        }
        row["paired_speedup_medians"] = {
            f"{name}_over_F_{mode}": statistics.median(
                r[f"{name}_{mode}"]["cuda_event_ms"] / r[f"F_{mode}"]["cuda_event_ms"]
                for r in row["paired_observations"]
            )
            for name in states
            if name != "F"
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
        del functions, states
        gc.collect()
        torch.cuda.empty_cache()
    report["telemetry_after"] = telemetry()
    report["completed"] = True
    report["all_requested_backends_validated"] = not report["failed_backends"]
    save()


if __name__ == "__main__":
    main()
