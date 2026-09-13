"""Controlled eager/graph ablation of the same full FP64 gradient.

Capture pools and static inputs/outputs remain owned for every replay. Graph
capture and numerical checks are outside performance measurements.
"""

import argparse
import gc
import json
import statistics
from datetime import UTC, datetime
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

BACKENDS = ("B1", "U", "F", "F3")


class CapturedStep:
    def __init__(self, native, case, backend, function=step):
        self.native, self.backend, self.function = native, backend, function
        self.x, self.q, self.weights, self.biases, self.meta = local_case(
            case, 1, 0, "cuda:0"
        )
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(5):
                self.eager()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        start = perf_counter()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.outputs = self.eager()
        self.capture = {
            "seconds": perf_counter() - start,
            "allocated_delta_bytes": torch.cuda.memory_allocated() - allocated,
            "reserved_delta_bytes": torch.cuda.memory_reserved() - reserved,
            "scope": "capture-only delta; excludes preexisting static inputs, warmup cache and other live graphs",
        }

    def eager(self):
        return self.function(
            self.x,
            self.q,
            self.meta["global_denominator"],
            self.weights,
            self.biases,
            self.native,
            self.backend,
        )

    def replay(self):
        self.graph.replay()
        return self.outputs

    def reference(self, nested=False):
        if nested:
            return nested_reference(
                self.x,
                self.q,
                self.meta["global_denominator"],
                self.weights,
                self.biases,
            )
        return step(
            self.x,
            self.q,
            self.meta["global_denominator"],
            self.weights,
            self.biases,
            self.native,
            "B1",
        )


def check_outputs(actual, expected):
    return {
        "loss": comparison(actual[0], expected[0]),
        "parameter_gradient": comparison(
            torch.cat([p.flatten() for p in actual[1]]),
            torch.cat([p.flatten() for p in expected[1]]),
        ),
    }


def validate_updates(native):
    rows = []
    case = make_case(5, 7, weighted=True)
    for backend in BACKENDS:
        captured = CapturedStep(native, case, backend)
        steps = []
        for index in range(4):
            with torch.no_grad():
                captured.x.add_(0.001)
            actual = captured.replay()
            expected = captured.reference(nested=True)
            check = check_outputs(actual, expected)
            old = [p.clone() for p in captured.weights + captured.biases]
            with torch.no_grad():
                for parameter, gradient in zip(
                    captured.weights + captured.biases, actual[1]
                ):
                    parameter.add_(gradient, alpha=-1e-3)
            check["updated_parameters"] = comparison(
                torch.cat([p.flatten() for p in captured.weights + captured.biases]),
                torch.cat([(p - 1e-3 * g).flatten() for p, g in zip(old, expected[1])]),
            )
            steps.append({"iteration": index, **check})
        rows.append(
            {
                "backend": backend,
                "checks": steps,
                "input_and_parameter_changes_checked": True,
            }
        )
        torch.cuda.synchronize()
        del captured
    return {
        "reference": "independent nested stable AD at each changed input/parameter state",
        "cases": rows,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--batches", default="4096,16384,65536,262144")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    native = Native()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes(),
        "build": native.build,
        "telemetry_before": telemetry(),
        "numerical_updates": validate_updates(native),
        "includes_optimizer_in_timing": False,
        "includes_independent_validation_in_timing": False,
        "includes_graph_capture_in_timing": False,
        "graph_pool_sharing": False,
        "cases": [],
    }
    for batch in [int(value) for value in args.batches.split(",")]:
        case = make_case(batch)
        states = {name: CapturedStep(native, case, name) for name in BACKENDS}
        checks = {
            name: check_outputs(state.replay(), state.reference())
            for name, state in states.items()
        }
        functions = {
            f"{name}_{mode}": getattr(state, mode)
            for name, state in states.items()
            for mode in ("eager", "replay")
        }
        row = {
            "case": case[4],
            "correctness": checks,
            "captures": {name: state.capture for name, state in states.items()},
            **measure(functions, args.repeats, 5),
        }
        row["paired_speedup_medians"] = {
            f"{name}_eager_over_replay": statistics.median(
                r[f"{name}_eager"]["cuda_event_ms"]
                / r[f"{name}_replay"]["cuda_event_ms"]
                for r in row["paired_observations"]
            )
            for name in BACKENDS
        }
        for numerator, denominator in (("B1", "F"), ("U", "F"), ("B1", "F3")):
            for mode in ("eager", "replay"):
                row["paired_speedup_medians"][
                    f"{numerator}_over_{denominator}_{mode}"
                ] = statistics.median(
                    r[f"{numerator}_{mode}"]["cuda_event_ms"]
                    / r[f"{denominator}_{mode}"]["cuda_event_ms"]
                    for r in row["paired_observations"]
                )
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            {
                "batch": batch,
                "event_ms": {
                    name: values["cuda_event_ms"]
                    for name, values in row["medians"].items()
                },
            },
            flush=True,
        )
        torch.cuda.synchronize()
        del functions, states
        gc.collect()
        torch.cuda.empty_cache()
    report["telemetry_after"] = telemetry()
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
