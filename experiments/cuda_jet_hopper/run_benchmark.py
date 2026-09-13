"""Screen a fixed grid, then independently retest matched copy families."""

import argparse
import gc
import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from build import name
from hopper import Hopper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cuda_jet_h100"))
from benchmark import measure, telemetry
from common import comparison, make_case, nested_reference
from gpu import Native
from graph_experiment import CapturedStep, check_outputs


class OriginalF(Native):
    def dgrad(self, d, weight, hidden, aux, dim, backend="F"):
        mode = (
            backend + "3" if weight.shape[0] == 3 and backend in ("U", "F") else backend
        )
        return super().dgrad(d, weight, hidden, aux, dim, mode)


class Capture:
    def __init__(self, function):
        self.function = function
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4):
                function()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        start = perf_counter()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = function()
        self.capture_seconds = perf_counter() - start

    def replay(self):
        self.graph.replay()
        return self.output


def local_inputs(batch, dim, channels, baseline, seed):
    torch.manual_seed(seed)
    q = 10 if dim == 2 else 20
    z = torch.randn(batch, q, channels, device="cuda", dtype=torch.float64) * 0.2
    h, aux = baseline.forward(z, dim)
    d = torch.randn_like(h) * 0.2
    w = torch.randn(channels, channels, device="cuda", dtype=torch.float64) * 0.2
    return d, w, h, aux


def timing(functions, repeats, inner):
    result = measure(functions, repeats, inner)
    rows = result["paired_observations"]
    result["relative_to_B1"] = {
        name: statistics.median(
            row["B1"]["cuda_event_ms"] / row[name]["cuda_event_ms"] for row in rows
        )
        for name in functions
        if name != "B1"
    }
    if "sync_F" in functions:
        result["paired_copy_speedups"] = {
            "sync_over_cp_async": statistics.median(
                row["sync_F"]["cuda_event_ms"] / row["cp_async_F"]["cuda_event_ms"]
                for row in rows
            ),
            "sync_over_tma": statistics.median(
                row["sync_F"]["cuda_event_ms"] / row["tma_F"]["cuda_event_ms"]
                for row in rows
            ),
            "cp_async_over_tma": statistics.median(
                row["cp_async_F"]["cuda_event_ms"] / row["tma_F"]["cuda_event_ms"]
                for row in rows
            ),
        }
    return result


def screen(directory, build, baseline, save_row):
    records = []
    for dim in (2, 3):
        data = local_inputs(16384, dim, 64, baseline, 91331 + dim)
        for identifier in build["libraries"]:
            native = Hopper(directory, identifier)
            functions = {
                "B1": partial(baseline.dgrad, *data, dim, "B1"),
                "F": partial(native.dgrad, *data, dim, "F"),
            }
            captures = {key: Capture(function) for key, function in functions.items()}
            comparison(captures["F"].replay(), captures["B1"].replay())
            result = timing(
                {key: capture.replay for key, capture in captures.items()}, 3, 15
            )
            row = {
                "configuration": identifier,
                "dimension": dim,
                "batch": 16384,
                "channels": 64,
                "graph_timing": result,
                "resources": native.resources(),
            }
            records.append(row)
            save_row(row)
            del captures, functions
            native.close()
            del native
        del data
        gc.collect()
        torch.cuda.empty_cache()
    return records


def local_retest(dim, batch, channels, baseline, original, matched, repeats):
    data = local_inputs(batch, dim, channels, baseline, 91673 + dim + batch + channels)
    functions = {
        "B1": partial(baseline.dgrad, *data, dim, "B1"),
        "original_F": partial(original.dgrad, *data, dim, "F"),
    }
    for label, native in matched.items():
        for mode in ("U", "F"):
            functions[label + "_" + mode] = partial(native.dgrad, *data, dim, mode)
    with torch.no_grad():
        reference = functions["B1"]()
        checks = {
            label: comparison(function(), reference)
            for label, function in functions.items()
        }
        captures = {label: Capture(function) for label, function in functions.items()}
        result = {
            "dimension": dim,
            "batch": batch,
            "channels": channels,
            "correctness": checks,
            "capture_seconds": {
                key: value.capture_seconds for key, value in captures.items()
            },
            "graph": timing(
                {key: value.replay for key, value in captures.items()}, repeats, 10
            ),
            "eager": timing(functions, repeats, 5),
        }
    return result


def network_retest(batch, baseline, original, matched, repeats):
    case = make_case(batch, weighted=True, seed=91743)
    states = {
        "B1": CapturedStep(baseline, case, "B1"),
        "original_F": CapturedStep(original, case, "F"),
    }
    for label, native in matched.items():
        states[label + "_F"] = CapturedStep(native, case, "F")
    states["tma_U"] = CapturedStep(matched["tma"], case, "U")
    reference = states["B1"].replay()
    checks = {
        key: check_outputs(state.replay(), reference) for key, state in states.items()
    }
    return {
        "batch": batch,
        "case": case[-1],
        "correctness": checks,
        "capture": {key: state.capture for key, state in states.items()},
        "graph": timing(
            {key: state.replay for key, state in states.items()}, repeats, 5
        ),
        "eager": timing(
            {key: state.eager for key, state in states.items()}, repeats, 3
        ),
        "optimizer_included": False,
        "cout3_tail": "identical U3/F3 fallback for all matched copies",
    }


def validate_graph_updates(matched):
    result = {}
    for label, native in matched.items():
        state = CapturedStep(native, make_case(9, weighted=True, seed=91761), "F")
        rows = []
        for iteration in range(4):
            actual = state.replay()
            expected = nested_reference(
                state.x,
                state.q,
                state.meta["global_denominator"],
                state.weights,
                state.biases,
            )
            rows.append({"iteration": iteration, **check_outputs(actual, expected)})
            with torch.no_grad():
                for p, g in zip(state.weights + state.biases, actual[1]):
                    p.add_(g, alpha=-1e-3)
        result[label] = rows
        del state
        native.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite performance results")
    build = json.loads((args.build / "build.json").read_text())
    validation = json.loads((args.build / "validation.json").read_text())
    sanitizers = json.loads((args.build / "sanitizers/sanitizers.json").read_text())
    if not validation["passed"] or not sanitizers["all_passed"]:
        raise RuntimeError("validation and every sanitizer must pass before timing")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    baseline, original = Native(), OriginalF()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.glob("*.py")
        },
        "build_sha256": hashlib.sha256(
            (args.build / "build.json").read_bytes()
        ).hexdigest(),
        "sanitizer_report_sha256": hashlib.sha256(
            (args.build / "sanitizers/sanitizers.json").read_bytes()
        ).hexdigest(),
        "telemetry_before": telemetry(),
        "selection_rule": "Fixed 48-configuration screen at B=16384,Cin=Cout=64. Select the fastest TMA F graph per dimension; retest all three copy modes with exactly its M,N,stages,lifetime and swizzle. Screen uses 3 repeats; final retest uses 15 new paired repeats and fresh data.",
        "screen": [],
        "local": [],
        "complete_gradient": [],
        "passed": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def screen_row(row):
        report["screen"].append(row)
        save()

    save()
    with torch.no_grad():
        screen(args.build, build, baseline, screen_row)
    report["selected"] = {}
    for dim in (2, 3):
        candidates = [
            row
            for row in report["screen"]
            if row["dimension"] == dim
            and build["libraries"][row["configuration"]]["config"]["copy"] == 2
        ]
        winner = min(
            candidates,
            key=lambda row: row["graph_timing"]["medians"]["F"]["cuda_event_ms"],
        )
        config = dict(build["libraries"][winner["configuration"]]["config"])
        matched = {
            label: Hopper(args.build, name({**config, "copy": copy}))
            for copy, label in enumerate(("sync", "cp_async", "tma"))
        }
        report["selected"][str(dim)] = {
            label: name(native.config) for label, native in matched.items()
        }
        print({"dimension": dim, "selected": report["selected"][str(dim)]}, flush=True)
        save()
        if dim == 2:
            report["graph_update_validation"] = validate_graph_updates(matched)
        for channels in (32, 64):
            for batch in (4096, 16384, 65536, 262144):
                row = local_retest(
                    dim, batch, channels, baseline, original, matched, args.repeats
                )
                report["local"].append(row)
                save()
                print(
                    {
                        "local": [dim, batch, channels],
                        "graph_speedups": row["graph"]["relative_to_B1"],
                    },
                    flush=True,
                )
                for native in matched.values():
                    native.close()
                gc.collect()
                torch.cuda.empty_cache()
        if dim == 2:
            for batch in (4096, 16384, 65536):
                row = network_retest(batch, baseline, original, matched, args.repeats)
                report["complete_gradient"].append(row)
                save()
                print(
                    {
                        "network": batch,
                        "graph_speedups": row["graph"]["relative_to_B1"],
                    },
                    flush=True,
                )
                for native in matched.values():
                    native.close()
                gc.collect()
                torch.cuda.empty_cache()
        report.setdefault("plan_statistics", {})[str(dim)] = {
            label: native.plan_statistics for label, native in matched.items()
        }
        del matched
    report["telemetry_after"] = telemetry()
    report["passed"] = True
    report["finished_utc"] = datetime.now(UTC).isoformat()
    save()


if __name__ == "__main__":
    main()
