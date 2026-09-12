"""SM120 tail-only correctness and paired resident-Graph screening.

Run with --build BUILD --output NEW.json. --preflight-only performs every
correctness check without timing. No optimizer or convergence claim is timed.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import traceback
import uuid as uuid_module
from functools import partial
from pathlib import Path
from time import perf_counter, time_ns

import numpy as np
import torch

from runtime import (Adam, NativeCUDA, Problem, ROOT, compare, device_metadata,
                     digest, make_step, parameters, snapshot, source_hashes, views)
from backends import nested_step
from flashns.jet_packed import PackedLayout, TensorActivation
from flashns.pinn_packed import PackedPINN

MODES = ("torch", "U3", "F3")
METRICS = ("cuda_event_ms_per_replay", "synchronized_wall_ms_per_replay")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def telemetry():
    return {
        "gpu": command_output(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,driver_version,"
                               "temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
                               "--format=csv,noheader,nounits"]),
        "processes": command_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                                     "--format=csv,noheader,nounits"]),
    }


def device_uuid():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = getattr(properties, "uuid", None)
    if uuid is not None:
        # PyTorch may expose a bare UUID or an NVML-prefixed GPU UUID.
        value = uuid_module.UUID(bytes=uuid) if isinstance(uuid, bytes) else uuid_module.UUID(str(uuid).removeprefix("GPU-"))
        return "GPU-" + str(value)
    # CUDA ordinals can be remapped: only fall back when exactly one UUID exists.
    result = command_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    rows = result.get("stdout", "").splitlines()
    if result.get("returncode") == 0 and len(rows) == 1:
        return rows[0].strip()
    raise RuntimeError("cannot establish selected CUDA device UUID; do not guess from its ordinal")


def require_isolation(sample, uuid):
    for name in ("gpu", "processes"):
        if sample[name].get("returncode") != 0:
            raise RuntimeError(f"GPU isolation unverified: {name} query unavailable")
    gpu_rows = list(csv.reader(sample["gpu"]["stdout"].splitlines()))
    if not any(len(row) >= 2 and row[1].strip() == uuid for row in gpu_rows):
        raise RuntimeError("GPU isolation unverified: selected UUID absent from NVML query")
    foreign = []
    for row in csv.reader(sample["processes"]["stdout"].splitlines()):
        if not row or not any(field.strip() for field in row):
            continue
        if len(row) < 4:
            raise RuntimeError("GPU isolation unverified: malformed compute-apps query")
        if row[0].strip() == uuid:
            try:
                pid = int(row[1].strip())
            except ValueError as error:
                raise RuntimeError("GPU isolation unverified: nonnumeric process PID") from error
            if pid != os.getpid():
                foreign.append(pid)
    if foreign:
        raise RuntimeError(f"screening invalid: concurrent processes on selected GPU {uuid}: {foreign}")


class TailGraph:
    """One immutable tail dispatch and private Graph pool, with supplied storage."""

    def __init__(self, problem, flat, native, mode, layout="compact", coordinate=False):
        self.flat = flat
        ni = problem.interior_count
        tail = None if mode == "torch" else partial(native.tail_dgrad_vjp, backend=mode)
        self.step = PackedPINN(
            problem.x, ni, problem.pde_weights[:ni], problem.boundary_weights[ni:],
            problem.target[ni:], layout=layout, seed="cuda", activation_factory=native.activation_factory,
            cuda_seed=native.seed, tail_dgrad_vjp=tail,
            coordinate_affine=native.coordinate_affine if coordinate else None,
        )
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        start = perf_counter()
        with torch.cuda.stream(self.stream):
            for _ in range(4):
                self.eager()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss, self.gradient = self.eager()
        self.metadata = dict(self.step.metadata, tail_mode=mode,
                             setup_seconds=perf_counter()-start,
                             parameter_data_ptr=flat.data_ptr(), graph_pool=list(self.graph.pool()),
                             stream=self.stream.cuda_stream)

    def eager(self):
        weights, biases = views(self.flat)
        loss, gradients = self.step(weights, biases)
        return loss, torch.cat([gradient.reshape(-1) for gradient in gradients])

    def replay(self):
        self.graph.replay()
        return self.loss, self.gradient


def tail(native, layout, mode, d, weight, hidden, aux):
    if mode == "torch":
        return native.activation_factory(layout).vjp(hidden, aux, d @ weight)
    return native.tail_dgrad_vjp(layout, d, weight, hidden, aux, backend=mode)


def kernel_preflight(native, quick=False):
    rows = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for dimension in (2, 3):
            for ni, nb in (((0, 0), (0, 3), (17, 5)) if quick else
                           ((0, 0), (0, 3), (1, 0), (17, 5), (129, 31))):
                for channels in ((7, 64) if quick else (7, 32, 64, 128)):
                    layout = PackedLayout(ni, nb, dimension)
                    z = torch.randn(layout.rows, channels, dtype=torch.float64, device="cuda") * 0.3
                    d = torch.randn(layout.rows, 3, dtype=z.dtype, device=z.device) * 0.3
                    weight = torch.randn(3, channels, dtype=z.dtype, device=z.device) * 0.3
                    reference = TensorActivation(layout)
                    hr, ar = reference.forward(z)
                    hidden, aux = native.activation_factory(layout).forward(z)
                    expected = reference.vjp(hr, ar, d @ weight)
                    worst = max(compare(hidden, hr), compare(aux, ar))
                    # Guard each supplied input, including offset contiguous views.
                    guarded, holders = [], []
                    for tensor in (d, weight, hidden, aux):
                        holder = torch.full((tensor.numel()+2,), 917., dtype=z.dtype, device=z.device)
                        view = holder[1:-1].view_as(tensor)
                        view.copy_(tensor)
                        guarded.append(view)
                        holders.append(holder)
                    outputs = {mode: tail(native, layout, mode, *guarded) for mode in MODES}
                    for mode, output in outputs.items():
                        worst = max(worst, compare(output, expected))
                    for view, original, holder in zip(guarded, (d, weight, hidden, aux), holders):
                        assert torch.equal(view, original), "tail modified an input"
                        assert holder[0].item() == 917. and holder[-1].item() == 917.
                    rows.append({"q": layout.q, "interior": ni, "value": nb, "channels": channels,
                                 "max_error_over_tolerance": worst})
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return {"cases": rows, "non_default_stream": True,
            "output_bounds": "exact allocations; full write bounds require compute-sanitizer"}


def kernel_graph_updates(native, quick=False):
    rows = []
    for dimension in (2, 3):
        for channels in ((7,) if quick else (7, 32, 64, 128)):
            layout = PackedLayout(17, 5, dimension)
            z = torch.randn(layout.rows, channels, dtype=torch.float64, device="cuda") * 0.3
            d = torch.randn(layout.rows, 3, dtype=z.dtype, device=z.device)
            weight = torch.randn(3, channels, dtype=z.dtype, device=z.device)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            def evaluate():
                hidden, aux = native.activation_factory(layout).forward(z)
                return {mode: tail(native, layout, mode, d, weight, hidden, aux) for mode in MODES}

            with torch.cuda.stream(stream):
                for _ in range(3):
                    evaluate()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = evaluate()
            worst, previous = 0., None
            for update in range(3):
                # These writes must be visible through the captured addresses.
                z.normal_(0., 0.3)
                d.normal_()
                weight.normal_()
                graph.replay()
                reference = TensorActivation(layout)
                hidden, aux = reference.forward(z)
                expected = reference.vjp(hidden, aux, d @ weight)
                for output in outputs.values():
                    worst = max(worst, compare(output, expected))
                current = outputs["F3"].clone()
                if previous is not None:
                    assert not torch.equal(previous, current), "Graph replay used stale inputs"
                previous = current
            rows.append({"q": layout.q, "channels": channels, "input_updates": 3,
                         "max_error_over_tolerance": worst})
    torch.cuda.synchronize()
    return rows


def module_preflight(problem, native, *, coordinate=False):
    initial = parameters(361901)
    reference = initial.numpy().copy()
    moment, variance = np.zeros_like(reference), np.zeros_like(reference)
    trajectory = []
    for update in range(1, 4):
        loss, gradient = nested_step(problem, torch.tensor(reference, dtype=torch.float64, device="cuda"))
        loss, gradient = loss.cpu().numpy(), gradient.cpu().numpy()
        moment = 0.9 * moment + 0.1 * gradient
        variance = 0.999 * variance + 0.001 * gradient * gradient
        reference -= 0.001 * (moment / (1-0.9**update)) / (np.sqrt(variance / (1-0.999**update)) + 1e-8)
        trajectory.append((loss, gradient, reference.copy()))
    rows = []
    for layout in ("dense", "split", "compact"):
        for mode in MODES:
            engine = TailGraph(problem, initial.to("cuda").clone(), native, mode, layout,
                               coordinate=coordinate)
            optimizer = Adam(engine.flat, engine.gradient)
            worst = 0.
            for expected_loss, expected_gradient, expected_parameters in trajectory:
                loss, gradient = engine.replay()
                worst = max(worst, compare(loss, expected_loss), compare(gradient, expected_gradient))
                optimizer.step()
                worst = max(worst, compare(engine.flat, expected_parameters))
            assert digest(engine.flat) != digest(initial), "Adam did not update parameter storage"
            rows.append({"layout": layout, "mode": mode, "adam_updates": 3,
                         "max_error_over_tolerance": worst, "metadata": engine.metadata})
            del optimizer, engine
    return {"problem": problem.meta, "rows": rows, "reference": "nested AD + NumPy Adam"}


def summarize(blocks, metric, bootstrap_seed):
    ratios = []
    for block in blocks:
        if any(not math.isfinite(arm[metric]) or arm[metric] <= 0 for arm in block["arms"]):
            raise ValueError("paired timing observations must be finite and positive")
        values = {label: statistics.geometric_mean(arm[metric] for arm in block["arms"] if arm["label"] == label)
                  for label in ("A", "P")}
        ratios.append(values["A"] / values["P"])
    logs = [math.log(ratio) for ratio in ratios]
    rng = random.Random(bootstrap_seed)
    samples = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(2000))
    return {"paired_block_ratios_A_over_P": ratios, "geometric_mean_ratio": math.exp(statistics.mean(logs)),
            "bootstrap_95pct_interval": [samples[49], samples[1949]],
            "independent_unit": "paired block within one resident process; exploratory interval"}


def screening(problem, native, args, report, save, locked, *, coordinate=False):
    report["screening"] = {"status": "preparing", "isolation": "unverified"}

    def isolated(sample):
        try:
            require_isolation(sample, report["device"]["uuid"])
        except RuntimeError as error:
            report["screening"].update(status="invalid", isolation="failed_or_unverified",
                                       invalid_reason=str(error), invalid_telemetry=sample)
            save()
            raise

    isolated(telemetry())
    flat = parameters(361902).to("cuda")
    weight_hash = digest(flat)
    if coordinate:
        engines = {"dense_affine": TailGraph(problem, flat, native, "torch"),
                   "coordinate_affine": TailGraph(problem, flat, native, "torch", coordinate=True)}
        pairs = (("A/A", "dense_affine", "dense_affine"),
                 ("dense/coordinate", "dense_affine", "coordinate_affine"))
    else:
        engines = {mode: TailGraph(problem, flat, native, mode) for mode in MODES}
        pairs = (("A/A", "torch", "torch"), ("torch/U3", "torch", "U3"),
                 ("torch/F3", "torch", "F3"), ("U3/F3", "U3", "F3"))
    assert len({engine.flat.data_ptr() for engine in engines.values()}) == 1
    assert len({tuple(engine.graph.pool()) for engine in engines.values()}) == len(engines)
    # Compare with the unmodified runtime entry point at the actual timing size.
    baseline = make_step(problem, layout="compact", seed="cuda", activation="cuda", native=native)
    expected_loss, gradients = baseline(*views(flat))
    expected_gradient = torch.cat([gradient.reshape(-1) for gradient in gradients])
    size_validation = {}
    for mode, engine in engines.items():
        loss, gradient = engine.replay()
        size_validation[mode] = max(compare(loss, expected_loss), compare(gradient, expected_gradient))
    locked()
    assert digest(flat) == weight_hash
    rng = random.Random(args.order_seed)
    schedule = []
    for name, baseline_mode, candidate_mode in pairs:
        patterns = ["APPA", "PAAP"] * (args.blocks // 2)
        rng.shuffle(patterns)
        schedule.extend({"comparison": name, "A": baseline_mode, "P": candidate_mode,
                         "pattern": pattern} for pattern in patterns)
    rng.shuffle(schedule)
    result = {"status": "running", "scope": "resident complete compact loss/parameter-gradient Graph screening",
              "isolation": "selected UUID checked for foreign compute processes before and after every arm",
              "optimizer_timed": False, "convergence_validated": False,
              "native_baseline_fidelity": "same callable/output checked; independent performance fidelity unverified",
              "problem": problem.meta, "parameter_sha256_before": weight_hash, "parameter_count": flat.numel(),
              "shared_read_only_storage": True, "setup": {mode: engine.metadata for mode, engine in engines.items()},
              "size_validation": size_validation, "schedule": schedule, "blocks": [],
              "simultaneous_graph_pool_allocated_bytes": torch.cuda.memory_allocated(),
              "simultaneous_graph_pool_reserved_bytes": torch.cuda.memory_reserved()}
    report["screening"] = result
    save()
    for block_index, plan in enumerate(schedule):
        block = dict(plan, index=block_index, arms=[])
        result["blocks"].append(block)
        for arm_index, label in enumerate(plan["pattern"]):
            mode = plan[label]
            before = telemetry()
            isolated(before)
            for _ in range(args.warmup):
                engines[mode].replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start_ns = time_ns()
            wall = perf_counter()
            start.record()
            for _ in range(args.replays):
                engines[mode].replay()
            end.record()
            end.synchronize()
            wall_ms = (perf_counter()-wall)*1000
            wall_end_ns = time_ns()
            event_ms = start.elapsed_time(end)
            after = telemetry()
            block["arms"].append({"arm": arm_index, "label": label, "mode": mode, "replays": args.replays,
                                  "wall_start_unix_ns": wall_start_ns, "wall_end_unix_ns": wall_end_ns,
                                  "cuda_event_total_ms": event_ms, "synchronized_wall_total_ms": wall_ms,
                                  "cuda_event_ms_per_replay": event_ms/args.replays,
                                  "synchronized_wall_ms_per_replay": wall_ms/args.replays,
                                  "telemetry_before": before, "telemetry_after": after})
            isolated(after)
        baseline_arms = [arm for arm in block["arms"] if arm["label"] == "A"]
        block["baseline_return_ratios"] = {metric: baseline_arms[1][metric]/baseline_arms[0][metric]
                                            for metric in METRICS}
        block["baseline_drift_within_5pct"] = all(abs(ratio-1) <= 0.05
                                                  for ratio in block["baseline_return_ratios"].values())
        block["completed"] = True
        save()
        print(f"screening block {block_index+1}/{len(schedule)} {plan['comparison']} {plan['pattern']}", flush=True)
    locked()
    result["parameter_sha256_after"] = digest(flat)
    assert result["parameter_sha256_after"] == weight_hash, "timing changed resident parameters"
    result["summary"] = {
        name: {metric: summarize([block for block in result["blocks"] if block["comparison"] == name],
                                 metric, args.order_seed + index)
               for metric in METRICS}
        for index, (name, _, _) in enumerate(pairs)
    }
    aa = result["summary"]["A/A"]
    result["aa_within_2pct"] = all(abs(aa[metric]["geometric_mean_ratio"]-1) <= 0.02 for metric in METRICS)
    result["all_baseline_drift_within_5pct"] = all(block["baseline_drift_within_5pct"] for block in result["blocks"])
    result["screening_minimum_gain"] = 0.01
    result["decision"] = {
        name: ("screening_signal_requires_clean_validation"
               if result["aa_within_2pct"] and result["all_baseline_drift_within_5pct"]
               and all(result["summary"][name][metric]["bootstrap_95pct_interval"][0] > 1.01
                                                  for metric in METRICS)
               else "no_clear_screening_signal")
        for name, _, _ in pairs if name != "A/A"
    }
    result["status"] = "completed"
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--kernel-only", action="store_true",
                        help="with --preflight-only: bounded native checks for compute-sanitizer")
    parser.add_argument("--blocks", type=int, default=12, help="even paired blocks per comparison, at least six")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=89241)
    args = parser.parse_args()
    if args.kernel_only and not args.preflight_only:
        parser.error("--kernel-only requires --preflight-only; it cannot authorize timing")
    if args.blocks < 6 or args.blocks % 2 or args.replays < 1 or args.warmup < 1:
        parser.error("even --blocks >= 6 and positive --replays/--warmup required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write('{"completed": false, "preflight_passed": false}\n')
    report = {"completed": False, "preflight_passed": False, "checks": [], "screening": None,
              "contract": {"atol": 2e-12, "rtol": 2e-11, "modes": MODES,
                           "layout": "compact", "seed": "cuda", "activation": "native stable FP64",
                           "workload": "2048 interior + 512 boundary; 2-64-64-3",
                           "weight_seed_preflight": 361901, "weight_seed_screening": 361902,
                           "order_seed": args.order_seed, "paired_blocks_per_comparison": args.blocks,
                           "warmup_replays_every_arm": args.warmup, "replays_per_arm": args.replays,
                           "minimum_gain": 0.01, "AA_ratio_deviation_limit": 0.02,
                           "baseline_return_deviation_limit": 0.05,
                           "drift_policy": "retain every raw block; any drift failure prevents a positive signal",
                           "sanitizers": "separate compute-sanitizer invocation required",
                           "kernel_only": args.kernel_only,
                           "evidence_level": "screening only; no solver convergence or clean PR validation"}}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("this run requires an SM120 CUDA GPU")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(83109)
        native = NativeCUDA(args.build)
        library = args.build / native.metadata["library"]
        report.update(device=dict(device_metadata(), uuid=device_uuid()), source_hashes=snapshot(args.output),
                      build=native.metadata, binary_path=str(library.resolve()),
                      binary_sha256=sha256(library), build_metadata_sha256=sha256(args.build / "build.json"),
                      process={"pid": os.getpid(), "python": sys.executable, "argv": sys.argv,
                               "modules": {name: str(Path(sys.modules[name].__file__).resolve())
                                           for name in ("runtime", "native", "backends", "flashns.pinn_packed")},
                               "git_head": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                               "git_status": command_output(["git", "-C", str(ROOT), "status", "--short"]),
                               "environment": {name: os.environ[name] for name in
                                               ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS")
                                               if name in os.environ}}, telemetry_start=telemetry())

        def locked():
            if (source_hashes() != report["source_hashes"] or sha256(library) != report["binary_sha256"]
                    or sha256(args.build / "build.json") != report["build_metadata_sha256"]):
                raise RuntimeError("source/build changed during this process; timings invalid")

        problem = Problem(interior_count=2048, edge_count=128, validation_count=64)
        checks = [("Q10_Q20_tail_boundaries_nondefault_stream",
                   lambda: kernel_preflight(native, quick=args.kernel_only)),
                  ("tail_graph_input_updates",
                   lambda: kernel_graph_updates(native, quick=args.kernel_only))]
        if not args.kernel_only:
            checks.append(("all_layouts_three_actual_adam_updates_vs_nested_AD",
                           lambda: module_preflight(problem, native)))
        for name, check in checks:
            report["active_check"] = name
            save()
            detail = check()
            report["checks"].append({"name": name, "passed": True, "detail": detail})
            save()
            print(name, "passed", flush=True)
        locked()
        del report["active_check"]
        report["preflight_passed"] = True
        save()
        if not args.preflight_only:
            screening(problem, native, args, report, save, locked)
        report["completed"] = True
        report["telemetry_end"] = telemetry()
        save()
    except Exception as error:
        if report.get("screening") is not None:
            report["screening"].update(status="invalid", invalid_reason=str(error))
        report["failure"] = {"error": str(error), "traceback": traceback.format_exc()}
        save()
        raise


if __name__ == "__main__":
    main()
