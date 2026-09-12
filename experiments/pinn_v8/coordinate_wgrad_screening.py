"""G0/G1 terminal coordinate-wgrad screening after full GPU preflight.

Both arms retain all forward GEMMs, complete native stable VJPs and torch tail.
The comparison changes only first-layer weight/bias contraction and its reduction order.
"""

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path
from time import perf_counter, time_ns

import torch

from runtime import (NativeCUDA, Problem, compare, device_metadata, digest,
                     make_step, parameters, snapshot, source_hashes, views)
from flashns.pinn_packed import PackedPINN
from tail_sm120 import (METRICS, device_uuid, require_isolation, sha256,
                        summarize, telemetry)

MODES = ("gemm", "coordinate")


class CoordinateWgradGraph:
    def __init__(self, problem, flat, native, mode):
        assert mode in MODES
        self.flat = flat
        ni = problem.interior_count
        self.step = PackedPINN(
            problem.x, ni, problem.pde_weights[:ni], problem.boundary_weights[ni:],
            problem.target[ni:], layout="compact", seed="cuda",
            activation_factory=native.activation_factory, cuda_seed=native.seed,
            coordinate_wgrad=native.coordinate_wgrad if mode == "coordinate" else None,
        )
        start = perf_counter()
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(4):
                self.eager()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss, self.gradient = self.eager()
        self.metadata = dict(self.step.metadata, wgrad_mode=mode,
                             setup_seconds=perf_counter()-start,
                             parameter_data_ptr=flat.data_ptr(), graph_pool=list(self.graph.pool()))

    def eager(self):
        loss, gradients = self.step(*views(self.flat))
        return loss, torch.cat([gradient.reshape(-1) for gradient in gradients])

    def replay(self):
        self.graph.replay()
        return self.loss, self.gradient


def run_screening(problem, native, args, report, save, locked):
    uuid = report["device"]["uuid"]
    require_isolation(telemetry(), uuid)
    flat = parameters(431902).to("cuda")
    weight_hash = digest(flat)
    engines = {mode: CoordinateWgradGraph(problem, flat, native, mode) for mode in MODES}
    assert len({engine.flat.data_ptr() for engine in engines.values()}) == 1
    assert len({tuple(engine.graph.pool()) for engine in engines.values()}) == 2
    baseline = make_step(problem, layout="compact", seed="cuda", activation="cuda", native=native)
    loss, gradients = baseline(*views(flat))
    gradient = torch.cat([item.reshape(-1) for item in gradients])
    size_validation = {}
    for mode, engine in engines.items():
        actual_loss, actual_gradient = engine.replay()
        size_validation[mode] = max(compare(actual_loss, loss), compare(actual_gradient, gradient))
    pairs = (("A/A", "gemm", "gemm"), ("gemm/coordinate", "gemm", "coordinate"))
    schedule, rng = [], random.Random(args.order_seed)
    for name, a, p in pairs:
        patterns = ["APPA", "PAAP"] * (args.blocks // 2)
        rng.shuffle(patterns)
        schedule.extend({"comparison": name, "A": a, "P": p, "pattern": pattern} for pattern in patterns)
    rng.shuffle(schedule)
    result = {"status": "running", "scope": "complete compact loss/parameter-gradient Graph replay",
              "matched_control": "library GEMM + bias sum; complete native activation VJP in both arms",
              "tail_mode": "torch in both arms", "optimizer_timed": False,
              "convergence_validated": False, "cross_process_validation": False,
              "problem": problem.meta, "parameter_sha256_before": weight_hash,
              "parameter_count": flat.numel(), "shared_read_only_parameter_storage": True,
              "setup": {mode: engine.metadata for mode, engine in engines.items()},
              "size_validation": size_validation, "schedule": schedule, "blocks": [],
              "simultaneous_graph_pool_allocated_bytes": torch.cuda.memory_allocated(),
              "simultaneous_graph_pool_reserved_bytes": torch.cuda.memory_reserved()}
    report["screening"] = result
    locked()
    assert digest(flat) == weight_hash
    save()
    for index, plan in enumerate(schedule):
        block = dict(plan, index=index, arms=[])
        result["blocks"].append(block)
        for arm_index, label in enumerate(plan["pattern"]):
            before = telemetry()
            require_isolation(before, uuid)
            engine = engines[plan[label]]
            for _ in range(args.warmup):
                engine.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_ns, wall = time_ns(), perf_counter()
            start.record()
            for _ in range(args.replays):
                engine.replay()
            end.record()
            end.synchronize()
            wall_ms, end_ns = (perf_counter()-wall)*1000, time_ns()
            event_ms = start.elapsed_time(end)
            after = telemetry()
            block["arms"].append({"arm": arm_index, "label": label, "mode": plan[label],
                                  "replays": args.replays, "wall_start_unix_ns": start_ns,
                                  "wall_end_unix_ns": end_ns, "cuda_event_total_ms": event_ms,
                                  "synchronized_wall_total_ms": wall_ms,
                                  "cuda_event_ms_per_replay": event_ms/args.replays,
                                  "synchronized_wall_ms_per_replay": wall_ms/args.replays,
                                  "telemetry_before": before, "telemetry_after": after})
            require_isolation(after, uuid)
        arms = [arm for arm in block["arms"] if arm["label"] == "A"]
        block["baseline_return_ratios"] = {metric: arms[1][metric]/arms[0][metric] for metric in METRICS}
        block["baseline_drift_within_5pct"] = all(abs(value-1) <= 0.05
                                                  for value in block["baseline_return_ratios"].values())
        block["completed"] = True
        save()
        print(f"coordinate wgrad block {index+1}/{len(schedule)} {plan['comparison']} {plan['pattern']}", flush=True)
    locked()
    result["parameter_sha256_after"] = digest(flat)
    assert result["parameter_sha256_after"] == weight_hash, "timing changed resident parameters"
    result["summary"] = {
        name: {metric: summarize([block for block in result["blocks"] if block["comparison"] == name],
                                 metric, args.order_seed+i) for metric in METRICS}
        for i, (name, _, _) in enumerate(pairs)
    }
    result["aa_within_2pct"] = all(abs(result["summary"]["A/A"][metric]["geometric_mean_ratio"]-1) <= 0.02
                                   for metric in METRICS)
    result["all_baseline_drift_within_5pct"] = all(block["baseline_drift_within_5pct"] for block in result["blocks"])
    result["decision"] = (
        "screening_signal_requires_clean_validation"
        if result["aa_within_2pct"] and result["all_baseline_drift_within_5pct"]
        and all(result["summary"]["gemm/coordinate"][metric]["bootstrap_95pct_interval"][0] > 1.01 for metric in METRICS)
        else "no_clear_screening_signal")
    result["status"] = "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=91531)
    args = parser.parse_args()
    if args.blocks < 6 or args.blocks % 2 or args.replays < 1 or args.warmup < 1:
        parser.error("even --blocks >= 6 and positive --replays/--warmup required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write('{"completed": false}\n')
    report = {"completed": False, "screening": None,
              "contract": {"candidate": "coordinate_wgrad_G1", "atol": 2e-12, "rtol": 2e-11,
                           "weight_seed": 431902, "order_seed": args.order_seed,
                           "paired_blocks_per_comparison": args.blocks,
                           "warmup_replays_every_arm": args.warmup, "replays_per_arm": args.replays,
                           "minimum_gain": 0.01, "AA_ratio_deviation_limit": 0.02,
                           "baseline_return_deviation_limit": 0.05,
                           "drift_policy": "retain every raw block; any drift failure prevents a positive signal",
                           "evidence_level": "resident screening; no solver convergence or clean PR validation"}}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("SM120 GPU required")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        native = NativeCUDA(args.build)
        library = args.build / native.metadata["library"]
        uuid = device_uuid()
        report.update(device=dict(device_metadata(), uuid=uuid), source_hashes=snapshot(args.output),
                      binary_sha256=sha256(library), build_metadata_sha256=sha256(args.build/"build.json"),
                      build=native.metadata, preflight_path=str(args.preflight.resolve()),
                      preflight_sha256=sha256(args.preflight), telemetry_start=telemetry(),
                      process={"pid": os.getpid(), "python": sys.executable, "argv": sys.argv,
                               "environment": {key: os.environ[key] for key in
                                               ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS")
                                               if key in os.environ}})
        preflight = json.loads(args.preflight.read_text())
        if (not preflight.get("completed") or not preflight.get("preflight_passed")
                or preflight.get("contract", {}).get("kernel_only") is not False
                or preflight.get("contract", {}).get("candidate") != "coordinate_wgrad_G1"
                or preflight.get("source_hashes") != report["source_hashes"]
                or preflight.get("binary_sha256") != report["binary_sha256"]
                or preflight.get("device", {}).get("uuid") != uuid):
            raise RuntimeError("full coordinate-wgrad preflight for exact source, binary and GPU required")

        def locked():
            if (source_hashes() != report["source_hashes"] or sha256(library) != report["binary_sha256"]
                    or sha256(args.build/"build.json") != report["build_metadata_sha256"]
                    or sha256(args.preflight) != report["preflight_sha256"]):
                raise RuntimeError("source/build/preflight changed; screening invalid")

        problem = Problem(interior_count=2048, edge_count=128, validation_count=64)
        run_screening(problem, native, args, report, save, locked)
        report["telemetry_end"] = telemetry()
        report["completed"] = True
        save()
    except Exception as error:
        if report.get("screening") is not None:
            report["screening"].update(status="invalid", invalid_reason=str(error))
        report["failure"] = {"error": str(error), "traceback": traceback.format_exc()}
        save()
        raise


if __name__ == "__main__":
    main()
