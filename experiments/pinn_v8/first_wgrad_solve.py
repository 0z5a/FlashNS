"""The frozen Kovasznay solve with v8 execution paths and unchanged stopping rules."""

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import torch

from runtime import (Adam, Engine, EvaluationActivation, NativeCUDA, Problem, device_metadata,
                     digest, ensure_preflight, evaluate, frozen_protocol, independent_metrics,
                     parameters, snapshot, source_hashes, symbolic_check)
import csv
import subprocess
import uuid as uuid_module

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



def accepted(metrics, thresholds):
    return all(math.isfinite(metrics[name]) and metrics[name] <= limit for name, limit in thresholds.items())


def json_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--first-wgrad-mode", choices=("torch", "coordinate", "fused"), default="torch")
    parser.add_argument("--first-preflight", type=Path)
    parser.add_argument("--layout", choices=("dense", "split", "compact"), default="compact")
    parser.add_argument("--seed-mode", choices=("autograd", "explicit", "compiled", "cuda"), default="cuda")
    parser.add_argument("--activation", choices=("cuda", "tensor", "compiled"), default="cuda")
    parser.add_argument("--seed", type=int, default=361901)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        raise RuntimeError("refusing to overwrite a solve")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    native = NativeCUDA(args.build)
    uuid = device_uuid()
    telemetry_start = telemetry()
    require_isolation(telemetry_start, uuid)
    preflight = ensure_preflight(args.preflight, native, activation=args.activation)
    if preflight.get("quick"):
        raise RuntimeError("full preflight required")
    if args.first_wgrad_mode != "torch":
        if args.first_preflight is None:
            parser.error("candidate solve requires --first-preflight")
        first_report = json.loads(args.first_preflight.read_text())
        required = "all_layouts_three_actual_adam_updates_vs_nested_AD"
        if (not first_report.get("completed") or not first_report.get("preflight_passed")
                or first_report.get("contract", {}).get("kernel_only") is not False
                or first_report.get("contract", {}).get("candidate") != ("coordinate_wgrad_G1" if args.first_wgrad_mode == "coordinate" else "coordinate_activation_wgrad_G2")
                or first_report.get("device", {}).get("uuid") != uuid
                or first_report.get("source_hashes") != source_hashes()
                or first_report.get("binary_sha256") != native.metadata["library_sha256"]
                or not any(c["name"] == required and c["passed"] for c in first_report.get("checks", []))):
            raise RuntimeError("candidate requires exact-source full first-wgrad preflight")
    config = frozen_protocol(args.protocol)
    # Objective/stopping fields are inherited exactly; selection/seeds are separate.
    original = frozen_protocol()
    for key in ("problem", "interior_count", "edge_count", "validation_count", "data_seed", "thresholds",
                "adam_steps", "adam_learning_rate", "adam_decay_every", "adam_decay_factor", "adam_check_every",
                "adam_betas", "adam_epsilon", "lbfgs_max_blocks", "lbfgs_max_iter_per_block", "lbfgs_max_eval_per_block",
                "lbfgs_history_size", "lbfgs_line_search", "lbfgs_lr", "lbfgs_tolerance_grad", "lbfgs_tolerance_change"):
        if config[key] != original[key]:
            raise RuntimeError(f"changed scientific contract: {key}; register a separate experiment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"completed": False, "seed": args.seed, "layout": args.layout, "seed_mode": args.seed_mode,
              "activation": args.activation, "first_wgrad_mode": args.first_wgrad_mode, "protocol": config, "source_hashes": snapshot(args.output),
              "device": dict(device_metadata(), uuid=uuid),
              "process": {"pid": os.getpid(), "python": sys.executable, "argv": sys.argv, "runtime_module": str(Path(sys.modules["runtime"].__file__).resolve()), "native_module": str(Path(sys.modules["native"].__file__).resolve())},
              "binary_sha256": native.metadata["library_sha256"], "build": native.metadata,
              "telemetry_start": telemetry_start, "evidence_level": "full_first_wgrad_solver_observation_requires_paired_validation",
              "symbolic_exact_solution_check": symbolic_check(), "progress": []}

    def save():
        args.output.write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+"\n")

    preparation = perf_counter()
    problem = Problem(**{key: config[key] for key in ("interior_count", "edge_count", "validation_count", "data_seed")})
    initial = parameters(args.seed)
    torch.cuda.synchronize()
    common_preparation = perf_counter()-preparation
    setup_start = perf_counter()
    engine_options = {} if args.first_wgrad_mode == "torch" else {"first_wgrad_mode": args.first_wgrad_mode}
    engine = Engine(problem, initial, layout=args.layout, seed=args.seed_mode, activation=args.activation, native=native, **engine_options)
    evaluator = EvaluationActivation(native)
    adam = Adam(engine.flat, engine.gradient, config["adam_learning_rate"])
    torch.cuda.synchronize()
    setup_seconds = perf_counter()-setup_start
    report.update(problem=problem.meta, common_preparation_seconds=common_preparation, backend_setup=engine.metadata)
    rows = report["progress"]
    adam_updates = lbfgs_iterations = 0
    converged, independent = False, None
    optimization_start = perf_counter()

    def check(phase):
        nonlocal converged, independent
        metrics = evaluate(problem, engine.flat, evaluator)
        candidate = accepted(metrics, config["thresholds"])
        if candidate:
            independent = independent_metrics(problem, engine.flat)
            converged = accepted(independent, config["thresholds"])
        rows.append({"phase": phase, "adam_updates": adam_updates, "lbfgs_iterations": lbfgs_iterations,
                     "loss_gradient_evaluations": engine.closure_calls, "metrics": metrics,
                     "independent_metrics": independent if candidate else None,
                     "converged": converged, "optimization_wall_seconds": perf_counter()-optimization_start})
        save()
        if converged or phase == "initial" or (len(rows) % 20 == 0):
            print({"phase": phase, "adam": adam_updates, "lbfgs": lbfgs_iterations, "metrics": metrics, "converged": converged}, flush=True)

    check("initial")
    for update in range(1, config["adam_steps"]+1):
        if converged:
            break
        if (update-1) % config["adam_decay_every"] == 0:
            adam.lr.fill_(config["adam_learning_rate"] * config["adam_decay_factor"] ** ((update-1)//config["adam_decay_every"]))
        engine.replay()
        adam.step()
        adam_updates += 1
        if update % config["adam_check_every"] == 0:
            check("adam")
    optimizer = torch.optim.LBFGS([engine.flat], lr=config["lbfgs_lr"], history_size=config["lbfgs_history_size"],
                                 max_iter=config["lbfgs_max_iter_per_block"], max_eval=config["lbfgs_max_eval_per_block"],
                                 tolerance_grad=config["lbfgs_tolerance_grad"], tolerance_change=config["lbfgs_tolerance_change"],
                                 line_search_fn=config["lbfgs_line_search"])
    for _ in range(config["lbfgs_max_blocks"]):
        if converged:
            break
        optimizer.step(engine.closure)
        lbfgs_iterations = optimizer.state[engine.flat].get("n_iter", 0)
        check("lbfgs")
        if not all(math.isfinite(value) for value in rows[-1]["metrics"].values()):
            break
    torch.cuda.synchronize()
    optimization_seconds = perf_counter()-optimization_start
    final = engine.flat.detach().cpu().clone()
    result = {"converged": converged, "setup_seconds": setup_seconds, "optimization_wall_seconds": optimization_seconds,
              "non_finite_metric_encoding": "non-finite metrics are strings (nan/inf/-inf), never a successful acceptance",
              "total_wall_seconds": common_preparation+setup_seconds+optimization_seconds,
              "adam_updates": adam_updates, "lbfgs_iterations": lbfgs_iterations,
              "loss_gradient_evaluations": engine.closure_calls, "metrics": rows[-1]["metrics"],
              "independent_metrics": independent, "parameter_initial_sha256": digest(initial),
              "parameter_final_sha256": digest(final),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    checkpoint = args.output.with_suffix(".pt")
    torch.save({"flat_parameters": final, "problem": problem.meta, "protocol": config, "result": result}, checkpoint)
    report["telemetry_end"] = telemetry()
    require_isolation(report["telemetry_end"], uuid)
    if hashlib.sha256((args.build / native.metadata["library"]).read_bytes()).hexdigest() != report["binary_sha256"]:
        raise RuntimeError("binary changed during solve")
    if source_hashes() != report["source_hashes"]:
        raise RuntimeError("source changed during solve")
    report.update(result=result, completed=True, checkpoint={"file": checkpoint.name, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    save()
    print(json.dumps(result, indent=2), flush=True)
    del optimizer, adam, engine
    gc.collect()
    torch.cuda.empty_cache()
    # A budget-limited solve is a completed observation, not a crash or a hidden success.


if __name__ == "__main__":
    main()
