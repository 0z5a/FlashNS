"""Finite continuation of the frozen two-seed normalized development problem.

Source, scales and independent holdouts are fixed. GPU objective evaluation may
use one compute-only CUDA graph; optimizers run outside it. This is development,
not a frozen formal protocol or a performance comparison. Intermediate Adam
hashes are recorded, but only declared checkpoints retain full gradient arrays.
Every actual L-BFGS closure retains its complete candidate and gradient.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.data import FAMILIES, views, torch_data
from box376.frozen_data import load_cache
from box376.normalized_objective import (
    PreparedNormalizedObjective, PreparedNormalizedADObjective, normalization_key,
)

TOPOLOGY = (4, 128, 128, 128, 128, 4)
SEEDS = (361901, 361902)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Reference shape/dtype mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite observation")
    error = np.abs(actual-expected)
    result = {"entries": actual.size,
              "maximum_absolute_error": float(np.max(error, initial=0)),
              "maximum_error_over_budget": float(np.max(error/(1e-11+1e-9*np.abs(expected)), initial=0))}
    if result["maximum_error_over_budget"] > 1:
        raise AssertionError(result)
    return result


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--adam-final", type=int, default=4000)
    parser.add_argument("--lbfgs-calls", type=int, default=10)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--lbfgs-closure-cap", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--check-every", type=int, default=100)
    args = parser.parse_args()
    if (args.adam_final <= 100 or args.lbfgs_calls < 0 or args.lbfgs_max_iter < 1
            or args.lbfgs_closure_cap < 1 or args.chunk < 1 or args.check_every < 1):
        raise ValueError("Invalid finite development budget")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        if torch.cuda.device_count() != 1:
            raise ValueError("Exactly one coordinator-granted physical GPU required")
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        if "H100" not in properties.name:
            raise ValueError("This packet targets the qualified single H100")
    calibration = json.loads((args.inputs/"calibration.json").read_text())
    if file_hash(args.inputs/"calibration.npz") != calibration["raw_sha256"]:
        raise ValueError("Calibration bytes changed")
    key = normalization_key(calibration)
    cache = load_cache(args.inputs/"training-source.npz")
    if [len(cache["families"][n]["points"]) for n in FAMILIES] != [4096, 512, 384, 7]:
        raise ValueError("Unexpected training family counts")
    data = torch_data(cache)
    data["normalization_key"] = key
    for family in data["families"].values():
        for name, value in list(family.items()):
            if isinstance(value, torch.Tensor):
                family[name] = value.to(device)
    engine = PreparedNormalizedObjective(data, calibration)
    reference = PreparedNormalizedADObjective(data, calibration, "a24")
    started = time.perf_counter()
    report = {"state": "RUNNING", "development_only": True, "device": str(device),
              "physical_GPU_count": int(device.type == "cuda"), "precision": "FP64",
              "topology": list(TOPOLOGY), "parameter_count": 50692,
              "torch_version": torch.__version__, "compiled_CUDA": torch.version.cuda,
              "compute_only_CUDA_graph": device.type == "cuda",
              "optimizer_inside_graph": False, "source_depends_on_parameters": False,
              "supervised_interior_velocity_term": False, "NCCL_used": False,
              "CPU_fallback_used": False, "formal_training_protocol_frozen": False,
              "training_converged": False, "time_to_accuracy_measured": False,
              "normalization_key": key, "training_cache_key": cache["metadata"]["cache_key"],
              "source_version": 0, "nu": data["nu"], "eta": data["eta"], "lambda_g": data["lambda_g"],
              "resume_absolute_Adam_update": 100, "Adam_final_absolute_update_budget": args.adam_final,
              "Adam_new_updates_per_seed_budget": args.adam_final-100,
              "Adam_lr_rule": "1e-3 * 0.5**floor(absolute_update/1000)",
              "Adam_betas": [.9, .999], "Adam_eps": 1e-8, "gradient_clipping": False,
              "Adam_foreach": False, "Adam_weight_decay": 0,
              "lbfgs_calls_per_seed_budget": args.lbfgs_calls, "lbfgs_max_iter": args.lbfgs_max_iter,
              "lbfgs_max_eval": 25, "lbfgs_closure_cap_per_seed": args.lbfgs_closure_cap,
              "lbfgs_history_size": 20, "lbfgs_lr": 1, "lbfgs_line_search": "strong_wolfe",
              "lbfgs_tolerance_grad": 0, "lbfgs_tolerance_change": 0,
              "chunk": args.chunk, "Adam_complete_gradient_check_interval": args.check_every,
              "every_intermediate_Adam_full_gradient_retained": False,
              "every_LBFGS_candidate_and_full_gradient_retained": True,
              "input_files": {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)}
                              for p in sorted(args.inputs.iterdir()) if p.is_file()},
              "groups": []}
    if device.type == "cuda":
        report.update(GPU_name=properties.name, GPU_total_bytes=properties.total_memory)
    events_path = args.output/"events.jsonl"

    def progress():
        report["process_wall_seconds"] = time.perf_counter()-started
        write_json(args.output/"report.json", report)

    try:
        with events_path.open("x", buffering=1) as events:
            for seed in SEEDS:
                resume_path = args.inputs/f"seed{seed}-Adam-update100.pt"
                resume = torch.load(resume_path, map_location="cpu", weights_only=True)
                assert resume["actual_Adam_updates"] == 100 and resume["source_version"] == 0
                assert resume["training_cache_key"] == report["training_cache_key"]
                assert resume["normalization_key"] == key
                assert resume["theta"].shape == (50692,) and resume["theta"].dtype == torch.float64
                theta = torch.nn.Parameter(resume["theta"].to(device).clone())
                parameters = views(theta.detach(), TOPOLOGY)
                pointer = theta.data_ptr()
                adam = torch.optim.Adam([theta], lr=1e-3, betas=(.9, .999), eps=1e-8, foreach=False)
                adam.load_state_dict(resume["optimizer_state"])
                adam.param_groups[0]["foreach"] = False
                group = {"seed": seed, "state": "RUNNING", "resume_parameter_sha256": array_hash(resume["theta"].numpy()),
                         "optimizer_state_loaded_from_actual_checkpoint": True,
                         "actual_new_Adam_updates": 0, "actual_LBFGS_calls": 0,
                         "actual_LBFGS_closures": 0, "retained_evaluations": [],
                         "raw_shards": [], "checkpoints": [], "Adam_complete_gradient_check_updates": []}
                report["groups"].append(group)
                retained = {}
                graph = None
                if device.type == "cuda":
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            warm = engine(parameters, data, args.chunk)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    del warm
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        graph_result = engine(parameters, data, args.chunk)
                    group["graph_captured"] = True

                def flush_shard(tag):
                    if not retained:
                        return
                    path = args.output/f"seed{seed}-{tag}.npz"
                    np.savez_compressed(path, **retained)
                    group["raw_shards"].append({"filename": path.name, "bytes": path.stat().st_size,
                                                "sha256": file_hash(path), "arrays": len(retained)})
                    retained.clear()
                    progress()

                def checkpoint(tag, optimizer, phase, absolute_update):
                    path = args.output/f"seed{seed}-{tag}.pt"
                    torch.save({"theta": theta.detach().cpu().clone(),
                                "optimizer_state": cpu_tree(optimizer.state_dict()), "optimizer": phase,
                                "actual_Adam_updates": absolute_update,
                                "actual_LBFGS_calls": group["actual_LBFGS_calls"],
                                "actual_LBFGS_closures": group["actual_LBFGS_closures"],
                                "training_cache_key": report["training_cache_key"],
                                "normalization_key": key, "source_version": 0}, path)
                    item = {"filename": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path),
                            "parameter_sha256": array_hash(theta.detach().cpu().numpy()),
                            "optimizer": phase, "absolute_Adam_update": absolute_update}
                    group["checkpoints"].append(item)
                    return item

                def evaluate(phase, index, retain, check):
                    prefix = f"{phase}-{index}"
                    candidate = theta.detach().cpu().numpy().copy()
                    record = {"seed": seed, "phase": phase, "index": index,
                              "parameter_sha256": array_hash(candidate), "full_arrays_retained": retain,
                              "state": "CANDIDATE_SAVED_BEFORE_COMPUTE"}
                    if retain:
                        retained[prefix+"_theta"] = candidate
                        group["retained_evaluations"].append(record)
                    assert theta.data_ptr() == pointer
                    if graph is None:
                        result = engine(parameters, data, args.chunk)
                    else:
                        graph.replay()
                        result = graph_result
                    actual = [v.detach().cpu().numpy().copy() for v in result]
                    for name, value in zip(("loss", "gradient", "numerators"), actual):
                        if not np.isfinite(value).all():
                            raise ValueError("Nonfinite "+name)
                        if retain:
                            retained[prefix+"_"+name] = value
                    record.update(state="ACTUAL_RESULT_OBSERVED", loss=float(actual[0]),
                                  gradient_sha256=array_hash(actual[1]), numerators=actual[2].tolist(),
                                  gradient_l2=float(np.linalg.norm(actual[1])))
                    # Save the observed result even when its independent gate fails.
                    events.write(json.dumps(record, allow_nan=False)+"\n")
                    if check:
                        expected = reference(parameters, data, args.chunk)
                        expected_cpu = [v.detach().cpu().numpy().copy() for v in expected]
                        record["independent_A24"] = {name: compare(a, b) for name, a, b in
                                                     zip(("loss", "gradient", "numerators"), actual, expected_cpu)}
                        for name, value in zip(("loss", "gradient", "numerators"), expected_cpu):
                            retained[prefix+"_a24_"+name] = value
                    return result, record

                try:
                    for update in range(100, args.adam_final+1):
                        selected = update == 100 or update == args.adam_final or update % args.check_every == 0
                        result, record = evaluate("Adam", update, selected, selected)
                        if selected:
                            group["Adam_complete_gradient_check_updates"].append(update)
                            if update == 100 or update == args.adam_final or update % 1000 == 0:
                                record["optimizer_checkpoint"] = checkpoint(f"Adam-{update}", adam, "Adam", update)
                            print(json.dumps({"seed": seed, "phase": "Adam", "absolute_updates": update,
                                              "loss": record["loss"], "elapsed_seconds": time.perf_counter()-started}), flush=True)
                            flush_shard(f"Adam-{update}")
                        if update < args.adam_final:
                            adam.param_groups[0]["lr"] = 1e-3*.5**(update//1000)
                            adam.zero_grad(set_to_none=True)
                            theta.grad = result[1].detach().clone()
                            adam.step()
                            group["actual_new_Adam_updates"] += 1
                        if update % args.check_every == 0:
                            progress()
                    group["Adam_final_loss"] = record["loss"]
                    if args.lbfgs_calls:
                        optimizer = torch.optim.LBFGS([theta], lr=1, max_iter=args.lbfgs_max_iter,
                                                      max_eval=25, history_size=20, tolerance_grad=0,
                                                      tolerance_change=0, line_search_fn="strong_wolfe")
                        checkpoint("LBFGS-initial", optimizer, "LBFGS", args.adam_final)

                        def closure():
                            if group["actual_LBFGS_closures"] >= args.lbfgs_closure_cap:
                                raise RuntimeError("Frozen finite L-BFGS closure cap exhausted")
                            index = group["actual_LBFGS_closures"]
                            group["actual_LBFGS_closures"] += 1
                            optimizer.zero_grad(set_to_none=True)
                            result, _ = evaluate("LBFGS", index, True, True)
                            theta.grad = result[1].detach().clone()
                            return result[0]

                        for call in range(args.lbfgs_calls):
                            optimizer.step(closure)
                            group["actual_LBFGS_calls"] += 1
                            path = args.output/f"seed{seed}-LBFGS-accepted-{call+1}.npy"
                            np.save(path, theta.detach().cpu().numpy())
                            group.setdefault("accepted_LBFGS_parameters", []).append({
                                "filename": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path),
                                "parameter_sha256": array_hash(theta.detach().cpu().numpy())})
                            print(json.dumps({"seed": seed, "phase": "LBFGS", "accepted_calls": call+1,
                                              "actual_closures": group["actual_LBFGS_closures"]}), flush=True)
                            flush_shard(f"LBFGS-call-{call+1}")
                        result, record = evaluate("final", 0, True, True)
                        group["final_loss"] = record["loss"]
                        record["optimizer_checkpoint"] = checkpoint("LBFGS-final", optimizer, "LBFGS", args.adam_final)
                        flush_shard("final")
                    else:
                        group["final_loss"] = group["Adam_final_loss"]
                    group.update(state="FINITE_DEVELOPMENT_SEED_FINISHED",
                                 parameter_storage_stable=theta.data_ptr() == pointer,
                                 normalization_constant_storage_stable=tuple(v.data_ptr() for v in engine.constants) == engine.constant_addresses)
                finally:
                    flush_shard("partial-finally")
                    if group["state"] != "FINITE_DEVELOPMENT_SEED_FINISHED":
                        checkpoint("failure-Adam-state", adam, "Adam", 100+group["actual_new_Adam_updates"])
                    if graph is not None:
                        del graph, graph_result
                        torch.cuda.synchronize()
                progress()
        report.update(state="FINITE_NORMALIZED_DEVELOPMENT_CONTINUATION_FINISHED",
                      actual_new_Adam_updates=sum(g["actual_new_Adam_updates"] for g in report["groups"]),
                      actual_LBFGS_calls=sum(g["actual_LBFGS_calls"] for g in report["groups"]),
                      actual_LBFGS_closures=sum(g["actual_LBFGS_closures"] for g in report["groups"]))
    except BaseException:
        report.update(state="FAILED", traceback=traceback.format_exc())
        raise
    finally:
        report["events_sha256"] = file_hash(events_path) if events_path.exists() else None
        report["runner_sha256"] = file_hash(__file__)
        progress()


if __name__ == "__main__":
    main()
