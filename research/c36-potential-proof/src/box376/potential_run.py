"""Execute a frozen C36 timing or training protocol; keep actual raw results."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.data import FAMILIES, flatten, initialize, torch_data, views
from box376.frozen_data import load_cache
from box376.normalized_objective import normalization_key
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.verify_potential_gradient import compare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_problem(args, protocol):
    assert sha(args.inputs / "training-source.npz") == protocol["source_sha256"]
    assert sha(args.inputs / "calibration.npz") == protocol["calibration_sha256"]
    calibration = json.loads((args.inputs / "calibration.json").read_text())
    data = torch_data(load_cache(args.inputs / "training-source.npz"))
    data["normalization_key"] = normalization_key(calibration)
    for family in data["families"].values():
        for key, value in family.items():
            if isinstance(value, torch.Tensor):
                family[key] = value.to("cuda")
    assert [len(data["families"][name]["points"]) for name in FAMILIES] == protocol["family_counts"]
    return data, calibration


def timed_call(function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def capture(function):
    """Capture warmed compute on a side stream; optimizer steps stay outside."""
    started = time.perf_counter()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = function()

    def replay():
        graph.replay()
        return result

    return replay, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("timing", "training"), required=True)
    parser.add_argument("--baseline", choices=("taylor_parameter_ad", "native_coordinate_ad"), default="taylor_parameter_ad")
    parser.add_argument("--eager", action="store_true", help="Train without graph capture in a fresh process.")
    parser.add_argument("--graphs", action="store_true", help="Use the separately frozen paired graph-timing protocol.")
    parser.add_argument("--train-engine", choices=("explicit", "taylor_ad"), default="explicit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    if args.mode == "timing":
        assert args.graphs == protocol["performance"]["CUDA_graphs_in_measured_windows"]
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report = {"state": "RUNNING", "mode": args.mode, "baseline": args.baseline,
              "protocol_sha256": sha(args.protocol), "protocol": protocol,
              "started_unix": time.time(), "torch": torch.__version__, "numpy": np.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "groups": [], "setup_seconds": {}, "training_converged": False,
              "physical_acceptance_evaluated": False, "complete_solver_speedup": None}
    baseline_training = args.mode == "training" and args.train_engine == "taylor_ad"
    report.update(training_engine=args.train_engine,
                  gradient_actual_prefix="taylor_ad" if baseline_training else "explicit",
                  gradient_reference_prefix="explicit" if baseline_training else "reference")

    def progress(phase):
        report.update(phase=phase, wall_seconds=time.perf_counter() - started)
        write_json(args.output / "report.json", report)
        print(json.dumps({"phase": phase, "wall_seconds": report["wall_seconds"]}), flush=True)

    def save_arrays(name, arrays):
        path = args.output / (name + ".npz")
        np.savez_compressed(path, **arrays)
        return {"filename": path.name, "sha256": sha(path), "arrays": len(arrays)}

    try:
        data, calibration = load_problem(args, protocol)
        explicit, report["setup_seconds"]["explicit"] = timed_call(
            lambda: PreparedPotentialNormalizedObjective(data, calibration))
        reference, report["setup_seconds"]["reference"] = timed_call(
            lambda: PreparedPotentialADObjective(data, calibration,
                coordinate_ad=args.mode == "timing" and args.baseline == "native_coordinate_ad"))
        chunk = protocol["explicit_chunk"]
        reference_chunk = protocol["coordinate_ad_chunk"] if args.mode == "timing" and args.baseline == "native_coordinate_ad" else protocol["taylor_ad_chunk"]
        training_engine = reference if baseline_training else explicit
        comparison_engine = explicit if baseline_training else reference
        comparison_chunk = chunk if baseline_training else reference_chunk
        topology = tuple(protocol["topology"])
        progress("prepared")
        for seed_index, seed in enumerate(protocol["seeds"]):
            theta = torch.nn.Parameter(flatten(initialize(topology, seed)).to("cuda"))
            assert theta.numel() == protocol["parameter_count"]
            parameters = views(theta.detach(), topology)
            group = {"seed": seed, "state": "RUNNING", "checks": [], "raw_files": []}
            report["groups"].append(group)

            def check(tag, actual=None):
                progress(f"seed{seed}-{tag}-started")
                candidate = training_engine(parameters, chunk) if actual is None else actual
                expected = comparison_engine(parameters, comparison_chunk)
                arrays = {"theta": cpu(theta)}
                checks = []
                for name, a, b in zip(("loss", "gradient", "numerators"), candidate, expected):
                    aa, bb = cpu(a), cpu(b)
                    checks.append({"name": name, **compare(aa, bb)})
                    arrays[report["gradient_actual_prefix"] + "_" + name] = aa
                    arrays[report["gradient_reference_prefix"] + "_" + name] = bb
                group["checks"].append({"tag": tag, "comparisons": checks})
                group["raw_files"].append(save_arrays(f"seed{seed}-{tag}", arrays))
                progress(f"seed{seed}-{tag}")
                return candidate

            if args.mode == "timing":
                spec = protocol["performance"]
                initial = check("initial-gradient")
                evaluators = {"explicit": lambda: explicit(parameters, chunk),
                              "reference": lambda: reference(parameters, reference_chunk)}
                if args.graphs:
                    group["graph_qualification"] = {}
                    original = theta.detach().clone()
                    for name in ("explicit", "reference"):
                        if name == "reference" and args.baseline == "native_coordinate_ad":
                            continue
                        replay, seconds = capture(evaluators[name])
                        with torch.no_grad():
                            theta.add_(.001)
                        check("changed-parameter-" + name + "-graph", replay())
                        with torch.no_grad():
                            theta.copy_(original)
                        actual = replay()
                        for a, b in zip(actual, initial):
                            compare(cpu(a), cpu(b))
                        evaluators[name] = replay
                        group["graph_qualification"][name] = {"passed": True, "capture_seconds": seconds,
                                                               "original_and_changed_parameters_checked": True}
                for _ in range(2):
                    timed_call(evaluators["explicit"])
                for _ in range(1 if args.baseline == "native_coordinate_ad" else 2):
                    timed_call(evaluators["reference"])
                group["pairs"] = []
                for index in range(spec["pairs_per_seed_per_baseline"]):
                    order = ("explicit", "reference") if (index + seed_index) % 2 == 0 else ("reference", "explicit")
                    pair = {"index": index, "order": list(order)}
                    group["active_pair"] = pair
                    for name in order:
                        progress(f"seed{seed}-pair{index}-{name}-started")
                        durations, elapsed = [], 0.
                        native = name == "reference" and args.baseline == "native_coordinate_ad"
                        minimum = spec["native_window_evaluations"] if native else spec["window_minimum_evaluations"]
                        while len(durations) < minimum or (not native and elapsed < spec["window_minimum_seconds"]):
                            result, duration = timed_call(evaluators[name])
                            assert all(torch.isfinite(value).all().item() for value in result)
                            durations.append(duration)
                            elapsed += duration
                            del result
                        pair[name] = {"evaluation_seconds": durations, "mean_seconds": float(np.mean(durations))}
                        progress(f"seed{seed}-pair{index}-{name}")
                    pair["speedup"] = pair["reference"]["mean_seconds"] / pair["explicit"]["mean_seconds"]
                    group["pairs"].append(pair)
                    group.pop("active_pair")
                    progress(f"seed{seed}-pair{index}-complete")
                check("final-gradient", evaluators["explicit"]())
            else:
                spec = protocol["training"]
                group.update(Adam_updates=0, LBFGS_calls=0, LBFGS_closures=0, checkpoints=[])
                result = check("Adam0")
                group["initial_loss"] = float(result[0])
                graph, graph_result = None, None
                if args.eager:
                    group["compute_graph_qualified"] = False
                    group["fresh_process_eager_requested"] = True
                else:
                    try:
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                training_engine(parameters, chunk)
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            graph_result = training_engine(parameters, chunk)
                        original = theta.detach().clone()
                        with torch.no_grad():
                            theta.add_(.001)
                        graph.replay()
                        changed = tuple(value.clone() for value in graph_result)
                        torch.cuda.synchronize()
                        check("changed-parameter-graph", changed)
                        with torch.no_grad():
                            theta.copy_(original)
                        graph.replay()
                        torch.cuda.synchronize()
                        for a, b in zip(graph_result, result):
                            compare(cpu(a), cpu(b))
                        group["compute_graph_qualified"] = True
                    except Exception:
                        group["compute_graph_qualified"] = False
                        group["graph_failure"] = traceback.format_exc()
                        if "original" in locals():
                            with torch.no_grad():
                                theta.copy_(original)
                        # A failed capture can leave the allocator in capture mode.
                        # Preserve it and exit; any eager fallback needs a fresh process.
                        raise RuntimeError("CUDA graph qualification failed; restart with a fresh CUDA context")
                optimizer = torch.optim.Adam([theta], lr=1e-3, betas=tuple(spec["Adam_betas"]),
                    eps=spec["Adam_eps"], weight_decay=0, foreach=False)

                def evaluate():
                    if graph is None:
                        return training_engine(parameters, chunk)
                    graph.replay()
                    return graph_result

                def checkpoint(tag):
                    path = args.output / f"seed{seed}-{tag}.pt"
                    torch.save({"theta": theta.detach().cpu(), "optimizer_state": optimizer.state_dict(),
                        "protocol_sha256": report["protocol_sha256"], "representation": "C36-curl-potential",
                        "Adam_updates": group["Adam_updates"], "LBFGS_calls": group["LBFGS_calls"],
                        "LBFGS_closures": group["LBFGS_closures"], "training_engine": args.train_engine}, path)
                    group["checkpoints"].append({"filename": path.name, "sha256": sha(path)})

                with (args.output / f"seed{seed}-updates.jsonl").open("x", buffering=1) as events:
                    checkpoint("initial")
                    for update in range(1, spec["Adam_updates_per_seed"] + 1):
                        tick = time.perf_counter()
                        loss, gradient, numerators = evaluate()
                        assert torch.isfinite(loss).item() and torch.isfinite(gradient).all().item()
                        optimizer.param_groups[0]["lr"] = 1e-3 * .5 ** ((update - 1) // 1000)
                        theta.grad = gradient.detach().clone()
                        event = {"phase": "Adam", "update": update, "loss_before_update": float(loss),
                                 "gradient_infinity": float(gradient.abs().max()), "lr": optimizer.param_groups[0]["lr"]}
                        optimizer.step()
                        group["Adam_updates"] = update
                        event["update_wall_seconds"] = time.perf_counter() - tick
                        events.write(json.dumps(event, allow_nan=False) + "\n")
                        if update in spec["complete_AD_gradient_check_Adam_updates"]:
                            check(f"Adam{update}", evaluate())
                        if update % spec["checkpoint_interval"] == 0:
                            checkpoint(f"Adam{update}")
                            group["latest_loss_before_update"] = event["loss_before_update"]
                            progress(f"seed{seed}-Adam{update}")
                    optimizer = torch.optim.LBFGS([theta], lr=spec["LBFGS_lr"], max_iter=spec["LBFGS_max_iter"],
                        max_eval=spec["LBFGS_max_eval"], history_size=spec["LBFGS_history_size"],
                        tolerance_grad=0, tolerance_change=0, line_search_fn=spec["LBFGS_line_search"])

                    def closure():
                        if group["LBFGS_closures"] >= spec["LBFGS_closure_cap_per_seed"]:
                            raise RuntimeError("Frozen LBFGS closure budget exhausted")
                        candidate = cpu(theta)
                        loss, gradient, numerators = evaluate()
                        assert torch.isfinite(loss).item() and torch.isfinite(gradient).all().item()
                        index = group["LBFGS_closures"]
                        group["raw_files"].append(save_arrays(f"seed{seed}-LBFGS{index}",
                            {"theta": candidate, "loss": cpu(loss), "gradient": cpu(gradient), "numerators": cpu(numerators)}))
                        group["LBFGS_closures"] += 1
                        theta.grad = gradient.detach().clone()
                        events.write(json.dumps({"phase": "LBFGS", "closure": index, "loss": float(loss)}) + "\n")
                        return loss.detach().clone()

                    for call in range(spec["LBFGS_calls_per_seed"]):
                        optimizer.step(closure)
                        group["LBFGS_calls"] = call + 1
                        checkpoint(f"LBFGS-call{call + 1}")
                        progress(f"seed{seed}-LBFGS-call{call + 1}")
                    final = check("final-gradient", evaluate())
                    group.update(final_loss=float(final[0]), final_gradient_infinity=float(final[1].abs().max()))
                    group["training_converged"] = group["final_loss"] <= spec["loss_convergence_limit"] and group["final_gradient_infinity"] <= spec["gradient_infinity_convergence_limit"]
                    checkpoint("final")
            group["state"] = "COMPLETE"
            progress(f"seed{seed}-complete")
        if args.mode == "timing":
            logarithms = np.log([p["speedup"] for g in report["groups"] for p in g["pairs"]])
            spec = protocol["performance"]
            rng = np.random.default_rng(spec["bootstrap_seed"])
            samples = np.exp(rng.choice(logarithms, (spec["bootstrap_resamples"], len(logarithms))).mean(1))
            report.update(paired_geometric_mean_speedup=float(np.exp(logarithms.mean())),
                          paired_bootstrap_95_percent_interval=np.quantile(samples, [.025, .975]).tolist())
        else:
            report["training_converged"] = all(group["training_converged"] for group in report["groups"])
        report["state"] = "COMPLETE"
    except BaseException:
        report.update(state="FAILED", failure=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        progress(report["state"])


if __name__ == "__main__":
    main()
