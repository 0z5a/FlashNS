"""Prospective C36 TTA trainer, derived from the frozen v7 optimizer recipe.
New seeds use fresh full dense parameter AD and a fixed native coordinate-AD subset.
Observation pauses training while preserving optimizer state; successful physical validation stops.
Historical v7 sources and protocols are unchanged.
"""
import time
PROCESS_STARTED = time.perf_counter()
PROCESS_STARTED_UNIX = time.time()

import argparse
import gc
import hashlib
import json
import time
import subprocess
import sys
import copy
import traceback
from pathlib import Path

import numpy as np
import torch

from torch_c36_packed import activate_packed
from torch_c36_packed_vjp import activate_packed_vjp
from box376.data import initialize, flatten, views
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.potential_run import capture, cpu, load_problem, sha, write_json
from box376.ssbroyden import SSBroyden
from box376.tta_control import ActiveClock, run_observer

REFERENCE_SHA = "172174ea49a82a21e0616571ae303560d275317d8b5319c22ea49dbd628d8b66"
KINDS = ("loss", "gradient", "numerators")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("fresh", "continue"), required=True)
    parser.add_argument("--start-checkpoint", type=Path, help="v5 final checkpoint (continue arm)")
    parser.add_argument("--reference", type=Path, help="preserved full native-AD reference (.npz)")
    parser.add_argument("--legacy-triton-target", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="CPU, eager, tiny budget; checks the control flow only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    spec = protocol["training"]
    assert args.seed in protocol["seeds"]
    args.engine = "analytic"
    start_spec = protocol["starts"].get(f"continue:{args.seed}") if args.arm == "continue" else None
    assert (args.arm == "continue") == (args.start_checkpoint is not None) == (start_spec is not None)
    device = "cpu" if args.self_test else "cuda"
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not args.self_test:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert args.reference is None, "New seeds must not use old initial parameter references"
    started = PROCESS_STARTED
    report = {"state": "RUNNING", "seed": args.seed, "protocol_sha256": sha(args.protocol),
              "protocol_version": protocol["version"], "self_test": args.self_test,
              "torch": torch.__version__, "numpy": np.__version__,
              "gpu": None if args.self_test else torch.cuda.get_device_name(0),
              "arm": args.arm, "optimizer": spec["optimizer"], "start_checkpoint": start_spec,
              "engine": args.engine, "engine_description": ("packed activation + analytic transpose"
                  if args.engine == "analytic" else "parameter AD over the packed activation")
                  + ("" if args.self_test else ", compiled, CUDA Graph"),
              "checks": [], "checkpoints": [], "setup_seconds": {}, "Adam_updates": 0,
              "SSBroyden_iterations": 0, "SSBroyden_evaluations": 0, "resets": 0, "skipped_updates": 0,
              "training_gate_first": None,
              "budget_truncated": False, "started_unix": PROCESS_STARTED_UNIX}
    events = (args.output / "events.jsonl").open("x", buffering=1)

    def progress(phase):
        report.update(phase=phase, wall_seconds=time.perf_counter() - started)
        write_json(args.output / "report.json", report)
        print(json.dumps({"phase": phase, "wall_seconds": round(report["wall_seconds"], 1)}), flush=True)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def check(tag, actual, expected, reference_name):
        arrays = {}
        row = {"tag": tag, "reference": reference_name, "comparisons": []}
        for kind, a, b in zip(KINDS, actual, expected):
            a, b = cpu(a), (cpu(b) if isinstance(b, torch.Tensor) else np.asarray(b).copy())
            assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
            assert np.isfinite(a).all() and np.isfinite(b).all()
            error = float(np.max(np.abs(a - b) / (protocol["atol"] + protocol["rtol"] * np.abs(b))))
            row["comparisons"].append({"name": kind, "entries": int(a.size),
                                       "maximum_error_over_original_budget": error})
            arrays[kind + "_actual"], arrays[kind + "_reference"] = a, b
            assert error <= 1, (tag, kind, error)
        arrays["theta"] = cpu(theta)
        path = args.output / f"check-{tag}.npz"
        np.savez_compressed(path, **arrays)
        row.update(filename=path.name, sha256=sha(path))
        report["checks"].append(row)
        progress(f"check-{tag}")

    def checkpoint(tag, values=None):
        path = args.output / f"seed{args.seed}-{args.arm}-{tag}.pt"
        torch.save({"theta": (theta.detach().cpu().clone() if values is None else values),
                    "protocol_sha256": report["protocol_sha256"], "representation": "C36-curl-potential",
                    "Adam_updates": report["Adam_updates"], "SSBroyden_iterations": report["SSBroyden_iterations"],
                    "SSBroyden_evaluations": report["SSBroyden_evaluations"], "training_engine": report["engine"],
                    "arm": args.arm}, path)
        report["checkpoints"].append({"tag": tag, "filename": path.name, "sha256": sha(path),
                                      "training_seconds": training_clock.elapsed()
                                      if training_started else 0.0})

    training_started = None
    training_clock = None
    try:
        data, calibration = load_problem(args, protocol) if not args.self_test else _cpu_problem(args, protocol)
        topology = tuple(protocol["topology"])
        reference = None
        if args.reference is not None:
            with np.load(args.reference, allow_pickle=False) as raw:
                reference = {k: raw[k].copy() for k in raw.files if k.startswith(f"seed{args.seed}_")}
        initial = flatten(initialize(topology, args.seed))
        if reference is not None:
            stored = torch.from_numpy(reference[f"seed{args.seed}_theta"])
            report["initialize_matches_stored_bytes"] = bool(torch.equal(initial, stored))
            initial = stored.clone()
        theta = torch.nn.Parameter(initial.to(device))
        assert theta.numel() == protocol["parameter_count"]
        report["initial_theta_sha256"] = hashlib.sha256(cpu(theta).tobytes()).hexdigest()
        params = views(theta.detach(), topology)
        native = None
        chunk = protocol["explicit_chunk"]

        t = time.perf_counter()
        dense_ad = PreparedPotentialADObjective(data, calibration)
        if args.engine == "analytic":
            eager = PreparedPotentialNormalizedObjective(data, calibration, activation=activate_packed,
                                                         activation_vjp=activate_packed_vjp)
        else:
            eager = PreparedPotentialADObjective(data, calibration, activation=activate_packed)
        report["setup_seconds"]["eager_and_dense_AD"] = time.perf_counter() - t
        original = theta.detach().clone()
        native = tuple(v.clone() for v in dense_ad(params, chunk))
        report["initial_reference"] = "fresh full dense parameter AD; separate native subset below"
        if not args.self_test:
            subset = copy.deepcopy(data)
            family = subset["families"]["interior"]
            index = torch.tensor(protocol["qualification"]["native_subset_indices"], device=device)
            for key, value in family.items():
                if isinstance(value, torch.Tensor):
                    family[key] = value[index].clone()
            subset_actual = PreparedPotentialNormalizedObjective(subset, calibration, activation=activate_packed, activation_vjp=activate_packed_vjp)
            subset_native = PreparedPotentialADObjective(subset, calibration, coordinate_ad=True)
            check("fresh-native-subset", subset_actual(params, 64), subset_native(params, 64), "fresh native coordinate AD, fixed subset; full parameter gradient")
            del subset_actual, subset_native, subset
        if native is not None:
            check("eager-fresh-denseAD", eager(params, chunk), native, "fresh full dense parameter AD")
        with torch.no_grad():
            theta.add_(.001)
        check("eager-changed-denseAD", eager(params, chunk), dense_ad(params, chunk), "dense parameter AD")
        with torch.no_grad():
            theta.copy_(original)

        if args.self_test:
            engine, evaluate = eager, None
        else:
            if args.legacy_triton_target:
                from box376.triton_target_compat import enable_legacy_target_compat
                report["compiler_compatibility"] = enable_legacy_target_compat()
            activate_packed(torch.zeros(128, 128, 36, device=device))
            forward = torch.compile(activate_packed, fullgraph=True, dynamic=False)
            transpose = torch.compile(activate_packed_vjp, fullgraph=True, dynamic=False)
            t = time.perf_counter()
            if args.engine == "analytic":
                engine = PreparedPotentialNormalizedObjective(data, calibration, activation=forward,
                                                              activation_vjp=transpose)
            else:
                engine = PreparedPotentialADObjective(data, calibration, activation=forward)
            for _ in range(2):
                engine(params, chunk)
            sync()
            report["setup_seconds"]["compile_and_warmup"] = time.perf_counter() - t
            check("compiled-fresh-denseAD", engine(params, chunk), native, "fresh full dense parameter AD")
            replay, report["setup_seconds"]["graph_capture"] = capture(lambda: engine(params, chunk))
            check("graph-fresh-denseAD", replay(), native, "fresh full dense parameter AD")
            with torch.no_grad():
                theta.add_(.001)
            check("graph-changed-denseAD", replay(), dense_ad(params, chunk), "dense parameter AD")
            with torch.no_grad():
                theta.copy_(original)
            evaluate = replay
        if evaluate is None:
            evaluate = lambda: engine(params, chunk)

        if args.arm == "continue":
            # Start checkpoint: file and parameter hashes must match the protocol.
            report["start_checkpoint"]["observed_checkpoint_sha256"] = sha(args.start_checkpoint)
            if report["start_checkpoint"]["observed_checkpoint_sha256"] != start_spec["checkpoint_sha256"]:
                raise RuntimeError("start checkpoint sha256 differs from the protocol; refusing to start")
            stored = torch.load(args.start_checkpoint, map_location="cpu", weights_only=True)["theta"]
            if hashlib.sha256(stored.numpy().tobytes()).hexdigest() != start_spec["theta_sha256"]:
                raise RuntimeError("start parameter sha256 differs from the protocol; refusing to start")
            with torch.no_grad():
                theta.copy_(stored.to(device))
            start = theta.detach().clone()
            check("start-denseAD", tuple(v.clone() for v in evaluate()), dense_ad(params, chunk), "dense parameter AD at the start")
            with torch.no_grad():
                theta.add_(.001)
            check("start-changed-denseAD", tuple(v.clone() for v in evaluate()), dense_ad(params, chunk), "dense parameter AD after a change")
            with torch.no_grad():
                theta.copy_(start)
            loss0 = cpu(evaluate()[0].clone())
            error = float(np.max(np.abs(loss0 - start_spec["source_final_loss"])
                                 / (protocol["atol"] + protocol["rtol"] * abs(start_spec["source_final_loss"]))))
            report["continuity"] = {"start_loss": float(loss0), "source_final_loss": start_spec["source_final_loss"],
                                    "maximum_error_over_original_budget": error}
            assert error <= 1, ("continuity", error)

        def full_check(tag):
            values = tuple(v.clone() for v in evaluate())
            check(tag, values, dense_ad(params, chunk), "dense parameter AD")

        adam_updates = 0 if args.arm == "continue" else (3 if args.self_test else spec["Adam_updates_per_seed"])
        max_seconds = spec["maximum_training_seconds"]
        iteration_cap = 3 if args.self_test else spec["SSBroyden_iteration_cap"]
        checkpoint_iterations = set(spec["checkpoint_SSBroyden_iterations"])
        check_every = 2 if args.self_test else spec["complete_gradient_check_every_SSBroyden_iterations"]
        optimizer = torch.optim.Adam([theta], lr=1e-3, betas=tuple(spec["Adam_betas"]),
                                     eps=spec["Adam_eps"], weight_decay=0, foreach=False)
        progress("training-started")
        training_started = time.perf_counter()
        training_clock = ActiveClock()
        checkpoint("initial")
        full_check("Adam0" if args.arm == "fresh" else "start")
        loss_limit, gradient_limit = spec["loss_convergence_limit"], spec["gradient_infinity_convergence_limit"]

        def note(event, values_for_checkpoint=None):
            event["training_seconds"] = training_clock.elapsed()
            events.write(json.dumps(event, allow_nan=False) + "\n")
            if (report["training_gate_first"] is None and event["loss"] <= loss_limit
                    and event["gradient_infinity"] <= gradient_limit):
                report["training_gate_first"] = dict(event)
                checkpoint("gate-first", values_for_checkpoint)
                progress("training-gate-first")

        for update in range(1, adam_updates + 1):
            loss, gradient, _ = evaluate()
            assert torch.isfinite(loss).item() and torch.isfinite(gradient).all().item()
            optimizer.param_groups[0]["lr"] = 1e-3 * .5 ** ((update - 1) // 1000)
            event = {"phase": "Adam", "update": update, "loss": float(loss),
                     "gradient_infinity": float(gradient.abs().max())}
            before = theta.detach().cpu().clone() if report["training_gate_first"] is None else None
            theta.grad = gradient.detach().clone()
            optimizer.step()
            report["Adam_updates"] = update
            note(event, before)
            if update in spec["complete_gradient_check_Adam_updates"]:
                full_check(f"Adam{update}")
            if update % 1000 == 0 or update == adam_updates:
                checkpoint(f"Adam{update}")
                progress(f"Adam{update}")
            if training_clock.elapsed() >= max_seconds:
                report["budget_truncated"] = True
                break
        if report["budget_truncated"]:
            iteration_cap = 0
        del optimizer
        theta.grad = None

        # The dense inverse Hessian is allocated after graph capture and the Adam phase.
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            needed = theta.numel() ** 2 * 8
            report["inverse_hessian_memory"] = {"free_bytes": free, "total_bytes": total, "needed_bytes": needed,
                                                "margin_bytes": spec["optimizer"]["memory_margin_bytes"]}
            assert free >= needed + spec["optimizer"]["memory_margin_bytes"], "not enough device memory for the dense inverse Hessian"
        options = spec["optimizer"]
        if args.self_test and theta.numel() ** 2 * 8 > 1e9:
            # The CPU control-flow test cannot hold the dense inverse Hessian of the full model.
            iteration_cap = 0
            report["self_test_note"] = "optimizer phase skipped: dense inverse Hessian too large for the CPU self-test"
        quasi_newton = SSBroyden(theta.numel() if iteration_cap else 1, theta.device, mode=options["mode"], c1=options["c1"],
                                 c2=options["c2"], max_ls=options["max_line_search_evaluations"])
        iterations_log = (args.output / "iterations.jsonl").open("x", buffering=1)

        def objective(x):
            with torch.no_grad():
                theta.copy_(x)
            loss, gradient, _ = evaluate()
            assert torch.isfinite(loss).item() and torch.isfinite(gradient).all().item()
            report["SSBroyden_evaluations"] += 1
            event = {"phase": "SSBroyden", "iteration": quasi_newton.iteration + 1,
                     "evaluation": report["SSBroyden_evaluations"], "loss": float(loss),
                     "gradient_infinity": float(gradient.abs().max())}
            note(event, theta.detach().cpu().clone() if report["training_gate_first"] is None else None)
            return loss.detach().clone(), gradient.detach().clone()

        def assign(x):
            with torch.no_grad():
                theta.copy_(x)

        loss, gradient, _ = evaluate()
        if iteration_cap:
            quasi_newton.start(theta.detach().clone(), float(loss), gradient.detach().clone())
        report["stop_reason"] = "training time guard" if report["budget_truncated"] else None
        report["observations"] = []
        report["tta"] = {"state": "NOT_YET_CONFIRMED", "time_to_verified_accuracy_s": None}

        def inspect(tag):
            sync()
            full_check(tag)
            checkpoint(tag)
            saved = report["checkpoints"][-1]
            before = theta.detach().clone()
            before_iteration = quasi_newton.iteration if iteration_cap else None
            request = {"tag": tag, "checkpoint": saved["filename"], "checkpoint_sha256": saved["sha256"],
                       "active_training_seconds": training_clock.elapsed(), "started_unix": time.time()}
            report["observations"].append(request)
            progress("physical-" + tag + "-started")
            with training_clock.pause():
                # Keep the exact dense Hessian in host memory while the separate observer
                # uses this same allocated card. Restore it only when training continues.
                if device == "cuda" and iteration_cap:
                    quasi_newton.H = quasi_newton.H.to("cpu")
                    request["hessian_host_spill_bytes"] = quasi_newton.H.numel() * quasi_newton.H.element_size()
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    free, _ = torch.cuda.mem_get_info()
                    request["free_gpu_bytes_before_observer"] = free
                    assert free >= protocol["time_to_accuracy"]["minimum_free_bytes_before_observer"], "insufficient observer GPU headroom"
                output = args.output.parent / ("physical-" + tag)
                observed = run_observer(args.output / saved["filename"], args.inputs, args.protocol, output,
                    protocol["physical_observation"]["observer_chunk"], protocol["time_to_accuracy"]["observer_timeout_seconds"], self_test=args.self_test)
                request.update(observed)
                if device == "cuda" and iteration_cap and not observed["physical_acceptance_pass"]:
                    quasi_newton.H = quasi_newton.H.to(device)
                    sync()
                    request["hessian_restored_for_continuation"] = True
            assert torch.equal(before, theta.detach()), "observer changed trainable parameters"
            assert not iteration_cap or quasi_newton.iteration == before_iteration
            request["parameters_and_optimizer_iteration_preserved"] = True
            request["observer_excluded_from_training_clock_s"] = training_clock.excluded
            if request["physical_acceptance_pass"]:
                report["tta"] = {"state": "PHYSICALLY_CONFIRMED", "first_passing_tag": tag,
                    "checkpoint_sha256": saved["sha256"], "active_training_seconds_at_checkpoint": saved["training_seconds"],
                    "time_to_verified_accuracy_s": time.perf_counter()-started,
                    "confirmation_unix": time.time(), "earliest_physical_crossing_known": False}
            progress("physical-" + tag + "-complete")
            return request["physical_acceptance_pass"]

        first_inspection_done = False
        progress("SSBroyden-started")
        while report["stop_reason"] is None:
            if not first_inspection_done and training_clock.elapsed() >= protocol["time_to_accuracy"]["inspection_training_seconds"][0]:
                first_inspection_done = True
                if inspect("tta-first"):
                    report["stop_reason"] = "physical acceptance confirmed"
                    break
            if training_clock.elapsed() > max_seconds:
                report["budget_truncated"], report["stop_reason"] = True, "training time guard"
                break
            if quasi_newton.iteration >= iteration_cap:
                report["stop_reason"] = "iteration cap"
                break
            tick = time.perf_counter()
            record = quasi_newton.step(objective, assign)
            sync()
            record["seconds"] = time.perf_counter() - tick
            report["SSBroyden_iterations"] = quasi_newton.iteration
            report["resets"] += int("reset" in record)
            report["skipped_updates"] += int("update_skipped" in record)
            iterations_log.write(json.dumps(record, allow_nan=False) + "\n")
            iteration = quasi_newton.iteration
            if iteration in checkpoint_iterations:
                checkpoint(f"SSBroyden-iteration{iteration}")
            if iteration % check_every == 0:
                report.setdefault("inverse_hessian_product_checks", []).append(
                    {"iteration": iteration, "relative_difference": quasi_newton.product_error()})
                full_check(f"SSBroyden-iteration{iteration}")
            if iteration % 200 == 0:
                progress(f"SSBroyden-iteration{iteration}")
            if record["stop"]:
                report["stop_reason"] = "two consecutive resets"
        iterations_log.close()
        report["iterations_sha256"] = sha(args.output / "iterations.jsonl")
        gc.collect()
        sync()
        report["training_seconds"] = training_clock.elapsed()
        loss, gradient, _ = (v.clone() for v in evaluate())
        report.update(final_loss=float(loss), final_gradient_infinity=float(gradient.abs().max()))
        report["training_converged"] = (report["final_loss"] <= loss_limit
                                        and report["final_gradient_infinity"] <= gradient_limit)
        checkpoint("final")
        full_check("final")
        if report["tta"]["state"] != "PHYSICALLY_CONFIRMED":
            if inspect("final-observation"):
                report["stop_reason"] = "physical acceptance confirmed"
            else:
                report["tta"].update(state="NOT_REACHED_WITHIN_BUDGET",
                    right_censor_elapsed_s=time.perf_counter()-started)
        report["training_seconds"] = training_clock.elapsed()
        report["observer_seconds_excluded"] = training_clock.excluded
        del quasi_newton
        report["state"] = "COMPLETE"
    except BaseException:
        report.update(state="FAILED", failure=traceback.format_exc())
    finally:
        events.close()
        report["finished_unix"] = time.time()
        if device == "cuda":
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report["events_sha256"] = sha(args.output / "events.jsonl")
        progress(report["state"])
    if report["state"] != "COMPLETE":
        raise SystemExit(1)


def _cpu_problem(args, protocol):
    """Self-test only: load_problem without moving tensors to CUDA."""
    from box376.data import FAMILIES, torch_data
    from box376.frozen_data import load_cache
    from box376.normalized_objective import normalization_key
    assert sha(args.inputs / "training-source.npz") == protocol["source_sha256"]
    assert sha(args.inputs / "calibration.npz") == protocol["calibration_sha256"]
    calibration = json.loads((args.inputs / "calibration.json").read_text())
    data = torch_data(load_cache(args.inputs / "training-source.npz"))
    data["normalization_key"] = normalization_key(calibration)
    assert [len(data["families"][name]["points"]) for name in FAMILIES] == protocol["family_counts"]
    return data, calibration


if __name__ == "__main__":
    main()
