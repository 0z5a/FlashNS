"""C36 resumed L-BFGS training on the original problem (development, protocol v4).

The run resumes from a sha256-verified start checkpoint (the final parameters of a v2
baseline or v3 C2 run) and adds a long L-BFGS budget at w = 1/2 with a fresh L-BFGS
history. Counters (Adam updates, L-BFGS calls and closures) restart at zero so that the
all-planes observer indexes this run's own event log; the source counts are recorded in
the report. The engine is the compiled packed activation with its analytic transpose,
replayed as a CUDA Graph and kept alive with its replay. The original-problem engine is
checked against the preserved native reference at the stored initial parameters, the
resume point against dense parameter AD (entry, changed, after allocator churn), and the
start loss against the source run's final loss. Checkpoints use the v1 format.
"""
import argparse
import gc
import hashlib
import json
import time
import traceback
import weakref
from pathlib import Path

import numpy as np
import torch

from torch_c36_packed import activate_packed
from torch_c36_packed_vjp import activate_packed_vjp
from box376 import schedule_family as family
from box376.data import flatten, initialize, views
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.potential_run import capture, cpu, load_problem, sha, write_json
from box376.potential_train_v2 import REFERENCE_SHA, KINDS, _cpu_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "protocol", "output", "start_checkpoint"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start", choices=("baseline", "C2"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--legacy-triton-target", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="CPU, eager, three L-BFGS calls")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    spec = protocol["training"]
    start_spec = protocol["starts"][f"{args.start}:{args.seed}"]
    assert args.seed in protocol["seeds"] and spec["width"] == family.ORIGINAL_WIDTH
    device = "cpu" if args.self_test else "cuda"
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not args.self_test:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert args.reference is not None and sha(args.reference) == REFERENCE_SHA
    started = time.perf_counter()
    report = {"state": "RUNNING", "seed": args.seed, "start": args.start, "schedule": f"resume-{args.start}",
              "protocol_sha256": sha(args.protocol), "protocol_version": protocol["version"],
              "self_test": args.self_test, "torch": torch.__version__, "numpy": np.__version__,
              "gpu": None if args.self_test else torch.cuda.get_device_name(0), "engine": "analytic",
              "start_checkpoint": dict(start_spec), "checks": [], "checkpoints": [], "stage_records": [],
              "setup_seconds": {}, "Adam_updates": 0, "LBFGS_calls": 0, "LBFGS_closures": 0,
              "training_gate_first": None, "budget_truncated": False, "started_unix": time.time()}
    events = (args.output / "events.jsonl").open("x", buffering=1)
    training_started = None

    def progress(phase):
        report.update(phase=phase, wall_seconds=time.perf_counter() - started)
        write_json(args.output / "report.json", report)
        print(json.dumps({"phase": phase, "wall_seconds": round(report["wall_seconds"], 1)}), flush=True)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def budget_error(a, b):
        return float(np.max(np.abs(a - b) / (protocol["atol"] + protocol["rtol"] * np.abs(b))))

    def check(tag, actual, expected, reference_name):
        arrays, row = {}, {"tag": tag, "reference": reference_name, "comparisons": []}
        for kind, a, b in zip(KINDS, actual, expected):
            a, b = cpu(a), (cpu(b) if isinstance(b, torch.Tensor) else np.asarray(b).copy())
            assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
            assert np.isfinite(a).all() and np.isfinite(b).all()
            error = budget_error(a, b)
            row["comparisons"].append({"name": kind, "entries": int(a.size), "maximum_error_over_original_budget": error})
            arrays[kind + "_actual"], arrays[kind + "_reference"] = a, b
            assert error <= 1, (tag, kind, error)
        arrays["theta"] = cpu(theta)
        path = args.output / f"check-{tag}.npz"
        np.savez_compressed(path, **arrays)
        row.update(filename=path.name, sha256=sha(path))
        report["checks"].append(row)
        progress(f"check-{tag}")

    def checkpoint(tag):
        path = args.output / f"seed{args.seed}-resume-{args.start}-{tag}.pt"
        torch.save({"theta": theta.detach().cpu().clone(),
                    "protocol_sha256": report["protocol_sha256"], "representation": "C36-curl-potential",
                    "Adam_updates": 0, "LBFGS_calls": report["LBFGS_calls"],
                    "LBFGS_closures": report["LBFGS_closures"], "training_engine": "analytic",
                    "schedule": report["schedule"], "width": family.ORIGINAL_WIDTH,
                    "start_checkpoint_sha256": start_spec["checkpoint_sha256"]}, path)
        report["checkpoints"].append({"tag": tag, "filename": path.name, "sha256": sha(path),
                                      "LBFGS_calls": report["LBFGS_calls"], "LBFGS_closures": report["LBFGS_closures"],
                                      "training_seconds": time.perf_counter() - training_started if training_started else 0.0})

    # The engine is kept here for as long as its replay is in use (a captured graph does not own it).
    current = {"evaluate": None, "dense": None, "engine": None}
    try:
        # Start checkpoint: file and parameter hashes must match the protocol.
        report["start_checkpoint"]["observed_checkpoint_sha256"] = sha(args.start_checkpoint)
        if report["start_checkpoint"]["observed_checkpoint_sha256"] != start_spec["checkpoint_sha256"]:
            raise RuntimeError("start checkpoint sha256 differs from the protocol; refusing to start")
        stored = torch.load(args.start_checkpoint, map_location="cpu", weights_only=True)["theta"]
        theta_sha = hashlib.sha256(stored.numpy().tobytes()).hexdigest()
        if theta_sha != start_spec["theta_sha256"]:
            raise RuntimeError("start parameter sha256 differs from the protocol; refusing to start")
        data, calibration = load_problem(args, protocol) if not args.self_test else _cpu_problem(args, protocol)
        topology = tuple(protocol["topology"])
        reference = None
        if args.reference is not None:
            with np.load(args.reference, allow_pickle=False) as raw:
                reference = {k: raw[k].copy() for k in raw.files if k.startswith(f"seed{args.seed}_")}
        initial = flatten(initialize(topology, args.seed))
        if reference is not None:
            stored_initial = torch.from_numpy(reference[f"seed{args.seed}_theta"])
            report["initialize_matches_stored_bytes"] = bool(torch.equal(initial, stored_initial))
            initial = stored_initial.clone()
        theta = torch.nn.Parameter(initial.to(device))
        params = views(theta.detach(), topology)
        native = None if reference is None else tuple(reference[f"seed{args.seed}_subset_native_" + k] for k in KINDS)
        chunk = protocol["explicit_chunk"]
        if args.self_test:
            forward, transpose = activate_packed, activate_packed_vjp
        else:
            if args.legacy_triton_target:
                from box376.triton_target_compat import enable_legacy_target_compat
                report["compiler_compatibility"] = enable_legacy_target_compat()
            activate_packed(torch.zeros(128, 128, 36, device=device))
            forward = torch.compile(activate_packed, fullgraph=True, dynamic=False)
            transpose = torch.compile(activate_packed_vjp, fullgraph=True, dynamic=False)

        t = time.perf_counter()
        engine, _ = family.prepared_objective(PreparedPotentialNormalizedObjective, data, calibration,
                                              family.ORIGINAL_WIDTH, activation=forward, activation_vjp=transpose)
        dense, _ = family.prepared_objective(PreparedPotentialADObjective, data, calibration, family.ORIGINAL_WIDTH)
        # The original-problem engine is qualified at the stored initial parameters.
        if native is not None:
            check("original-problem-native", engine(params, chunk), native, "preserved native AD")
        with torch.no_grad():
            theta.copy_(stored.to(device))
        if args.self_test:
            handle = weakref.ref(engine)

            def evaluate():
                live = handle()
                assert live is not None, "engine released while its replay is in use"
                return live(params, chunk)
        else:
            for _ in range(2):
                engine(params, chunk)
            sync()
            evaluate, report["setup_seconds"]["graph_capture"] = capture(lambda: engine(params, chunk))
        current.update(evaluate=evaluate, dense=dense, engine=engine)
        del engine
        resume = theta.detach().clone()
        check("stage0-entry-denseAD", tuple(v.clone() for v in evaluate()), dense(params, chunk), "dense parameter AD at the start")
        with torch.no_grad():
            theta.add_(.001)
        check("stage0-changed-denseAD", tuple(v.clone() for v in evaluate()), dense(params, chunk), "dense parameter AD after a change")
        with torch.no_grad():
            theta.copy_(resume)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            churn = torch.full((int(min(.5 * free, 8e9)) // 8,), float("nan"), device=device)
            del churn
        check("stage0-retained", tuple(v.clone() for v in evaluate()), dense(params, chunk), "dense parameter AD after churn")
        # Continuity with the source run.
        loss0 = cpu(evaluate()[0].clone())
        error = budget_error(loss0, np.asarray(start_spec["source_final_loss"]))
        report["continuity"] = {"start_loss": float(loss0), "source_final_loss": start_spec["source_final_loss"],
                                "maximum_error_over_original_budget": error}
        assert error <= 1, ("continuity", error)
        report["setup_seconds"]["stage"] = time.perf_counter() - t
        progress("resume-ready")

        def full_check(tag):
            check(tag, tuple(v.clone() for v in current["evaluate"]()), current["dense"](params, chunk), "dense parameter AD")

        loss_limit, gradient_limit = spec["loss_convergence_limit"], spec["gradient_infinity_convergence_limit"]
        calls_total = spec["LBFGS_calls_per_seed"] if not args.self_test else 3
        checkpoint_calls = set(spec["checkpoint_LBFGS_calls"]) if not args.self_test else {1, 2}
        check_calls = spec["complete_gradient_check_every_LBFGS_calls"] if not args.self_test else 2
        closure_cap = spec["LBFGS_closure_cap_per_seed"]
        lbfgs = torch.optim.LBFGS([theta], lr=spec["LBFGS_lr"], max_iter=spec["LBFGS_max_iter"],
                                  max_eval=spec["LBFGS_max_eval"], history_size=spec["LBFGS_history_size"],
                                  tolerance_grad=0, tolerance_change=0, line_search_fn=spec["LBFGS_line_search"])
        training_started = time.perf_counter()
        progress("training-started")

        def closure():
            if report["LBFGS_closures"] >= closure_cap:
                raise RuntimeError("Frozen LBFGS closure budget exhausted")
            loss, gradient, _ = current["evaluate"]()
            assert torch.isfinite(loss).item() and loss.item() >= 0 and torch.isfinite(gradient).all().item()
            index = report["LBFGS_closures"]
            report["LBFGS_closures"] += 1
            event = {"phase": "LBFGS", "call": report["LBFGS_calls"] + 1, "closure": index,
                     "loss": float(loss), "gradient_infinity": float(gradient.abs().max()),
                     "training_seconds": time.perf_counter() - training_started, "width": family.ORIGINAL_WIDTH}
            events.write(json.dumps(event, allow_nan=False) + "\n")
            if (report["training_gate_first"] is None and event["loss"] <= loss_limit
                    and event["gradient_infinity"] <= gradient_limit):
                report["training_gate_first"] = dict(event)
            theta.grad = gradient.detach().clone()
            return loss.detach().clone()

        for _ in range(calls_total):
            if time.perf_counter() - training_started > spec["maximum_training_seconds"]:
                report["budget_truncated"] = True
                break
            lbfgs.step(closure)
            report["LBFGS_calls"] += 1
            call = report["LBFGS_calls"]
            if call in checkpoint_calls:
                checkpoint(f"LBFGS-call{call}")
            if call % check_calls == 0:
                full_check(f"LBFGS-call{call}")
            if call % 20 == 0:
                progress(f"LBFGS-call{call}")
        sync()
        report["training_seconds"] = time.perf_counter() - training_started
        loss, gradient, _ = (v.clone() for v in current["evaluate"]())
        report.update(final_loss=float(loss), final_gradient_infinity=float(gradient.abs().max()))
        report["training_converged"] = report["final_loss"] <= loss_limit and report["final_gradient_infinity"] <= gradient_limit
        checkpoint("final")
        full_check("final")
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


if __name__ == "__main__":
    main()
