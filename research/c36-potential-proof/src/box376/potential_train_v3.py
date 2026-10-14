"""C36 training with a temporal-smoothness continuation schedule (development).

The run trains on a sequence of motion-schedule widths w (box376.schedule_family), ending
on the original problem (w = 1/2), with the v2 recipe and total budget: 4,000 Adam updates
on the v1 learning-rate schedule and 800 L-BFGS calls. Adam state carries across stages;
L-BFGS restarts at each stage. The engine is the compiled packed activation with its
analytic transpose, replayed as a CUDA Graph. Each stage is qualified against dense
parameter AD at that width, and the original problem additionally against the preserved
native reference at the initial parameters. The training gate and all loss thresholds are
evaluated only on events of the original problem. Checkpoints use the v1 format.
"""
import argparse
import hashlib
import json
import time
import traceback
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
    for name in ("inputs", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--legacy-triton-target", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="CPU, eager, tiny budget")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    spec = protocol["training"]
    stages = protocol["schedules"][args.schedule]
    assert args.seed in protocol["seeds"] and stages[-1]["width"] == family.ORIGINAL_WIDTH
    assert sum(s.get("Adam_updates", 0) for s in stages) == spec["Adam_updates_per_seed"]
    assert sum(s.get("LBFGS_calls", 0) for s in stages) == spec["LBFGS_calls_per_seed"]
    device = "cpu" if args.self_test else "cuda"
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not args.self_test:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert args.reference is not None and sha(args.reference) == REFERENCE_SHA
    started = time.perf_counter()
    report = {"state": "RUNNING", "seed": args.seed, "schedule": args.schedule, "stages": stages,
              "protocol_sha256": sha(args.protocol), "protocol_version": protocol["version"],
              "self_test": args.self_test, "torch": torch.__version__, "numpy": np.__version__,
              "gpu": None if args.self_test else torch.cuda.get_device_name(0), "engine": "analytic",
              "checks": [], "checkpoints": [], "stage_records": [], "setup_seconds": {},
              "Adam_updates": 0, "LBFGS_calls": 0, "LBFGS_closures": 0, "training_gate_first": None,
              "budget_truncated": False, "started_unix": time.time()}
    events = (args.output / "events.jsonl").open("x", buffering=1)
    training_started = None

    def progress(phase):
        report.update(phase=phase, wall_seconds=time.perf_counter() - started)
        write_json(args.output / "report.json", report)
        print(json.dumps({"phase": phase, "wall_seconds": round(report["wall_seconds"], 1)}), flush=True)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def check(tag, actual, expected, reference_name):
        arrays, row = {}, {"tag": tag, "reference": reference_name, "comparisons": []}
        for kind, a, b in zip(KINDS, actual, expected):
            a, b = cpu(a), (cpu(b) if isinstance(b, torch.Tensor) else np.asarray(b).copy())
            assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
            assert np.isfinite(a).all() and np.isfinite(b).all()
            error = float(np.max(np.abs(a - b) / (protocol["atol"] + protocol["rtol"] * np.abs(b))))
            row["comparisons"].append({"name": kind, "entries": int(a.size), "maximum_error_over_original_budget": error})
            arrays[kind + "_actual"], arrays[kind + "_reference"] = a, b
            assert error <= 1, (tag, kind, error)
        arrays["theta"] = cpu(theta)
        path = args.output / f"check-{tag}.npz"
        np.savez_compressed(path, **arrays)
        row.update(filename=path.name, sha256=sha(path))
        report["checks"].append(row)
        progress(f"check-{tag}")

    def checkpoint(tag, values=None):
        path = args.output / f"seed{args.seed}-{args.schedule}-{tag}.pt"
        torch.save({"theta": (theta.detach().cpu().clone() if values is None else values),
                    "protocol_sha256": report["protocol_sha256"], "representation": "C36-curl-potential",
                    "Adam_updates": report["Adam_updates"], "LBFGS_calls": report["LBFGS_calls"],
                    "LBFGS_closures": report["LBFGS_closures"], "training_engine": "analytic",
                    "schedule": args.schedule, "width": current["width"]}, path)
        report["checkpoints"].append({"tag": tag, "filename": path.name, "sha256": sha(path),
                                      "width": current["width"],
                                      "training_seconds": time.perf_counter() - training_started if training_started else 0.0})

    current = {"width": None, "evaluate": None, "dense": None}
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
        report["initial_theta_sha256"] = hashlib.sha256(cpu(theta).tobytes()).hexdigest()
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

        def enter_stage(index):
            stage = stages[index]
            width = stage["width"]
            current["evaluate"] = current["dense"] = None
            if device == "cuda":
                torch.cuda.empty_cache()
            t = time.perf_counter()
            engine, record = family.prepared_objective(PreparedPotentialNormalizedObjective, data, calibration, width,
                                                       activation=forward, activation_vjp=transpose)
            dense, _ = family.prepared_objective(PreparedPotentialADObjective, data, calibration, width)
            record = dict(record or {"width": width})
            record.update(stage=index, entered_Adam_updates=report["Adam_updates"],
                          entered_LBFGS_calls=report["LBFGS_calls"])
            original = theta.detach().clone()
            if args.self_test:
                evaluate = lambda: engine(params, chunk)
            else:
                for _ in range(2):
                    engine(params, chunk)
                sync()
                evaluate, record["graph_capture_seconds"] = capture(lambda: engine(params, chunk))
            if index == 0 and native is not None and width == family.ORIGINAL_WIDTH:
                check(f"stage{index}-original-native", evaluate(), native, "preserved native AD")
            check(f"stage{index}-entry-denseAD", tuple(v.clone() for v in evaluate()), dense(params, chunk),
                  f"dense parameter AD at width {width}")
            with torch.no_grad():
                theta.add_(.001)
            check(f"stage{index}-changed-denseAD", tuple(v.clone() for v in evaluate()), dense(params, chunk),
                  f"dense parameter AD at width {width}")
            with torch.no_grad():
                theta.copy_(original)
            record["setup_seconds"] = time.perf_counter() - t
            report["stage_records"].append(record)
            current.update(width=width, evaluate=evaluate, dense=dense, index=index)
            progress(f"stage{index}-width{width}")

        # The original problem is qualified against the preserved native reference before training.
        if native is not None and stages[0]["width"] != family.ORIGINAL_WIDTH:
            probe, _ = family.prepared_objective(PreparedPotentialNormalizedObjective, data, calibration,
                                                 family.ORIGINAL_WIDTH, activation=forward, activation_vjp=transpose)
            check("original-problem-native", probe(params, chunk), native, "preserved native AD")
            del probe

        def full_check(tag):
            check(tag, tuple(v.clone() for v in current["evaluate"]()), current["dense"](params, chunk),
                  f"dense parameter AD at width {current['width']}")

        loss_limit, gradient_limit = spec["loss_convergence_limit"], spec["gradient_infinity_convergence_limit"]
        checkpoint_calls = set(spec["checkpoint_LBFGS_calls"])
        check_calls = spec["complete_gradient_check_every_LBFGS_calls"]
        closure_cap = spec["LBFGS_closure_cap_per_seed"]
        enter_stage(0)
        optimizer = torch.optim.Adam([theta], lr=1e-3, betas=tuple(spec["Adam_betas"]),
                                     eps=spec["Adam_eps"], weight_decay=0, foreach=False)
        progress("training-started")
        training_started = time.perf_counter()
        checkpoint("initial")

        def note(event, values_for_checkpoint=None):
            event["training_seconds"] = time.perf_counter() - training_started
            event["width"] = current["width"]
            events.write(json.dumps(event, allow_nan=False) + "\n")
            if (report["training_gate_first"] is None and current["width"] == family.ORIGINAL_WIDTH
                    and event["loss"] <= loss_limit and event["gradient_infinity"] <= gradient_limit):
                report["training_gate_first"] = dict(event)
                checkpoint("gate-first", values_for_checkpoint)
                progress("training-gate-first")

        scale = 1 if not args.self_test else 0
        update = 0
        for index, stage in enumerate(stages):
            if index > 0:
                checkpoint(f"stage{index - 1}-end")
                enter_stage(index)
            adam = stage.get("Adam_updates", 0) if scale else min(2, stage.get("Adam_updates", 0))
            for _ in range(adam):
                update += 1
                loss, gradient, _ = current["evaluate"]()
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
                if update % 1000 == 0:
                    checkpoint(f"Adam{update}")
                    progress(f"Adam{update}")
            calls = stage.get("LBFGS_calls", 0) if scale else min(1, stage.get("LBFGS_calls", 0))
            if not calls:
                continue
            lbfgs = torch.optim.LBFGS([theta], lr=spec["LBFGS_lr"], max_iter=spec["LBFGS_max_iter"],
                                      max_eval=spec["LBFGS_max_eval"], history_size=spec["LBFGS_history_size"],
                                      tolerance_grad=0, tolerance_change=0, line_search_fn=spec["LBFGS_line_search"])

            def closure():
                if report["LBFGS_closures"] >= closure_cap:
                    raise RuntimeError("Frozen LBFGS closure budget exhausted")
                loss, gradient, _ = current["evaluate"]()
                assert torch.isfinite(loss).item() and torch.isfinite(gradient).all().item()
                index_ = report["LBFGS_closures"]
                report["LBFGS_closures"] += 1
                event = {"phase": "LBFGS", "call": report["LBFGS_calls"] + 1, "closure": index_,
                         "loss": float(loss), "gradient_infinity": float(gradient.abs().max())}
                note(event, theta.detach().cpu().clone() if report["training_gate_first"] is None else None)
                theta.grad = gradient.detach().clone()
                return loss.detach().clone()

            for _ in range(calls):
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
        assert current["width"] == family.ORIGINAL_WIDTH
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
