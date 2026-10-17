"""Development C36 v2 training: the v1 recipe with a larger L-BFGS budget.

Only the L-BFGS call and closure budget differs from v1. The training engine is either
the qualified packed activation with its analytic transpose (``analytic``) or parameter
AD over the same packed forward (``parameter_ad``); both are compiled and replayed as a
CUDA Graph. Complete loss/gradient checks compare it with dense eager parameter AD and,
at the original parameters, with the preserved full native-AD reference. Checkpoints use
the v1 format so the unchanged physical observer can read them.
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
from box376.data import initialize, flatten, views
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.potential_run import capture, cpu, load_problem, sha, write_json

REFERENCE_SHA = "172174ea49a82a21e0616571ae303560d275317d8b5319c22ea49dbd628d8b66"
KINDS = ("loss", "gradient", "numerators")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--engine", choices=("analytic", "parameter_ad"), default="analytic")
    parser.add_argument("--reference", type=Path, help="preserved full native-AD reference (.npz)")
    parser.add_argument("--legacy-triton-target", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="CPU, eager, tiny budget; checks the control flow only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    spec = protocol["training"]
    assert args.seed in protocol["seeds"]
    device = "cpu" if args.self_test else "cuda"
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not args.self_test:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert args.reference is not None and sha(args.reference) == REFERENCE_SHA
    started = time.perf_counter()
    report = {"state": "RUNNING", "seed": args.seed, "protocol_sha256": sha(args.protocol),
              "protocol_version": protocol["version"], "self_test": args.self_test,
              "torch": torch.__version__, "numpy": np.__version__,
              "gpu": None if args.self_test else torch.cuda.get_device_name(0),
              "engine": args.engine, "engine_description": ("packed activation + analytic transpose"
                  if args.engine == "analytic" else "parameter AD over the packed activation")
                  + ("" if args.self_test else ", compiled, CUDA Graph"),
              "checks": [], "checkpoints": [], "setup_seconds": {}, "Adam_updates": 0,
              "LBFGS_calls": 0, "LBFGS_closures": 0, "training_gate_first": None,
              "budget_truncated": False, "started_unix": time.time()}
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
        path = args.output / f"seed{args.seed}-{args.engine}-{tag}.pt"
        torch.save({"theta": (theta.detach().cpu().clone() if values is None else values),
                    "protocol_sha256": report["protocol_sha256"], "representation": "C36-curl-potential",
                    "Adam_updates": report["Adam_updates"], "LBFGS_calls": report["LBFGS_calls"],
                    "LBFGS_closures": report["LBFGS_closures"], "training_engine": report["engine"]}, path)
        report["checkpoints"].append({"tag": tag, "filename": path.name, "sha256": sha(path),
                                      "training_seconds": time.perf_counter() - training_started
                                      if training_started else 0.0})

    training_started = None
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
        native = None if reference is None else tuple(reference[f"seed{args.seed}_subset_native_" + k] for k in KINDS)
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
        if native is not None:
            check("eager-original-native", eager(params, chunk), native, "preserved native AD")
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
            check("compiled-original-native", engine(params, chunk), native, "preserved native AD")
            replay, report["setup_seconds"]["graph_capture"] = capture(lambda: engine(params, chunk))
            check("graph-original-native", replay(), native, "preserved native AD")
            with torch.no_grad():
                theta.add_(.001)
            check("graph-changed-denseAD", replay(), dense_ad(params, chunk), "dense parameter AD")
            with torch.no_grad():
                theta.copy_(original)
            evaluate = replay
        if evaluate is None:
            evaluate = lambda: engine(params, chunk)

        def full_check(tag):
            values = tuple(v.clone() for v in evaluate())
            check(tag, values, dense_ad(params, chunk), "dense parameter AD")

        adam_updates = 3 if args.self_test else spec["Adam_updates_per_seed"]
        lbfgs_calls = 2 if args.self_test else spec["LBFGS_calls_per_seed"]
        closure_cap = spec["LBFGS_closure_cap_per_seed"]
        max_seconds = spec["maximum_training_seconds"]
        checkpoint_calls = set(spec["checkpoint_LBFGS_calls"])
        check_calls = spec["complete_gradient_check_every_LBFGS_calls"]
        optimizer = torch.optim.Adam([theta], lr=1e-3, betas=tuple(spec["Adam_betas"]),
                                     eps=spec["Adam_eps"], weight_decay=0, foreach=False)
        progress("training-started")
        training_started = time.perf_counter()
        checkpoint("initial")
        full_check("Adam0")
        loss_limit, gradient_limit = spec["loss_convergence_limit"], spec["gradient_infinity_convergence_limit"]

        def note(event, values_for_checkpoint=None):
            event["training_seconds"] = time.perf_counter() - training_started
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

        optimizer = torch.optim.LBFGS([theta], lr=spec["LBFGS_lr"], max_iter=spec["LBFGS_max_iter"],
                                      max_eval=spec["LBFGS_max_eval"], history_size=spec["LBFGS_history_size"],
                                      tolerance_grad=0, tolerance_change=0,
                                      line_search_fn=spec["LBFGS_line_search"])

        def closure():
            if report["LBFGS_closures"] >= closure_cap:
                raise RuntimeError("Frozen LBFGS closure budget exhausted")
            loss, gradient, _ = evaluate()
            assert torch.isfinite(loss).item() and torch.isfinite(gradient).all().item()
            index = report["LBFGS_closures"]
            report["LBFGS_closures"] += 1
            event = {"phase": "LBFGS", "call": report["LBFGS_calls"] + 1, "closure": index,
                     "loss": float(loss), "gradient_infinity": float(gradient.abs().max())}
            note(event, theta.detach().cpu().clone() if report["training_gate_first"] is None else None)
            theta.grad = gradient.detach().clone()
            return loss.detach().clone()

        for call in range(1, lbfgs_calls + 1):
            if time.perf_counter() - training_started > max_seconds:
                report["budget_truncated"] = True
                break
            optimizer.step(closure)
            report["LBFGS_calls"] = call
            if call in checkpoint_calls:
                checkpoint(f"LBFGS-call{call}")
            if call % check_calls == 0:
                full_check(f"LBFGS-call{call}")
            if call % 20 == 0:
                progress(f"LBFGS-call{call}")
        sync()
        report["training_seconds"] = time.perf_counter() - training_started
        loss, gradient, _ = (v.clone() for v in evaluate())
        report.update(final_loss=float(loss), final_gradient_infinity=float(gradient.abs().max()))
        report["training_converged"] = (report["final_loss"] <= loss_limit
                                        and report["final_gradient_infinity"] <= gradient_limit)
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
