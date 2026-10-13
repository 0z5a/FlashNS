"""Finite H100 qualification and paired timing of packed analytic activation.

Both measured arms share the same packed forward and optional compilation. The complete native
reference is preserved evidence, not recomputed here. No training is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from torch_c36_packed import activate_packed
from torch_c36_packed_vjp import activate_packed_vjp
from box376.data import FAMILIES, views
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.potential_run import capture, cpu, load_problem, sha, timed_call, write_json

REFERENCE_SHA = "172174ea49a82a21e0616571ae303560d275317d8b5319c22ea49dbd628d8b66"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "protocol", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--activation-mode", choices=("compiled", "eager"), default="compiled")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    assert sha(args.reference) == REFERENCE_SHA
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    with np.load(args.reference, allow_pickle=False) as raw:
        reference = {name: raw[name].copy() for name in raw.files}
    arrays, checks = {}, []
    source_names = ["torch_c36.py", "torch_c36_packed.py", "torch_c36_packed_vjp.py",
                    "box376/potential_model.py", "box376/potential_objective.py",
                    "box376/potential_ad_objective.py", "box376/packed_analytic_gpu_run.py"]
    source = Path(__file__).resolve().parents[1]
    report = {"state": "RUNNING", "started": time.time(), "device": "cuda",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "numpy": np.__version__, "cuda": torch.version.cuda, "points": 4999,
        "parameters": 50692, "seeds": protocol["seeds"], "checks": checks,
        "atol": 1e-11, "rtol": 1e-9, "tolerance_relaxed": False,
        "reference_sha256": REFERENCE_SHA, "fresh_native_coordinate_AD_repeated": False,
        "protocol_sha256": sha(args.protocol), "groups": [], "setup_seconds": {},
        "source_sha256": {name: sha(source / name) for name in source_names},
        "complete_solver_speedup": None, "training_performed": False,
        "physical_acceptance_evaluated": False, "default_activation_changed": False}
    report["activation_mode"] = args.activation_mode

    def progress(phase):
        report["phase"] = phase
        write_json(args.output / "progress.json", report)
        # Preserve partial arrays even when a later compiler/capture/check fails.
        np.savez_compressed(args.output / "raw-proof.npz", **arrays)
        print(json.dumps({"phase": phase, "checks": len(checks), "unix": time.time()}), flush=True)

    def check(prefix, actual, expected):
        for kind, a, b in zip(("loss", "gradient", "numerators"), actual, expected):
            a, b = cpu(a).copy(), np.asarray(b).copy()
            assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
            assert a.size == {"loss": 1, "gradient": 50692, "numerators": 4}[kind]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            name = prefix + "_" + kind
            arrays[name + "_actual"], arrays[name + "_reference"] = a, b
            error = float(np.max(np.abs(a - b) / (1e-11 + 1e-9 * np.abs(b))))
            checks.append({"name": name, "maximum_error_over_original_budget": error})
            assert error <= 1, (name, error)
        progress(prefix)

    try:
        data, calibration = load_problem(args, protocol)
        for family in FAMILIES:
            for key in ("points", "weights", "target") + (("force", "force_gradient") if family == "interior" else ()):
                assert np.array_equal(cpu(data["families"][family][key]), reference["canonical_" + family + "_" + key])
        eager, report["setup_seconds"]["eager"] = timed_call(lambda:
            PreparedPotentialNormalizedObjective(data, calibration,
                activation=activate_packed, activation_vjp=activate_packed_vjp))
        dense_ad, report["setup_seconds"]["dense_AD"] = timed_call(lambda:
            PreparedPotentialADObjective(data, calibration))
        progress("prepared; qualifying eager analytic path before compilation")
        topology = tuple(protocol["topology"])
        for seed in protocol["seeds"]:
            theta = torch.from_numpy(reference[f"seed{seed}_theta"]).to("cuda")
            params = views(theta, topology)
            native = tuple(reference[f"seed{seed}_subset_native_" + kind] for kind in ("loss", "gradient", "numerators"))
            for chunk in (128, 64):
                check(f"seed{seed}-eager{chunk}-native", eager(params, chunk), native)
            theta.add_(.001)
            expected = tuple(cpu(x).copy() for x in dense_ad(params, 128))
            check(f"seed{seed}-eager128-changed", eager(params, 128), expected)
        # Warm device index constants before Dynamo traces their consumers.
        activate_packed(torch.zeros(128, 128, 36, device="cuda"))
        forward, transpose = activate_packed, activate_packed_vjp
        if args.activation_mode == "compiled":
            forward = torch.compile(forward, fullgraph=True, dynamic=False)
            transpose = torch.compile(transpose, fullgraph=True, dynamic=False)
        candidate, report["setup_seconds"]["candidate"] = timed_call(lambda:
            PreparedPotentialNormalizedObjective(data, calibration,
                activation=forward, activation_vjp=transpose))
        baseline, report["setup_seconds"]["baseline"] = timed_call(lambda:
            PreparedPotentialADObjective(data, calibration, activation=forward))
        spec = protocol["performance"]
        for index, seed in enumerate(protocol["seeds"]):
            theta = torch.from_numpy(reference[f"seed{seed}_theta"]).to("cuda")
            original = theta.clone()
            params = views(theta, topology)
            native = tuple(reference[f"seed{seed}_subset_native_" + kind] for kind in ("loss", "gradient", "numerators"))
            functions = {"candidate": lambda: candidate(params, 128),
                         "baseline": lambda: baseline(params, 128)}
            progress(f"seed{seed}-compilation-and-warmup-started")
            group = {"seed": seed, "pairs": [], "warmup_seconds": {}, "capture_seconds": {}}
            for name, fn in functions.items():
                _, duration = timed_call(lambda: [fn() for _ in range(2)])
                group["warmup_seconds"][name] = duration
                check(f"seed{seed}-{name}-{args.activation_mode}-native", fn(), native)
            theta.add_(.001)
            expected = tuple(cpu(x).copy() for x in dense_ad(params, 128))
            for name, fn in functions.items():
                check(f"seed{seed}-{name}-{args.activation_mode}-changed", fn(), expected)
            theta.copy_(original)
            for name, fn in list(functions.items()):
                replay, duration = capture(fn)
                group["capture_seconds"][name] = duration
                check(f"seed{seed}-{name}-graph-native", replay(), native)
                theta.add_(.001)
                check(f"seed{seed}-{name}-graph-changed", replay(), expected)
                theta.copy_(original)
                functions[name] = replay
            # Frozen finite batch measures graph replay after both eager and
            # captured implementations passed original and changed parameters.
            report["groups"].append(group)
            for pair in range(spec["pairs_per_seed_per_baseline"]):
                order = ["candidate", "baseline"] if (pair + index) % 2 == 0 else ["baseline", "candidate"]
                row = {"index": pair, "order": order}
                for name in order:
                    progress(f"seed{seed}-pair{pair}-{name}-started")
                    durations, elapsed = [], 0.
                    while len(durations) < spec["window_minimum_evaluations"] or elapsed < spec["window_minimum_seconds"]:
                        values, duration = timed_call(functions[name])
                        assert all(torch.isfinite(x).all().item() for x in values)
                        durations.append(duration)
                        elapsed += duration
                    row[name] = {"evaluation_seconds": durations, "mean_seconds": float(np.mean(durations))}
                row["ratio"] = row["baseline"]["mean_seconds"] / row["candidate"]["mean_seconds"]
                group["pairs"].append(row)
                progress(f"seed{seed}-pair{pair}-complete")
            for name, fn in functions.items():
                check(f"seed{seed}-{name}-graph-final", fn(), native)
        logs = np.log([row["ratio"] for group in report["groups"] for row in group["pairs"]])
        rng = np.random.default_rng(spec["bootstrap_seed"])
        samples = np.exp(rng.choice(logs, (spec["bootstrap_resamples"], len(logs))).mean(1))
        report.update(state="COMPLETE", mode=f"both_{args.activation_mode}_forward_CUDA_graphs",
            paired_geometric_mean_ratio=float(np.exp(logs.mean())),
            paired_bootstrap_95_percent_interval=np.quantile(samples, [.025, .975]).tolist())
    except BaseException:
        report.update(state="FAILED", failure=traceback.format_exc())
    finally:
        report["finished"] = time.time()
        np.savez_compressed(args.output / "raw-proof.npz", **arrays)
        report["raw_sha256"] = sha(args.output / "raw-proof.npz")
        for name, digest in report["source_sha256"].items():
            assert sha(source / name) == digest, name
        write_json(args.output / "report.json", report)
    if report["state"] != "COMPLETE":
        raise RuntimeError(report["failure"])
    print(json.dumps({"state": report["state"], "ratio": report["paired_geometric_mean_ratio"]}), flush=True)


if __name__ == "__main__":
    main()
