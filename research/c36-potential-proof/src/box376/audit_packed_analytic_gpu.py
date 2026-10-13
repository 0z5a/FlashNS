"""NumPy-only reader of the finite packed-analytic H100 experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proof", "reference", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    report = json.loads((args.proof / "report.json").read_text())
    protocol = json.loads(args.protocol.read_text())
    assert report["state"] == "COMPLETE" and report["device"] == "cuda"
    assert report["points"] == 4999 and report["parameters"] == 50692
    assert report["seeds"] == protocol["seeds"] == [361901, 361902]
    assert report["atol"] == 1e-11 and report["rtol"] == 1e-9 and not report["tolerance_relaxed"]
    mode = report["activation_mode"]
    assert mode in ["eager", "compiled"]
    assert report["mode"] == f"both_{mode}_forward_CUDA_graphs"
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == report["protocol_sha256"]
    assert hashlib.sha256(args.reference.read_bytes()).hexdigest() == report["reference_sha256"] == "172174ea49a82a21e0616571ae303560d275317d8b5319c22ea49dbd628d8b66"
    raw_path = args.proof / "raw-proof.npz"
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == report["raw_sha256"]
    required = set()
    for seed in [361901, 361902]:
        for variant in ["eager128-native", "eager64-native", "eager128-changed"] + [
            f"{name}-{phase}" for name in ["candidate", "baseline"] for phase in
            [f"{mode}-native", f"{mode}-changed", "graph-native", "graph-changed", "graph-final"]]:
            for kind in ["loss", "gradient", "numerators"]:
                required.add(f"seed{seed}-{variant}_{kind}")
    assert len(required) == 78
    assert {row["name"] for row in report["checks"]} == required and len(report["checks"]) == 78
    errors = []
    with np.load(raw_path, allow_pickle=False) as raw, np.load(args.reference, allow_pickle=False) as original:
        assert set(raw.files) == {name + suffix for name in required for suffix in ["_actual", "_reference"]}
        for name in sorted(required):
            kind = name.rsplit("_", 1)[1]
            a, b = raw[name + "_actual"], raw[name + "_reference"]
            assert a.shape == b.shape and a.dtype == b.dtype == np.float64
            assert a.size == {"loss": 1, "gradient": 50692, "numerators": 4}[kind]
            seed = int(name.split("-", 1)[0][4:])
            assert a.shape == original[f"seed{seed}_subset_native_{kind}"].shape
            assert np.isfinite(a).all() and np.isfinite(b).all()
            if "-native_" in name or "-graph-final_" in name:
                assert np.array_equal(b, original[f"seed{seed}_subset_native_{kind}"])
            error = float(np.max(np.abs(a - b) / (1e-11 + 1e-9 * np.abs(b))))
            assert error <= 1, (name, error)
            errors.append(error)
    ratios, candidate_means, baseline_means = [], [], []
    assert [g["seed"] for g in report["groups"]] == [361901, 361902]
    spec = protocol["performance"]
    for index, group in enumerate(report["groups"]):
        assert len(group["pairs"]) == spec["pairs_per_seed_per_baseline"] == 5
        for pair, row in enumerate(group["pairs"]):
            assert row["index"] == pair
            assert row["order"] == (["candidate", "baseline"] if (index + pair) % 2 == 0 else ["baseline", "candidate"])
            means = {}
            for name in ["candidate", "baseline"]:
                durations = np.asarray(row[name]["evaluation_seconds"], dtype=np.float64)
                assert durations.ndim == 1 and len(durations) >= spec["window_minimum_evaluations"]
                assert np.isfinite(durations).all() and np.all(durations > 0)
                assert durations.sum() >= spec["window_minimum_seconds"]
                means[name] = float(durations.mean())
                assert np.isclose(means[name], row[name]["mean_seconds"], rtol=1e-12, atol=0)
            ratio = means["baseline"] / means["candidate"]
            assert np.isclose(ratio, row["ratio"], rtol=1e-12, atol=0)
            ratios.append(ratio)
            candidate_means.append(means["candidate"])
            baseline_means.append(means["baseline"])
    logs = np.log(ratios)
    rng = np.random.default_rng(spec["bootstrap_seed"])
    samples = np.exp(rng.choice(logs, (spec["bootstrap_resamples"], len(logs))).mean(1))
    ratio = float(np.exp(logs.mean()))
    interval = np.quantile(samples, [.025, .975]).tolist()
    assert np.isclose(ratio, report["paired_geometric_mean_ratio"], rtol=1e-12, atol=0)
    assert np.allclose(interval, report["paired_bootstrap_95_percent_interval"], rtol=1e-12, atol=0)
    result = {"state": "PASS", "reader": "NumPy only; no Torch imports",
        "gradient_and_objective_comparisons": 78, "saved_arrays": 156, "paired_windows_checked": 20,
        "maximum_error_over_original_budget": max(errors), "ratio": ratio, "interval": interval,
        "candidate_mean_seconds": float(np.mean(candidate_means)),
        "baseline_mean_seconds": float(np.mean(baseline_means)),
        "fresh_native_coordinate_AD_repeated": False, "complete_solver_speedup": None}
    result["activation_mode"] = mode
    if report.get("zero_gate_folding", False):
        from box376.audit_zero_gate_folding import audit_folding
        result["zero_gate_folding"] = audit_folding(args.proof, args.inputs, args.reference, report)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
