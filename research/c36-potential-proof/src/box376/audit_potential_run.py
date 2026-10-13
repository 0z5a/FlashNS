"""NumPy-only audit of saved timings, training checks and physical observations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from box376.source import source, trajectory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    ratio = float(np.max(np.abs(a - b) / (1e-11 + 1e-9 * np.abs(b))))
    assert ratio <= 1, ratio
    return ratio


def rms(value, weights):
    assert np.isfinite(value).all() and np.isfinite(weights).all() and np.all(weights >= 0)
    return float(np.sqrt(np.sum(weights * np.mean(value.reshape(len(value), -1)**2, axis=1)) / weights.sum()))


def validated_saved_source(raw, name):
    analytic = source(raw[name + "_points"])
    saved = {key: raw[name + "_source_" + key] for key in analytic}
    for key, value in analytic.items():
        close(saved[key], value)
    # Residuals subtract the frozen force bytes. Re-generating those bytes on a
    # different NumPy build can amplify last-bit rounding through cancellation.
    return saved


def physical_metrics(state, exact, gauge, weights, scales):
    n = len(state)
    assert state.shape == (n, 4, 17) and state.dtype == np.float64 and np.isfinite(state).all()
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    hessian = np.empty((n, 4, 3, 3), dtype=np.float64)
    for slot, (i, j) in enumerate(pairs):
        hessian[:, :, i, j] = hessian[:, :, j, i] = state[:, :, 4 + slot]
    u, g, h = state[:, :3, 0], state[:, :3, 1:4], hessian[:, :3]
    residual = state[:, :3, 13] + np.einsum("nj,nij->ni", u, g) - .01 * np.trace(h, axis1=2, axis2=3) + state[:, 3, 1:4] - exact["f"]
    spatial = state[:, :3, 14:17] + np.einsum("nij,njk->nik", g, g) + np.einsum("nj,nijk->nik", u, h) - .01 * state[:, :3, 10:13] + hessian[:, 3] - exact["df"]
    divergence = np.trace(g, axis1=1, axis2=2)
    dg = sum(h[:, i, i, :] for i in range(3))
    delta, denominator = u - exact["U"], rms(exact["U"], weights)
    metrics = {"velocity_relative_l2": rms(delta, weights) / denominator if denominator > 0 else None,
        "velocity_reference_is_zero": denominator == 0,
        "normalized_velocity_error_rms": rms(delta, weights) / scales["velocity"],
        "normalized_divergence_rms": rms(divergence, weights) / scales["continuity_residual"],
        "normalized_divergence_gradient_rms": rms(dg, weights) / scales["continuity_residual_gradient"],
        "normalized_momentum_rms": rms(residual, weights) / scales["momentum_residual"],
        "normalized_momentum_gradient_rms": rms(spatial, weights) / scales["momentum_residual_gradient"],
        "normalized_corrected_pressure_rms": rms(state[:, 3, 0] - gauge, weights) / scales["pressure"],
        "normalized_pressure_gauge_rms": rms(gauge, weights) / scales["pressure"]}
    return metrics, (residual, spatial, divergence, dg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proof", "inputs", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--training", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    report = json.loads((args.proof / "report.json").read_text())
    protocol = json.loads(args.protocol.read_text())
    assert report["state"] == "COMPLETE" and report["protocol_sha256"] == sha(args.protocol)
    result = {"status": "PASS", "reader": "NumPy only; saved-array audit", "comparisons": 0,
              "independent_optimizer_trajectory_replay": False, "complete_solver_speedup": None}
    actual_prefix = report.get("gradient_actual_prefix", "explicit")
    reference_prefix = report.get("gradient_reference_prefix", "reference")
    if "mode" in report:
        source_file = args.inputs / "training-source.npz"
        assert sha(source_file) == protocol["source_sha256"]
        meta = json.loads(source_file.with_suffix(".json").read_text())
        factors = np.array([1., 5., 5., 5.]) / np.array([meta["global_weight_sums"][n] for n in ("interior", "initial", "boundary", "gauge")])
        for group in report["groups"]:
            assert group["state"] == "COMPLETE"
            for row in group["raw_files"]:
                path = args.proof / row["filename"]
                assert sha(path) == row["sha256"]
                with np.load(path, allow_pickle=False) as raw:
                    assert len(raw.files) == row["arrays"] and all(np.isfinite(raw[k]).all() for k in raw.files)
                    assert raw["theta"].shape == (50692,) and raw["theta"].dtype == np.float64
                    if actual_prefix + "_loss" in raw:
                        for name in ("loss", "gradient", "numerators"):
                            close(raw[actual_prefix + "_" + name], raw[reference_prefix + "_" + name])
                            result["comparisons"] += 1
                        close(raw[actual_prefix + "_loss"], (raw[actual_prefix + "_numerators"] * factors).sum())
                    else:
                        close(raw["loss"], (raw["numerators"] * factors).sum())
            if report["mode"] == "training":
                spec = protocol["training"]
                assert group["Adam_updates"] == spec["Adam_updates_per_seed"]
                assert group["LBFGS_calls"] == spec["LBFGS_calls_per_seed"]
                assert group["LBFGS_closures"] <= spec["LBFGS_closure_cap_per_seed"]
                events = [json.loads(line) for line in (args.proof / f"seed{group['seed']}-updates.jsonl").read_text().splitlines()]
                assert [e["update"] for e in events if e["phase"] == "Adam"] == list(range(1, spec["Adam_updates_per_seed"] + 1))
                assert [e["closure"] for e in events if e["phase"] == "LBFGS"] == list(range(group["LBFGS_closures"]))
                for row in group["checkpoints"]:
                    assert sha(args.proof / row["filename"]) == row["sha256"]
                with np.load(args.proof / f"seed{group['seed']}-final-gradient.npz", allow_pickle=False) as final:
                    close(group["final_loss"], final[actual_prefix + "_loss"])
                    close(group["final_gradient_infinity"], np.abs(final[actual_prefix + "_gradient"]).max())
                converged = group["final_loss"] <= spec["loss_convergence_limit"] and group["final_gradient_infinity"] <= spec["gradient_infinity_convergence_limit"]
                assert group["training_converged"] == converged
            else:
                spec = protocol["performance"]
                assert len(group["pairs"]) == spec["pairs_per_seed_per_baseline"]
                for index, pair in enumerate(group["pairs"]):
                    expected = ["explicit", "reference"] if (index + protocol["seeds"].index(group["seed"])) % 2 == 0 else ["reference", "explicit"]
                    assert pair["order"] == expected
                    for name in expected:
                        durations = np.array(pair[name]["evaluation_seconds"])
                        assert np.isfinite(durations).all() and np.all(durations > 0)
                        close(pair[name]["mean_seconds"], durations.mean())
                        if name == "reference" and report["baseline"] == "native_coordinate_ad":
                            assert len(durations) == 1
                        else:
                            assert len(durations) >= 3 and durations.sum() >= 60
                    close(pair["speedup"], pair["reference"]["mean_seconds"] / pair["explicit"]["mean_seconds"])
        if report["mode"] == "timing":
            logs = np.log([p["speedup"] for g in report["groups"] for p in g["pairs"]])
            rng = np.random.default_rng(protocol["performance"]["bootstrap_seed"])
            samples = np.exp(rng.choice(logs, (protocol["performance"]["bootstrap_resamples"], len(logs))).mean(1))
            close(report["paired_geometric_mean_speedup"], np.exp(logs.mean()))
            close(report["paired_bootstrap_95_percent_interval"], np.quantile(samples, [.025, .975]))
            result["paired_speedup"] = report["paired_geometric_mean_speedup"]
    else:
        assert args.training is not None and sha(args.proof / "raw-physical.npz") == report["raw_sha256"]
        scales = json.loads((args.inputs / "calibration.json").read_text())["candidate_fixed_physical_scales"]
        with np.load(args.proof / "raw-physical.npz", allow_pickle=False) as raw:
            for key, digest in report["validation_metadata"]["array_hashes"].items():
                value = np.ascontiguousarray(raw[key])
                h = hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()
                assert h == digest
            final_files = sorted(args.training.glob("seed*-final-gradient.npz"))
            assert any(np.array_equal(raw["theta"], np.load(p, allow_pickle=False)["theta"]) for p in final_files)
            gates = {}
            for name, observed in report["datasets"].items():
                exact = validated_saved_source(raw, name)
                metrics, residuals = physical_metrics(raw[name + "_native_state"], exact, raw[name + "_gauge"], raw[name + "_weights"], scales)
                for key, value in metrics.items():
                    if value is None or isinstance(value, bool):
                        assert value == observed["global"][key]
                    else:
                        close(value, observed["global"][key])
                for key, value in zip(("residual", "spatial_residual_gradient", "divergence", "divergence_gradient"), residuals):
                    close(raw[name + "_" + key], value)
                limits = {k[:-6]: v for k, v in protocol["physical"].items() if k.endswith("_limit") and k[:-6] in metrics}
                gates[name] = all(metrics[k] is not None and 0 <= metrics[k] <= v for k, v in limits.items() if k != "velocity_relative_l2")
                gates[name] &= metrics["normalized_velocity_error_rms"] <= .001 if metrics["velocity_reference_is_zero"] else metrics["velocity_relative_l2"] <= .01
                assert gates[name] == observed["field_gate_pass"]
                result["comparisons"] += len(metrics) + len(residuals)
            close(raw["particle_exact"], trajectory(raw["particle_times"][:, None], raw["particle_initial"][None, :, :]))
            endpoint = float(np.linalg.norm(raw["particle_integrated1"][-1] - raw["particle_exact"][-1], axis=-1).max())
            refinement = float(np.linalg.norm(raw["particle_integrated1"] - raw["particle_integrated0"], axis=-1).max())
            close(report["trajectories"][-1]["maximum_endpoint_error"], endpoint)
            close(report["trajectories"][-1]["maximum_refinement_change"], refinement)
            acceptance = all(gates[n] for n in ("stratified", "uniform")) and endpoint <= .001 and refinement <= 1e-7
            for name in ("initial", "boundary"):
                metric = rms(raw[name + "_native_values"][:, :3], np.ones(len(raw[name + "_points"]))) / scales["velocity"]
                close(report[name + "_normalized_velocity_rms"], metric)
                acceptance &= metric <= .001
            assert acceptance == report["physical_acceptance_pass"]
            result["physical_acceptance_pass"] = bool(acceptance)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
