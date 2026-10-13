"""Observe a trained C36 network through native AD and independent physics."""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

from box376.c36_coordinate_oracle import native_c17
from box376.data import views
from box376.potential_geometry import native_field
from box376.potential_run import cpu, sha, write_json
from box376.source import C0, H0, trajectory
from box376.validation import field_metrics, field_passes_proposed_gate, fields_from_state, weighted_rms
from box376.validation_data import make_holdouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "protocol", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--observer-chunk", type=int, default=128)
    args = parser.parse_args()
    if args.observer_chunk < 1:
        parser.error("--observer-chunk must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    calibration = json.loads((args.inputs / "calibration.json").read_text())
    scales = calibration["candidate_fixed_physical_scales"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["protocol_sha256"] == sha(args.protocol)
    assert checkpoint["representation"] == "C36-curl-potential"
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    if args.device == "cuda":
        assert torch.cuda.device_count() == 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parameters = views(checkpoint["theta"].to(args.device), tuple(protocol["topology"]))
    datasets, raw, metadata = make_holdouts(seed=protocol["physical"]["holdout_seed"])
    report = {"state": "RUNNING", "started_unix": time.time(), "protocol_sha256": sha(args.protocol),
              "checkpoint_sha256": sha(args.checkpoint), "validation_metadata": metadata,
              "observer": "native coordinate AD", "consumer": "independent NumPy",
              "device": args.device, "observer_chunk": args.observer_chunk,
              "datasets": {}, "physical_acceptance_pass": False}
    raw["theta"] = checkpoint["theta"].numpy().copy()

    def progress(phase):
        report["phase"] = phase
        write_json(args.output / "report.json", report)
        print(json.dumps({"phase": phase, "unix": time.time()}), flush=True)

    def values(points):
        x = torch.from_numpy(np.ascontiguousarray(points)).to(args.device).requires_grad_(True)
        return cpu(native_field(parameters, x))

    for name, dataset in datasets.items():
        states = []
        for begin in range(0, len(dataset["points"]), args.observer_chunk):
            x = torch.from_numpy(dataset["points"][begin:begin + args.observer_chunk].copy()).to(args.device).requires_grad_(True)
            states.append(cpu(native_c17(lambda p: native_field(parameters, p), x)))
            progress(f"{name}-native-{begin}")
        state = np.concatenate(states)
        points = dataset["points"]
        gauge = values(np.c_[np.full((len(points), 3), 1/16), points[:, 3]])[:, 3]
        fields = fields_from_state(state)
        metrics, residuals = field_metrics(fields, dataset["reference"], dataset["weights"], gauge, scales)
        regional = {}
        for region in np.unique(dataset["regions"]):
            selected = dataset["regions"] == region
            metric, _ = field_metrics({k: v[selected] for k, v in fields.items()},
                {k: v[selected] for k, v in dataset["reference"].items()}, dataset["weights"][selected],
                gauge[selected], scales)
            regional[str(int(region))] = metric
        report["datasets"][name] = {"global": metrics, "regions": regional,
                                    "field_gate_pass": field_passes_proposed_gate(metrics)}
        raw.update({name + "_native_state": state, name + "_gauge": gauge})
        raw.update({name + "_" + key: value for key, value in residuals.items()})
        # Seal partial physical evidence after each dataset as well as at completion.
        np.savez_compressed(args.output / (name + "-fields.npz"), state=state, gauge=gauge)
        progress(name + "-complete")
    for name in ("initial", "boundary"):
        actual = values(raw[name + "_points"])
        raw[name + "_native_values"] = actual
        report[name + "_normalized_velocity_rms"] = weighted_rms(actual[:, :3], np.ones(len(actual))) / scales["velocity"]

    initial = np.array(list(itertools.product((-1., 0., 1.), repeat=3))) * H0 + C0
    times = np.unique(np.r_[np.linspace(0, 1, 129), .25, .375, .5, .625, .75])
    exact = trajectory(times[:, None], initial[None, :, :])
    records, previous = [], None
    raw.update(particle_initial=initial, particle_times=times, particle_exact=exact)

    def rhs(t, flat):
        points = np.c_[flat.reshape(-1, 3), np.full(len(initial), t)]
        return values(points)[:, :3].reshape(-1)

    for index, (rtol, atol, max_step) in enumerate(protocol["physical"]["trajectory_refinements"]):
        solution = solve_ivp(rhs, (0, 1), initial.reshape(-1), method="DOP853", rtol=rtol, atol=atol,
                             max_step=max_step, dense_output=True)
        assert solution.success, solution.message
        actual = solution.sol(times).T.reshape(len(times), len(initial), 3)
        assert np.isfinite(actual).all()
        error = np.linalg.norm(actual - exact, axis=-1)
        row = {"rtol": rtol, "atol": atol, "max_step": max_step, "nfev": solution.nfev,
               "maximum_path_error": float(error.max()), "maximum_endpoint_error": float(error[-1].max())}
        if previous is not None:
            row["maximum_refinement_change"] = float(np.linalg.norm(actual - previous, axis=-1).max())
        raw[f"particle_integrated{index}"] = actual
        previous = actual
        records.append(row)
        progress(f"trajectory-refinement{index}-complete")
    report["trajectories"] = records
    spec = protocol["physical"]
    trajectory_pass = records[-1]["maximum_refinement_change"] <= spec["maximum_trajectory_refinement_change"] and records[-1]["maximum_endpoint_error"] <= spec["maximum_trajectory_endpoint_error"]
    report["physical_acceptance_pass"] = (
        all(report["datasets"][name]["field_gate_pass"] for name in spec["gated_datasets"])
        and all(report[name + "_normalized_velocity_rms"] <= spec["initial_boundary_velocity_normalized_rms_limit"] for name in ("initial", "boundary"))
        and trajectory_pass)
    np.savez_compressed(args.output / "raw-physical.npz", **raw)
    report.update(state="COMPLETE", finished_unix=time.time(), raw_sha256=sha(args.output / "raw-physical.npz"))
    progress("COMPLETE")


if __name__ == "__main__":
    main()
