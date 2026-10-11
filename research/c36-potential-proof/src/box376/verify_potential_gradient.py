"""Finite CPU-only experimental potential proof; no optimizer or GPU work.

Run on the coordinator test machine. The full Main parameter gradient is checked
on a declared subset of the unchanged physical source cache. This is not a full
4096-point qualification, training run, accepted flow or concurrency result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch

import torch_c17 as c17
from box376.c36_coordinate_oracle import native_c17
from box376.data import flatten, initialize, torch_data, views
from box376.frozen_data import load_cache
from box376.normalized_objective import PreparedNormalizedADObjective, PreparedNormalizedObjective, normalization_key
from box376.potential_geometry import native_field
from box376.potential_model import forward, parameter_vjp, prepare_geometry

TOPOLOGY = (4, 128, 128, 128, 128, 4)
SEEDS = (361901, 361902)


def compare(actual, reference):
    actual = actual.detach().numpy() if isinstance(actual, torch.Tensor) else actual
    reference = reference.detach().numpy() if isinstance(reference, torch.Tensor) else reference
    assert actual.shape == reference.shape and actual.dtype == reference.dtype == np.float64
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    error = np.abs(actual - reference)
    maximum = float(np.max(error / (1e-11 + 1e-9 * np.abs(reference))))
    assert maximum <= 1, ("Original derivative comparison budget exceeded", maximum)
    return {"entries": int(actual.size), "maximum_error_over_original_budget": maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    record = {"state": "RUNNING", "CPU_only": True, "GPU_workload_created": False,
              "optimizer_updates": 0, "complete_physical_acceptance_evaluated": False,
              "GPU_qualification": False, "training_converged": False,
              "high_concurrency_qualified": False, "formal_protocol_frozen": False,
              "state_channels": 36, "parameter_count": 50692,
              "representation": "prescribed geometry features; learned curl potential and pressure",
              "exact_velocity_inserted": False, "source_depends_on_parameters": False,
              "atol": 1e-11, "rtol": 1e-9, "tolerance_relaxed": False, "checks": []}
    arrays = {}
    try:
        data = torch_data(load_cache(args.inputs / "training-source.npz"))
        calibration = json.loads((args.inputs / "calibration.json").read_text())
        data["normalization_key"] = normalization_key(calibration)
        assert data["nu"] == .01 and data["eta"] == 1/32
        explicit = PreparedNormalizedObjective(data, calibration)
        independent = PreparedNormalizedADObjective(data, calibration, "a24")
        family = data["families"]["interior"]
        indices = np.linspace(0, len(family["points"]) - 1, 17, dtype=np.int64)
        points = family["points"][indices].clone()
        force, df, weights = (family[key][indices].clone() for key in ("force", "force_gradient", "weights"))
        inputs, gate = prepare_geometry(points, data["eta"])
        assert torch.isfinite(inputs).all() and torch.isfinite(gate).all()
        record.update(training_cache_key=data["metadata"]["cache_key"],
                      normalization_key=data["normalization_key"], N=17,
                      selected_interior_indices=indices.tolist(),
                      checked_families=["interior"], full_4096_point_objective_checked=False,
                      potential_scale=.01, pressure_scale=.1)
        arrays.update(points=points.numpy(), force=force.numpy(), force_gradient=df.numpy(),
                      weights=weights.numpy(), geometry_input_states=inputs.numpy(), geometry_gate_states=gate.numpy())
        for seed in SEEDS:
            theta = flatten(initialize(TOPOLOGY, seed))
            parameters = views(theta, TOPOLOGY)
            state, tape = forward(parameters, inputs, gate)
            loss, bar = explicit.interior(state, force, df, weights, data["nu"], data["lambda_g"])
            gradient = flatten(parameter_vjp(tape, bar))
            native_theta = theta.detach().clone().requires_grad_(True)
            native_parameters = views(native_theta, TOPOLOGY)
            x = points.detach().clone().requires_grad_(True)
            reference = native_c17(lambda p: native_field(native_parameters, p, data["eta"]), x)
            reference_loss = independent.independent_interior(reference, force, df, weights, data["nu"], data["lambda_g"])
            reference_gradient = torch.autograd.grad(reference_loss, native_theta)[0]
            divergence = sum(state[:, i, 1+i] for i in range(3))
            divergence_gradient = sum(c17.unpack(state)[2][:, i, i, :] for i in range(3))
            check = {"seed": seed, "state": compare(state, reference),
                     "loss": compare(loss.reshape(1), reference_loss.reshape(1)),
                     "complete_parameter_gradient": compare(gradient, reference_gradient),
                     "maximum_absolute_divergence": float(divergence.abs().max()),
                     "maximum_absolute_divergence_gradient": float(divergence_gradient.abs().max())}
            record["checks"].append(check)
            for name, value in {"theta": theta, "C36_state": state, "native_state": reference,
                                "C36_loss": loss, "native_loss": reference_loss,
                                "C36_gradient": gradient, "native_gradient": reference_gradient}.items():
                arrays[f"seed{seed}_{name}"] = value.detach().numpy().copy()
            print(json.dumps({"seed": seed, "scope": "N17 CPU potential derivative proof",
                              "maximum_gradient_error_over_original_budget": check["complete_parameter_gradient"]["maximum_error_over_original_budget"]}), flush=True)
        record.update(state="EXPERIMENTAL_POTENTIAL_SUBSET_COMPLETE_PARAMETER_GRADIENT_PROOF_PASS",
                      complete_gradient_comparisons=2, complete_gradient_entries=2*50692)
    except BaseException:
        record.update(state="FAILED", traceback=traceback.format_exc())
        raise
    finally:
        path = args.output / "raw-proof.npz"
        np.savez_compressed(path, **arrays)
        record["raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / "proof.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
