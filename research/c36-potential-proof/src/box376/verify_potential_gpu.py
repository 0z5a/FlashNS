"""Qualify the C36 potential model on CUDA against native AD and frozen CPU arrays."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch_c17 as c17

from box376.c36_coordinate_oracle import native_c17
from box376.data import flatten, initialize, torch_data, views
from box376.frozen_data import load_cache
from box376.normalized_objective import (
    PreparedNormalizedADObjective, PreparedNormalizedObjective, normalization_key,
)
from box376.potential_geometry import native_field
from box376.potential_model import forward, parameter_vjp, prepare_geometry
from box376.verify_potential_gradient import SEEDS, TOPOLOGY, compare


def cpu_array(value):
    return value.detach().cpu().numpy().copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--cpu-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    record = {"status": "RUNNING", "device": "cuda", "N": 17,
              "seeds": list(SEEDS), "parameter_count": 50692,
              "atol": 1e-11, "rtol": 1e-9, "tolerance_relaxed": False,
              "full_4096_point_objective_checked": False, "optimizer_updates": 0,
              "training_converged": False, "physical_acceptance_evaluated": False,
              "checks": [], "diagnostic_stage_seconds": {}, "started": time.time()}
    arrays = {}

    def progress(phase):
        record["phase"] = phase
        (args.output / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"phase": phase, "time": time.time()}), flush=True)

    def check(name, actual, expected):
        actual = cpu_array(actual) if isinstance(actual, torch.Tensor) else actual
        expected = cpu_array(expected) if isinstance(expected, torch.Tensor) else expected
        row = {"name": name, **compare(actual, expected)}
        record["checks"].append(row)
        return row

    def timed(name, function):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        record["diagnostic_stage_seconds"][name] = time.perf_counter() - start
        return value

    try:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        record.update(torch=torch.__version__, numpy=np.__version__,
                      cuda_runtime=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0),
                      gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                      dtype="float64", timing_scope="diagnostic first calls; not a speed benchmark")
        cpu_record = json.loads((args.cpu_proof / "proof.json").read_text())
        assert cpu_record["state"] == "EXPERIMENTAL_POTENTIAL_SUBSET_COMPLETE_PARAMETER_GRADIENT_PROOF_PASS"
        assert cpu_record["complete_gradient_entries"] == 101384
        raw_cpu = args.cpu_proof / "raw-proof.npz"
        assert hashlib.sha256(raw_cpu.read_bytes()).hexdigest() == cpu_record["raw_sha256"]
        record["cpu_reference_sha256"] = cpu_record["raw_sha256"]
        with np.load(raw_cpu, allow_pickle=False) as source:
            cpu = {name: source[name].copy() for name in source.files}
        data = torch_data(load_cache(args.inputs / "training-source.npz"))
        for family in data["families"].values():
            for name, value in family.items():
                if isinstance(value, torch.Tensor):
                    family[name] = value.to(device)
        calibration = json.loads((args.inputs / "calibration.json").read_text())
        data["normalization_key"] = normalization_key(calibration)
        assert data["nu"] == .01 and data["eta"] == 1 / 32
        explicit = PreparedNormalizedObjective(data, calibration)
        independent = PreparedNormalizedADObjective(data, calibration, "a24")
        family = data["families"]["interior"]
        indices = np.linspace(0, len(family["points"]) - 1, 17, dtype=np.int64)
        record.update(selected_interior_indices=indices.tolist(),
                      training_cache_key=data["metadata"]["cache_key"],
                      normalization_key=data["normalization_key"], checked_families=["interior"])
        points = family["points"][indices].clone()
        force, df, weights = (family[key][indices].clone()
                              for key in ("force", "force_gradient", "weights"))
        for name, value in {"points": points, "force": force, "force_gradient": df,
                            "weights": weights}.items():
            arrays[name] = cpu_array(value)
            assert np.array_equal(arrays[name], cpu[name]), name
        progress("geometry_on_cuda")
        torch.cuda.reset_peak_memory_stats()
        inputs, gate = timed("geometry", lambda: prepare_geometry(points, data["eta"]))
        assert inputs.is_cuda and gate.is_cuda and inputs.dtype == gate.dtype == torch.float64
        for name, value in {"geometry_input_states": inputs, "geometry_gate_states": gate}.items():
            arrays[name] = cpu_array(value)
            check(name + "_vs_cpu", value, cpu[name])
        for seed in SEEDS:
            progress(f"seed_{seed}_candidate")
            theta = flatten(initialize(TOPOLOGY, seed)).to(device)
            assert np.array_equal(cpu_array(theta), cpu[f"seed{seed}_theta"])
            parameters = views(theta, TOPOLOGY)

            def candidate():
                state, tape = forward(parameters, inputs, gate)
                loss, bar = explicit.interior(state, force, df, weights, data["nu"], data["lambda_g"])
                gradient = flatten(parameter_vjp(tape, bar))
                return state, loss, gradient

            state, loss, gradient = timed(f"seed_{seed}_candidate", candidate)
            assert all(value.is_cuda and value.dtype == torch.float64
                       for value in (state, loss, gradient))
            for name, value in {"theta": theta, "C36_state": state,
                                "C36_loss": loss, "C36_gradient": gradient}.items():
                key = f"seed{seed}_{name}"
                arrays[key] = cpu_array(value)
                check(key + "_vs_cpu", value, cpu[key])
            progress(f"seed_{seed}_native_coordinate_ad")

            def native():
                native_theta = theta.detach().clone().requires_grad_(True)
                native_parameters = views(native_theta, TOPOLOGY)
                x = points.detach().clone().requires_grad_(True)
                reference = native_c17(lambda p: native_field(native_parameters, p, data["eta"]), x)
                reference_loss = independent.independent_interior(
                    reference, force, df, weights, data["nu"], data["lambda_g"])
                reference_gradient = torch.autograd.grad(reference_loss, native_theta)[0]
                return reference.detach(), reference_loss.detach(), reference_gradient.detach()

            reference, reference_loss, reference_gradient = timed(f"seed_{seed}_native_ad", native)
            values = {"native_state": reference, "native_loss": reference_loss,
                      "native_gradient": reference_gradient}
            for name, value in values.items():
                key = f"seed{seed}_{name}"
                arrays[key] = cpu_array(value)
                check(key + "_vs_cpu", value, cpu[key])
            for name, actual, expected in [("state", state, reference), ("loss", loss, reference_loss),
                                          ("gradient", gradient, reference_gradient)]:
                check(f"seed{seed}_{name}_gpu_vs_native", actual, expected)
            divergence = sum(state[:, i, 1+i] for i in range(3))
            divergence_gradient = sum(c17.unpack(state)[2][:, i, i, :] for i in range(3))
            record.setdefault("divergence", []).append({
                "seed": seed, "maximum_absolute_divergence": float(divergence.abs().max()),
                "maximum_absolute_divergence_gradient": float(divergence_gradient.abs().max())})
            progress(f"seed_{seed}_pass")
        record.update(status="PASS", complete_gradient_entries=101384,
                      gpu_native_gradient_comparisons=2, cpu_reference_gradient_comparisons=4)
    except BaseException:
        record.update(status="FAILED", failure=traceback.format_exc())
        raise
    finally:
        record["finished"] = time.time()
        if torch.cuda.is_initialized():
            record["whole_process_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["whole_process_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        raw = args.output / "raw-proof.npz"
        np.savez_compressed(raw, **arrays)
        record["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        (args.output / "proof.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"status": record["status"], "checks": len(record["checks"])}), flush=True)


if __name__ == "__main__":
    main()
