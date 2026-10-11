"""Check full interior partition consistency and an expanded independent CUDA oracle."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.c36_coordinate_oracle import native_c17
from box376.data import flatten, initialize, torch_data, views
from box376.frozen_data import load_cache
from box376.normalized_objective import (
    PreparedNormalizedADObjective, PreparedNormalizedObjective, normalization_key,
)
from box376.potential_geometry import native_field
from box376.potential_model import forward, parameter_vjp, prepare_geometry
from box376.verify_potential_gradient import SEEDS, TOPOLOGY, compare


def cpu(value):
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
    record = {"status": "RUNNING", "started": time.time(), "device": "cuda",
              "seeds": list(SEEDS), "parameter_count": 50692, "interior_points": 4096,
              "native_reference_points": 131, "geometry_chunk": 256, "candidate_chunks": [128, 64], "native_chunk": 66,
              "atol": 1e-11, "rtol": 1e-9, "tolerance_relaxed": False,
              "checked_families": ["interior"], "full_4096_point_objective_checked": False,
              "full_interior_native_reference_checked": False, "optimizer_updates": 0,
              "training_converged": False, "physical_acceptance_evaluated": False,
              "checks": [], "diagnostic_stage_seconds": {}}
    arrays = {}

    def progress(phase):
        record["phase"] = phase
        (args.output / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"phase": phase, "time": time.time()}), flush=True)

    def check(name, actual, expected):
        record["checks"].append({"name": name, **compare(actual, expected)})

    def timed(name, fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        record["diagnostic_stage_seconds"][name] = time.perf_counter() - start
        return result

    try:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        record.update(torch=torch.__version__, numpy=np.__version__, cuda_runtime=torch.version.cuda,
                      gpu_name=torch.cuda.get_device_name(0), dtype="float64",
                      timing_scope="diagnostic calls; no comparative speed claim")
        data = torch_data(load_cache(args.inputs / "training-source.npz"))
        for family in data["families"].values():
            for key, value in family.items():
                if isinstance(value, torch.Tensor):
                    family[key] = value.to("cuda")
        calibration = json.loads((args.inputs / "calibration.json").read_text())
        data["normalization_key"] = normalization_key(calibration)
        explicit = PreparedNormalizedObjective(data, calibration)
        independent = PreparedNormalizedADObjective(data, calibration, "a24")
        family = data["families"]["interior"]
        assert len(family["points"]) == 4096 and data["nu"] == .01 and data["eta"] == 1 / 32
        record.update(training_cache_key=data["metadata"]["cache_key"],
                      normalization_key=data["normalization_key"],
                      source_file_sha256=hashlib.sha256((args.inputs / "training-source.npz").read_bytes()).hexdigest())
        for key in ["points", "force", "force_gradient", "weights"]:
            arrays[key] = cpu(family[key])
        indices = np.unique(np.r_[np.linspace(0, 4095, 129, dtype=np.int64), [1965, 2489]])
        assert len(indices) == 131
        record["regression_source_indices"] = [1965, 2489]
        anchor_indices = np.linspace(0, 4095, 17, dtype=np.int64)
        cpu_record = json.loads((args.cpu_proof / "proof.json").read_text())
        assert cpu_record["state"] == "EXPERIMENTAL_POTENTIAL_SUBSET_COMPLETE_PARAMETER_GRADIENT_PROOF_PASS"
        cpu_path = args.cpu_proof / "raw-proof.npz"
        assert hashlib.sha256(cpu_path.read_bytes()).hexdigest() == cpu_record["raw_sha256"]
        record["cpu_reference_sha256"] = cpu_record["raw_sha256"]
        with np.load(cpu_path, allow_pickle=False) as archive:
            cpu_reference = {key: archive[key].copy() for key in archive.files}
        for key in ["points", "force", "force_gradient", "weights"]:
            assert np.array_equal(arrays[key][anchor_indices], cpu_reference[key]), key
        arrays["native_indices"] = indices
        record["selected_interior_indices"] = indices.tolist()
        progress("geometry_4096")
        def geometry():
            input_chunks, gate_chunks = [], []
            for begin in range(0, 4096, 256):
                progress(f"geometry_{begin}_to_{begin + 256}")
                local_inputs, local_gate = prepare_geometry(family["points"][begin:begin + 256], data["eta"])
                input_chunks.append(local_inputs)
                gate_chunks.append(local_gate)
            return torch.cat(input_chunks), torch.cat(gate_chunks)

        inputs, gate = timed("geometry_4096", geometry)
        assert inputs.is_cuda and gate.is_cuda
        assert torch.isfinite(inputs).all() and torch.isfinite(gate).all()
        arrays["geometry_input_states"], arrays["geometry_gate_states"] = cpu(inputs), cpu(gate)

        for key in ["geometry_input_states", "geometry_gate_states"]:
            check(key + "_anchor_vs_cpu", arrays[key][anchor_indices], cpu_reference[key])

        for seed in SEEDS:
            theta = flatten(initialize(TOPOLOGY, seed)).to("cuda")
            assert theta.numel() == 50692
            arrays[f"seed{seed}_theta"] = cpu(theta)
            assert np.array_equal(arrays[f"seed{seed}_theta"], cpu_reference[f"seed{seed}_theta"])
            parameters = views(theta, TOPOLOGY)

            def candidate(selected, chunk, label):
                states, losses, gradients = [], [], []
                for begin in range(0, len(selected), chunk):
                    idx = selected[begin:begin + chunk]
                    state, tape = forward(parameters, inputs[idx], gate[idx])
                    loss, bar = explicit.interior(state, family["force"][idx],
                                                  family["force_gradient"][idx], family["weights"][idx],
                                                  data["nu"], data["lambda_g"])
                    gradient = flatten(parameter_vjp(tape, bar))
                    assert all(v.is_cuda and v.dtype == torch.float64 for v in (state, loss, gradient))
                    states.append(cpu(state))
                    losses.append(cpu(loss))
                    gradients.append(cpu(gradient))
                prefix = f"seed{seed}_{label}"
                arrays[prefix + "_state"] = np.concatenate(states)
                arrays[prefix + "_chunk_losses"] = np.asarray(losses)
                arrays[prefix + "_chunk_gradients"] = np.stack(gradients)

            candidate(anchor_indices, 17, "anchor")
            for name, suffix in [("state", "_state"), ("loss", "_chunk_losses"),
                                 ("gradient", "_chunk_gradients")]:
                actual = arrays[f"seed{seed}_anchor" + suffix]
                if name != "state":
                    actual = actual.sum(axis=0)
                check(f"seed{seed}_anchor_{name}_vs_cpu", actual, cpu_reference[f"seed{seed}_C36_{name}"])
            for chunk in (128, 64):
                label = f"full{chunk}"
                progress(f"seed{seed}_{label}")
                timed(f"seed{seed}_{label}", lambda: candidate(np.arange(4096), chunk, label))
            for name in ("state", "loss", "gradient"):
                suffix = {"state": "_state", "loss": "_chunk_losses", "gradient": "_chunk_gradients"}[name]
                a, b = (arrays[f"seed{seed}_full{chunk}" + suffix] for chunk in (128, 64))
                if name != "state":
                    a, b = a.sum(axis=0), b.sum(axis=0)
                check(f"seed{seed}_full_partition_{name}", a, b)
            progress(f"seed{seed}_subset_candidate")
            timed(f"seed{seed}_subset_candidate", lambda: candidate(indices, 66, "subset"))

            def native():
                states, losses, gradients = [], [], []
                for begin in range(0, len(indices), 66):
                    progress(f"seed{seed}_native_{begin}")
                    idx = indices[begin:begin + 66]
                    native_theta = theta.detach().clone().requires_grad_(True)
                    x = family["points"][idx].detach().clone().requires_grad_(True)
                    reference = native_c17(lambda p: native_field(views(native_theta, TOPOLOGY), p, data["eta"]), x)
                    loss = independent.independent_interior(reference, family["force"][idx],
                                                           family["force_gradient"][idx], family["weights"][idx],
                                                           data["nu"], data["lambda_g"])
                    gradient = torch.autograd.grad(loss, native_theta)[0]
                    states.append(cpu(reference))
                    losses.append(cpu(loss))
                    gradients.append(cpu(gradient))
                    del reference, loss, gradient, native_theta, x
                prefix = f"seed{seed}_native"
                arrays[prefix + "_state"] = np.concatenate(states)
                arrays[prefix + "_chunk_losses"] = np.asarray(losses)
                arrays[prefix + "_chunk_gradients"] = np.stack(gradients)

            timed(f"seed{seed}_native", native)
            for name, suffix in [("state", "_state"), ("loss", "_chunk_losses"),
                                 ("gradient", "_chunk_gradients")]:
                a, b = (arrays[f"seed{seed}_{label}" + suffix] for label in ("subset", "native"))
                check(f"seed{seed}_native_chunk_{name}", a, b)
                if name != "state":
                    check(f"seed{seed}_native_total_{name}", a.sum(axis=0), b.sum(axis=0))
            check(f"seed{seed}_subset_state_vs_full", arrays[f"seed{seed}_subset_state"],
                  arrays[f"seed{seed}_full128_state"][indices])
            progress(f"seed{seed}_pass")
        record.update(status="PASS", complete_gradient_entries_per_comparison=101384,
                      full_interior_partition_consistency_checked=True)
    except BaseException:
        record.update(status="FAILED", failure=traceback.format_exc())
        raise
    finally:
        record["finished"] = time.time()
        if torch.cuda.is_initialized():
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        raw = args.output / "raw-proof.npz"
        np.savez_compressed(raw, **arrays)
        record["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        (args.output / "proof.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"status": record["status"], "checks": len(record["checks"])}), flush=True)


if __name__ == "__main__":
    main()
