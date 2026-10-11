"""Qualify bounded geometry preparation against native AD and frozen CUDA arrays."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.c36_coordinate_oracle import native_c36
from box376.data import flatten, initialize, torch_data, views
from box376.frozen_data import load_cache
from box376.normalized_objective import PreparedNormalizedObjective, normalization_key
from box376.potential_geometry import features_and_gate
from box376.potential_model import forward, parameter_vjp, prepare_geometry
from box376.verify_potential_gradient import SEEDS, TOPOLOGY, compare


def cpu(value):
    return value.detach().cpu().numpy().copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "cpu-proof", "scale-proof", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    record = {"status": "RUNNING", "started": time.time(), "device": "cuda",
              "interior_points": 4096, "fresh_native_geometry_points": 131,
              "geometry_chunk": 256, "candidate_chunk": 128, "seeds": list(SEEDS),
              "parameter_count": 50692, "atol": 1e-11, "rtol": 1e-9,
              "fresh_native_parameter_gradient_checked": False,
              "full_objective_checked": False, "optimizer_updates": 0,
              "checks": [], "diagnostic_stage_seconds": {}}
    arrays = {}

    def check(name, a, b):
        record["checks"].append({"name": name, **compare(a, b)})

    def phase(name):
        record["phase"] = name
        (args.output / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"phase": name, "time": time.time()}), flush=True)

    def timed(name, fn):
        phase(name)
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        record["diagnostic_stage_seconds"][name] = time.perf_counter() - start
        return value

    try:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        record.update(torch=torch.__version__, numpy=np.__version__, cuda_runtime=torch.version.cuda,
                      gpu_name=torch.cuda.get_device_name(0), dtype="float64",
                      timing_scope="diagnostic setup calls only; no paired speed claim")
        references = {}
        for label, path in (("scale", args.scale_proof), ("cpu", args.cpu_proof)):
            proof = json.loads((path / "proof.json").read_text())
            raw = path / "raw-proof.npz"
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == proof["raw_sha256"]
            if label == "scale":
                assert proof["status"] == "PASS" and len(proof["checks"]) == 26
                assert proof["atol"] == 1e-11 and proof["rtol"] == 1e-9
            else:
                assert proof["state"] == "EXPERIMENTAL_POTENTIAL_SUBSET_COMPLETE_PARAMETER_GRADIENT_PROOF_PASS"
            record[label + "_reference_sha256"] = proof["raw_sha256"]
            with np.load(raw, allow_pickle=False) as archive:
                references[label] = {key: archive[key].copy() for key in archive.files}
        previous, anchor = references["scale"], references["cpu"]
        source = args.inputs / "training-source.npz"
        record["source_file_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        data = torch_data(load_cache(source))
        for family in data["families"].values():
            for key, value in family.items():
                if isinstance(value, torch.Tensor):
                    family[key] = value.to("cuda")
        calibration = json.loads((args.inputs / "calibration.json").read_text())
        data["normalization_key"] = normalization_key(calibration)
        objective = PreparedNormalizedObjective(data, calibration)
        family = data["families"]["interior"]
        assert len(family["points"]) == 4096 and data["eta"] == 1/32 and data["nu"] == .01
        record.update(training_cache_key=data["metadata"]["cache_key"], normalization_key=data["normalization_key"])
        arrays["points"] = cpu(family["points"])
        assert np.array_equal(arrays["points"], previous["points"])
        indices = np.unique(np.r_[np.linspace(0, 4095, 129, dtype=np.int64), [1965, 2489]])
        anchors = np.linspace(0, 4095, 17, dtype=np.int64)
        arrays["native_indices"] = indices
        torch.cuda.reset_peak_memory_stats()
        inputs, gate = timed("geometry_4096", lambda: prepare_geometry(family["points"], data["eta"]))
        record["geometry_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        arrays["inputs"], arrays["gate"] = cpu(inputs), cpu(gate)
        for name, old in (("inputs", "geometry_input_states"), ("gate", "geometry_gate_states")):
            check(name + "_full_vs_frozen_native", arrays[name], previous[old])
            check(name + "_anchors_vs_cpu", arrays[name][anchors], anchor[old])
        # A new coordinate-AD computation does not call prepare_geometry.
        def native():
            values = [[], []]
            for begin in range(0, len(indices), 64):
                for component in (0, 1):
                    x = family["points"][indices[begin:begin + 64]].detach().clone().requires_grad_(True)
                    state = native_c36(lambda p: features_and_gate(p, data["eta"])[component], x)
                    values[component].append(cpu(state))
                    del state, x
            return [np.concatenate(parts) for parts in values]
        arrays["native_inputs"], arrays["native_gate"] = timed("native_geometry_131", native)
        for name in ("inputs", "gate"):
            check(name + "_fresh_native", arrays[name][indices], arrays["native_" + name])
        small = timed("geometry_131_chunk13", lambda: prepare_geometry(family["points"][indices], data["eta"], chunk_size=13))
        for name, value in zip(("inputs", "gate"), small):
            arrays["chunk13_" + name] = cpu(value)
            check(name + "_partition", arrays["chunk13_" + name], arrays[name][indices])
        del small
        for seed in SEEDS:
            phase("gradient_4096_seed" + str(seed))
            theta = flatten(initialize(TOPOLOGY, seed)).to("cuda")
            arrays[f"seed{seed}_theta"] = cpu(theta)
            assert np.array_equal(arrays[f"seed{seed}_theta"], previous[f"seed{seed}_theta"])
            parameters = views(theta, TOPOLOGY)
            states, losses, gradients = [], [], []
            for begin in range(0, 4096, 128):
                sl = slice(begin, begin + 128)
                state, tape = forward(parameters, inputs[sl], gate[sl])
                loss, bar = objective.interior(state, family["force"][sl], family["force_gradient"][sl],
                                               family["weights"][sl], data["nu"], data["lambda_g"])
                gradient = flatten(parameter_vjp(tape, bar))
                states.append(cpu(state))
                losses.append(cpu(loss))
                gradients.append(cpu(gradient))
            for suffix, value in (("state", np.concatenate(states)), ("chunk_losses", np.asarray(losses)),
                                   ("chunk_gradients", np.stack(gradients))):
                key = f"seed{seed}_" + suffix
                arrays[key] = value
                check(key + "_vs_frozen", value, previous[f"seed{seed}_full128_" + suffix])
                if suffix != "state":
                    check(key + "_total_vs_frozen", value.sum(0), previous[f"seed{seed}_full128_" + suffix].sum(0))
        record.update(status="PASS", complete_gradient_entries_per_comparison=101384)
    except BaseException:
        record.update(status="FAILED", failure=traceback.format_exc())
        raise
    finally:
        record["finished"] = time.time()
        if torch.cuda.is_initialized():
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        raw = args.output / "raw-proof.npz"
        np.savez_compressed(raw, **arrays)
        record["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        (args.output / "proof.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"status": record["status"], "checks": len(record["checks"])}), flush=True)


if __name__ == "__main__":
    main()
