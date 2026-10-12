"""Qualify the normalized C36 potential objective on CUDA, including value families."""
import argparse
import copy
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.c36_coordinate_oracle import native_c17
from box376.data import COEFFICIENTS, FAMILIES, flatten, initialize, torch_data, views
from box376.frozen_data import load_cache
from box376.normalized_objective import PreparedNormalizedADObjective, normalization_key
from box376.potential_geometry import features_and_gate, native_field
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.verify_potential_gradient import SEEDS, TOPOLOGY, compare


def cpu(value):
    return value.detach().cpu().numpy().copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native-all", action="store_true",
                        help="Check every canonical interior point with chunked native CUDA autograd.")
    parser.add_argument("--native-interior-chunk", type=int, choices=(32, 66, 128, 256), default=66)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    arrays = {}
    record = {"status": "RUNNING", "started": time.time(), "device": "cuda",
              "seeds": list(SEEDS), "parameter_count": 50692, "atol": 1e-11, "rtol": 1e-9,
              "tolerance_relaxed": False, "checked_families": list(FAMILIES),
              "family_counts": [4096, 512, 384, 7], "full_candidate_points": 4999,
              "fresh_native_family_counts": [4096 if args.native_all else 131, 512, 384, 7],
              "fresh_native_points": 4999 if args.native_all else 1034,
              "diagnostic_family_counts": [17] * 4, "diagnostic_points": 68,
              "geometry_chunk": 256, "candidate_chunks": [128, 64],
              "native_interior_chunk": args.native_interior_chunk,
              "full_interior_native_reference_checked": False, "optimizer_updates": 0,
              "training_converged": False, "physical_acceptance_evaluated": False,
              "checks": [], "diagnostic_stage_seconds": {}}

    def progress(name):
        record["phase"] = name
        (args.output / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"phase": name, "time": time.time()}), flush=True)

    def timed(name, fn):
        progress(name)
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        record["diagnostic_stage_seconds"][name] = time.perf_counter() - started
        return value

    def check(name, actual, expected):
        record["checks"].append({"name": name, **compare(actual, expected)})

    def save_data(label, data):
        arrays[label + "_denominators"] = np.array([data["families"][n]["W"] for n in FAMILIES])
        for name in FAMILIES:
            for key in ("points", "weights", "target", "force", "force_gradient"):
                if key in data["families"][name]:
                    arrays[label + "_" + name + "_" + key] = cpu(data["families"][name][key])

    def collect(prefix, contributions, denominators, prototype):
        indices, numerators, gradients = [], [], []
        family_numerators = prototype.new_zeros(4)
        family_gradients = prototype.new_zeros((4, 50692))
        for index, numerator, gradient in contributions:
            indices.append(index)
            numerators.append(cpu(numerator))
            gradients.append(cpu(gradient))
            family_numerators[index] += numerator.detach()
            family_gradients[index] += gradient.detach()
        arrays[prefix + "_family"] = np.array(indices, dtype=np.int64)
        arrays[prefix + "_chunk_numerators"] = np.asarray(numerators)
        arrays[prefix + "_chunk_gradients"] = np.stack(gradients)
        arrays[prefix + "_numerators"] = cpu(family_numerators)
        arrays[prefix + "_family_gradients"] = cpu(family_gradients)
        arrays[prefix + "_gradient"] = cpu(family_gradients.sum(0))
        arrays[prefix + "_loss"] = cpu((family_numerators * prototype.new_tensor(COEFFICIENTS)
                                       / prototype.new_tensor(denominators)).sum())

    def candidate(prefix, objective, parameters, data, chunk):
        denominators = [data["families"][n]["W"] for n in FAMILIES]
        collect(prefix, objective.contributions(parameters, chunk), denominators, parameters[0][0])
        # Exercise the public aggregate interface separately from its iterator.
        aggregate = objective(parameters, chunk)
        indices = arrays[prefix + "_family"]
        raw = arrays[prefix + "_chunk_numerators"]
        summed = np.array([raw[indices == i].sum() for i in range(4)])
        expected = ((summed * np.array(COEFFICIENTS) / np.array(denominators)).sum(),
                    arrays[prefix + "_chunk_gradients"].sum(0), summed)
        for name, actual, reference in zip(("loss", "gradient", "numerators"), aggregate, expected):
            arrays[prefix + "_" + name] = cpu(actual)
            check(prefix + "_aggregate_" + name, arrays[prefix + "_" + name], np.asarray(reference))

    def native_contributions(theta, data, independent, prefix):
        scales = calibration["candidate_fixed_physical_scales"]
        for index, name in enumerate(FAMILIES):
            family = data["families"][name]
            chunk = args.native_interior_chunk if name == "interior" else 128
            for begin in range(0, len(family["points"]), chunk):
                progress(prefix + "_" + name + "_" + str(begin))
                sl = slice(begin, begin + chunk)
                native_theta = theta.detach().clone().requires_grad_(True)
                x = family["points"][sl].detach().clone().requires_grad_(True)
                parameters = views(native_theta, TOPOLOGY)
                if name == "interior":
                    state = native_c17(lambda p: native_field(parameters, p, data["eta"]), x)
                    numerator = independent.independent_interior(state, family["force"][sl],
                                                                family["force_gradient"][sl], family["weights"][sl],
                                                                data["nu"], data["lambda_g"])
                else:
                    state = native_field(parameters, x, data["eta"])
                    components = slice(3, 4) if name == "gauge" else slice(0, 3)
                    scale = scales["pressure"] if name == "gauge" else scales["velocity"]
                    difference = state[:, components] - family["target"][sl, components]
                    numerator = (family["weights"][sl] * difference.square().sum(-1) / scale**2).sum()
                gradient = torch.autograd.grad(numerator * (COEFFICIENTS[index] / family["W"]), native_theta)[0]
                yield index, numerator.detach(), gradient.detach()
                del state, numerator, gradient, parameters, native_theta, x

    try:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        record.update(torch=torch.__version__, numpy=np.__version__, cuda_runtime=torch.version.cuda,
                      gpu_name=torch.cuda.get_device_name(0), dtype="float64",
                      timing_scope="diagnostic calls; no paired performance claim")
        source = args.inputs / "training-source.npz"
        record["source_file_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        assert record["source_file_sha256"] == "d378c28991dd9a852297f22c94f0727d4c0c1e5d4140fe718324859c09b78bc2"
        data = torch_data(load_cache(source))
        for family in data["families"].values():
            for key, value in family.items():
                if isinstance(value, torch.Tensor):
                    family[key] = value.to("cuda")
        calibration = json.loads((args.inputs / "calibration.json").read_text())
        data["normalization_key"] = normalization_key(calibration)
        assert [len(data["families"][n]["points"]) for n in FAMILIES] == record["family_counts"]
        record.update(training_cache_key=data["metadata"]["cache_key"], normalization_key=data["normalization_key"])
        arrays["coefficients"] = np.array(COEFFICIENTS)
        save_data("canonical", data)
        indices = np.unique(np.r_[np.linspace(0, 4095, 129, dtype=np.int64), [1965, 2489]])
        if args.native_all:
            indices = np.arange(4096, dtype=np.int64)
        arrays["native_indices"] = indices
        subset = copy.deepcopy(data)
        for key, value in subset["families"]["interior"].items():
            if isinstance(value, torch.Tensor):
                subset["families"]["interior"][key] = value[indices].clone()
        # Subset contributions retain the canonical global denominators.
        save_data("subset", subset)
        _, gate = features_and_gate(data["families"]["interior"]["points"], data["eta"])
        eligible = torch.nonzero(gate[:, 0].abs() > 1e-6).flatten()
        assert len(eligible) >= 17
        selected = eligible[torch.linspace(0, len(eligible) - 1, 17, device="cuda").long()]
        arrays["diagnostic_indices"] = cpu(selected)
        challenge = copy.deepcopy(data)
        for index, name in enumerate(FAMILIES):
            weights = torch.linspace(.3, 1.7, 17, device="cuda") * (index + 1)
            target = torch.arange(1, 18, device="cuda")[:, None] * .002 * weights.new_tensor([1, -1, 1, -1])
            family = {"points": data["families"]["interior"]["points"][selected].clone(),
                      "weights": weights, "target": target, "W": float(weights.sum())}
            if name == "interior":
                for key in ("force", "force_gradient"):
                    family[key] = data["families"]["interior"][key][selected].clone()
            challenge["families"][name] = family
        save_data("diagnostic", challenge)
        full_objective = timed("prepare_canonical_4999", lambda: PreparedPotentialNormalizedObjective(data, calibration))
        subset_objective = timed("prepare_native_reference_" + str(record["fresh_native_points"]),
                                 lambda: PreparedPotentialNormalizedObjective(subset, calibration))
        challenge_objective = timed("prepare_diagnostic_68", lambda: PreparedPotentialNormalizedObjective(challenge, calibration))
        independent = PreparedNormalizedADObjective(data, calibration, "a24")
        for seed in SEEDS:
            theta = flatten(initialize(TOPOLOGY, seed)).to("cuda")
            parameters = views(theta, TOPOLOGY)
            arrays[f"seed{seed}_theta"] = cpu(theta)
            for chunk in (128, 64):
                prefix = f"seed{seed}_full{chunk}"
                timed(prefix, lambda: candidate(prefix, full_objective, parameters, data, chunk))
            for name in ("loss", "gradient", "numerators"):
                check(f"seed{seed}_full_partition_" + name, arrays[f"seed{seed}_full128_" + name],
                      arrays[f"seed{seed}_full64_" + name])
            for label, local_data, objective, chunk in (("subset", subset, subset_objective, 64),
                                                       ("diagnostic", challenge, challenge_objective, 7)):
                prefix = f"seed{seed}_" + label
                timed(prefix, lambda: candidate(prefix, objective, parameters, local_data, chunk))
                native_prefix = prefix + "_native"
                denominators = [local_data["families"][n]["W"] for n in FAMILIES]
                timed(native_prefix, lambda: collect(native_prefix, native_contributions(theta, local_data, independent, native_prefix),
                                                    denominators, theta))
                for name in ("loss", "gradient", "numerators", "family_gradients"):
                    check(prefix + "_fresh_native_" + name, arrays[prefix + "_" + name], arrays[native_prefix + "_" + name])
                if label == "diagnostic":
                    assert np.all(np.linalg.norm(arrays[native_prefix + "_family_gradients"], axis=1) > 0)
                    # Mutating the original caller tensors must leave the prepared objective intact.
                    caller = copy.deepcopy(challenge)
                    frozen = PreparedPotentialNormalizedObjective(caller, calibration)
                    for family in caller["families"].values():
                        for value in family.values():
                            if isinstance(value, torch.Tensor):
                                value.add_(.125)
                        family["W"] *= 2
                    caller["nu"] *= 2
                    caller["normalization_key"] = "changed-after-preparation"
                    for name, value in zip(("loss", "gradient", "numerators"), frozen(parameters, 7)):
                        arrays[prefix + "_frozen_" + name] = cpu(value)
                        check(prefix + "_caller_mutation_" + name, arrays[prefix + "_frozen_" + name], arrays[prefix + "_" + name])
            progress(f"seed{seed}_complete")
        assert len(record["checks"]) == 52
        record.update(status="PASS", complete_gradient_entries_per_comparison=101384,
                      full_candidate_partition_consistency_checked=True, fresh_native_parameter_gradient_checked=True,
                      full_interior_native_reference_checked=args.native_all,
                      nonzero_value_family_gradients_checked=True, caller_mutation_isolated=True)
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
