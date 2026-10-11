"""Independent checkpoint gradients and physical gates for finite development.

The wider A24/native parameter AD reference is recomputed from observed arrays.
Hash-only intermediate Adam events are not counted as complete gradient audits.
No convergence is inferred from a finite optimizer process exiting successfully.
"""
import argparse
import itertools
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.data import FAMILIES, views, torch_data
from box376.frozen_data import load_cache
from box376.normalized_objective import PreparedNormalizedADObjective, normalization_key
from box376.source import C0, H0
from box376.train_normalized_development import TOPOLOGY, SEEDS, file_hash, array_hash, compare, write_json
from box376.validation import network_dataset_metrics, network_values, weighted_rms, integrate_network_trajectories
from box376.validation_data import make_holdouts


def equal_tensor_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual.cpu(), expected.cpu()) and torch.isfinite(actual).all()
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_tensor_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal_tensor_tree(a, b)
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("node", "Root", "offbox", "CPU"), required=True)
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument("--observed-references-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit_path = args.output/(args.scope+"-audit.json")
    assert not audit_path.exists()
    report = json.loads((args.output/"report.json").read_text())
    exits_path = args.output/"actual-exit.json" if report["device"] == "cpu" else args.output.parent/"exit.json"
    exits = json.loads(exits_path.read_text())
    actual_exit = exits["actual_exit_code"] if report["device"] == "cpu" else exits["compute_actual_exit_code"]
    assert actual_exit == 0
    assert report["state"] == "FINITE_NORMALIZED_DEVELOPMENT_CONTINUATION_FINISHED"
    assert report["topology"] == list(TOPOLOGY) and report["parameter_count"] == 50692
    assert report["development_only"] and not report["formal_training_protocol_frozen"]
    for field in ("CPU_fallback_used", "source_depends_on_parameters", "supervised_interior_velocity_term",
                  "NCCL_used", "gradient_clipping", "optimizer_inside_graph", "time_to_accuracy_measured",
                  "every_intermediate_Adam_full_gradient_retained"):
        assert report[field] is False
    assert report["every_LBFGS_candidate_and_full_gradient_retained"] is True
    assert report["runner_sha256"] == file_hash(Path(__file__).with_name("train_normalized_development.py"))
    for name, item in report["input_files"].items():
        assert Path(name).name == name
        path = args.inputs/name
        assert path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"]
    calibration = json.loads((args.inputs/"calibration.json").read_text())
    assert file_hash(args.inputs/"calibration.npz") == calibration["raw_sha256"]
    key = normalization_key(calibration)
    data = torch_data(load_cache(args.inputs/"training-source.npz"))
    data["normalization_key"] = key
    assert key == report["normalization_key"] and data["metadata"]["cache_key"] == report["training_cache_key"]
    assert [len(data["families"][n]["points"]) for n in FAMILIES] == [4096, 512, 384, 7]
    if args.observed_references_only and args.scope != "node":
        raise ValueError("Offbox/CPU audits must independently recompute A24")
    reference = PreparedNormalizedADObjective(data, calibration, "a24")
    assert file_hash(args.output/"events.jsonl") == report["events_sha256"]
    events = [json.loads(line) for line in (args.output/"events.jsonl").read_text().splitlines()]
    assert [g["seed"] for g in report["groups"]] == list(SEEDS)
    checks, physical, refs, validation_raw = [], [], {}, {}
    audit = {"state": "RUNNING", "scope": args.scope, "actual_compute_exit_code": actual_exit,
             "complete_gradient_checks": checks, "physical_results": physical,
             "all_intermediate_Adam_gradients_independently_verified": False,
             "complete_optimizer_trajectory_replay_claimed": False,
             "formal_training_protocol_frozen": False, "time_to_accuracy_measured": False,
             "physical_GPU_count": report["physical_GPU_count"], "training_converged": False}
    audit["independent_CPU_A24_recomputation_executed"] = not args.observed_references_only
    audit["observed_GPU_A24_array_check_only"] = args.observed_references_only
    if not args.skip_physical:
        datasets, held_raw, metadata = make_holdouts()
        audit["validation_key"] = metadata["validation_key"]
        particles = np.array(list(itertools.product((-1, 0, 1), repeat=3)))*H0+C0
        for name in ("initial_points", "boundary_points", "gauge_points"):
            validation_raw[name] = held_raw[name]

    def physical_check(seed, phase, theta):
        parameters = views(torch.from_numpy(theta.copy()), TOPOLOGY)
        result = {"seed": seed, "phase": phase, "parameter_sha256": array_hash(theta), "datasets": {}}
        prefix = f"seed{seed}-{phase}"
        for name, dataset in datasets.items():
            metrics, observed = network_dataset_metrics(parameters, dataset, calibration["candidate_fixed_physical_scales"])
            result["datasets"][name] = metrics
            for field, value in observed.items():
                validation_raw[prefix+"-"+name+"-"+field] = value
        for name in ("initial", "boundary"):
            values = network_values(parameters, held_raw[name+"_points"])
            validation_raw[prefix+"-"+name+"-values"] = values
            result["normalized_"+name+"_velocity_rms"] = weighted_rms(values[:, :3], np.ones(len(values)))/calibration["candidate_fixed_physical_scales"]["velocity"]
        result["trajectory"], paths = integrate_network_trajectories(parameters, particles)
        for name, value in paths.items():
            validation_raw[prefix+"-trajectory-"+name] = value
        result["proposed_combined_gate_pass"] = (
            all(result["datasets"][n]["proposed_field_gate_pass"] for n in ("stratified", "uniform"))
            and all(result["normalized_"+n+"_velocity_rms"] <= 1e-3 for n in ("initial", "boundary"))
            and result["trajectory"]["proposed_trajectory_gate_pass"])
        physical.append(result)
        print(json.dumps({"seed": seed, "phase": phase, "physical_gate": result["proposed_combined_gate_pass"],
                          "mixed_velocity_relative_l2": result["datasets"]["stratified"]["global"]["velocity_relative_l2"],
                          "endpoint_error": result["trajectory"]["records"][-1]["maximum_endpoint_error"]}), flush=True)

    try:
        raw_arrays = 0
        for group in report["groups"]:
            seed = group["seed"]
            assert group["state"] == "FINITE_DEVELOPMENT_SEED_FINISHED"
            assert group["parameter_storage_stable"] and group["normalization_constant_storage_stable"]
            assert group["actual_new_Adam_updates"] == report["Adam_final_absolute_update_budget"]-100
            assert group["actual_LBFGS_calls"] == report["lbfgs_calls_per_seed_budget"]
            assert group["actual_LBFGS_closures"] <= report["lbfgs_closure_cap_per_seed"]
            expected_adam_checks = [u for u in range(100, report["Adam_final_absolute_update_budget"]+1)
                                    if u == 100 or u == report["Adam_final_absolute_update_budget"]
                                    or u % report["Adam_complete_gradient_check_interval"] == 0]
            assert group["Adam_complete_gradient_check_updates"] == expected_adam_checks
            assert [e["index"] for e in group["retained_evaluations"] if e["phase"] == "Adam"] == expected_adam_checks
            assert [e["index"] for e in group["retained_evaluations"] if e["phase"] == "LBFGS"] == list(range(group["actual_LBFGS_closures"]))
            assert [e["index"] for e in group["retained_evaluations"] if e["phase"] == "final"] == ([0] if group["actual_LBFGS_calls"] else [])
            seed_events = [e for e in events if e["seed"] == seed]
            adam_events = [e for e in seed_events if e["phase"] == "Adam"]
            assert [e["index"] for e in adam_events] == list(range(100, report["Adam_final_absolute_update_budget"]+1))
            assert [e["index"] for e in adam_events if e["full_arrays_retained"]] == expected_adam_checks
            lbfgs_events = [e for e in seed_events if e["phase"] == "LBFGS"]
            assert [e["index"] for e in lbfgs_events] == list(range(group["actual_LBFGS_closures"]))
            event_index = {(e["phase"], e["index"]): e for e in seed_events}
            assert len(event_index) == len(seed_events)
            arrays = {}
            for item in group["raw_shards"]:
                assert Path(item["filename"]).name == item["filename"]
                path = args.output/item["filename"]
                assert path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"]
                with np.load(path, allow_pickle=False) as raw:
                    assert len(raw.files) == item["arrays"]
                    for name in raw.files:
                        assert name not in arrays and np.isfinite(raw[name]).all()
                        arrays[name] = raw[name].copy()
                    raw_arrays += len(raw.files)
            expected_arrays = set()
            resume = torch.load(args.inputs/f"seed{seed}-Adam-update100.pt", map_location="cpu", weights_only=True)
            assert array_hash(resume["theta"].numpy()) == group["resume_parameter_sha256"]
            assert np.array_equal(arrays["Adam-100_theta"], resume["theta"].numpy())
            for record in group["retained_evaluations"]:
                phase, index = record["phase"], record["index"]
                prefix = f"{phase}-{index}"
                theta, gradient = arrays[prefix+"_theta"], arrays[prefix+"_gradient"]
                assert theta.shape == gradient.shape == (50692,) and theta.dtype == gradient.dtype == np.float64
                observed = event_index[(phase, index)]
                assert array_hash(theta) == record["parameter_sha256"] == observed["parameter_sha256"]
                assert array_hash(gradient) == record["gradient_sha256"] == observed["gradient_sha256"]
                actual = [arrays[prefix+"_"+name] for name in ("loss", "gradient", "numerators")]
                assert actual[0].shape == () and actual[2].shape == (4,)
                assert float(actual[0]) == record["loss"] == observed["loss"]
                expected = ([torch.from_numpy(arrays[prefix+"_a24_"+name].copy())
                             for name in ("loss", "gradient", "numerators")]
                            if args.observed_references_only else
                            reference(views(torch.from_numpy(theta.copy()), TOPOLOGY), data, report["chunk"]))
                check = {"seed": seed, "phase": phase, "index": index, "parameter_sha256": array_hash(theta)}
                for name, a, b in zip(("loss", "gradient", "numerators"), actual, expected):
                    b = b.numpy().copy()
                    check[name] = compare(a, b)
                    refs[f"seed{seed}-{prefix}-"+name] = b
                    compare(arrays[prefix+"_a24_"+name], b)
                    expected_arrays.update((prefix+"_"+name, prefix+"_a24_"+name))
                expected_arrays.add(prefix+"_theta")
                checks.append(check)
            assert set(arrays) == expected_arrays
            for item in group["checkpoints"]:
                assert Path(item["filename"]).name == item["filename"]
                path = args.output/item["filename"]
                assert path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"]
                saved = torch.load(path, map_location="cpu", weights_only=True)
                assert saved["training_cache_key"] == report["training_cache_key"] and saved["normalization_key"] == key
                assert saved["source_version"] == 0 and saved["optimizer"] == item["optimizer"]
                assert array_hash(saved["theta"].numpy()) == item["parameter_sha256"]
                assert saved["actual_Adam_updates"] == item["absolute_Adam_update"]
                if item["optimizer"] == "Adam":
                    assert np.array_equal(saved["theta"].numpy(), arrays[f"Adam-{item['absolute_Adam_update']}_theta"])
                    state = saved["optimizer_state"]["state"][0]
                    assert float(state["step"]) == item["absolute_Adam_update"]
                    assert state["exp_avg"].shape == state["exp_avg_sq"].shape == (50692,)
                    assert torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()
                    hyper = saved["optimizer_state"]["param_groups"][0]
                    assert tuple(hyper["betas"]) == (.9, .999) and hyper["eps"] == 1e-8
                    assert hyper["weight_decay"] == 0 and hyper["foreach"] is False
                if item["filename"] == f"seed{seed}-Adam-100.pt":
                    equal_tensor_tree(saved["optimizer_state"]["state"], resume["optimizer_state"]["state"])
                if item["filename"] == f"seed{seed}-LBFGS-final.pt":
                    assert np.array_equal(saved["theta"].numpy(), arrays["final-0_theta"])
                    state = saved["optimizer_state"]["state"][0]
                    assert state["func_evals"] == group["actual_LBFGS_closures"]
                    assert 0 <= state["n_iter"] <= group["actual_LBFGS_calls"]*report["lbfgs_max_iter"]
            accepted = group.get("accepted_LBFGS_parameters", [])
            assert len(accepted) == group["actual_LBFGS_calls"]
            for item in accepted:
                path = args.output/item["filename"]
                assert path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"]
                assert array_hash(np.load(path, allow_pickle=False)) == item["parameter_sha256"]
            if accepted:
                assert np.array_equal(np.load(args.output/accepted[-1]["filename"], allow_pickle=False), arrays["final-0_theta"])
            if not args.skip_physical:
                physical_check(seed, "Adam100", arrays["Adam-100_theta"])
                physical_check(seed, "Adam-final", arrays[f"Adam-{report['Adam_final_absolute_update_budget']}_theta"])
                if group["actual_LBFGS_calls"]:
                    physical_check(seed, "LBFGS-final", arrays["final-0_theta"])
            print(json.dumps({"seed": seed, "full_gradient_audit_checks": len(group["retained_evaluations"])}), flush=True)
        audit.update(state="FINITE_NORMALIZED_DEVELOPMENT_CHECKPOINT_REFERENCE_AUDIT_PASS",
                     retained_candidate_complete_gradients_verified=len(checks),
                     independently_recomputed_gradient_entries=0 if args.observed_references_only else 50692*len(checks),
                     maximum_gradient_error_over_original_budget=max(c["gradient"]["maximum_error_over_budget"] for c in checks),
                     raw_arrays_byte_and_content_verified=raw_arrays,
                     actual_new_Adam_updates=report["actual_new_Adam_updates"],
                     actual_LBFGS_calls=report["actual_LBFGS_calls"], actual_LBFGS_closures=report["actual_LBFGS_closures"],
                     checkpoint_optimizer_moments_at_resume_exact=True, physical_validation_executed=not args.skip_physical,
                     proposed_physical_gates_met=None if args.skip_physical else all(
                         p["proposed_combined_gate_pass"] for p in physical if p["phase"] == ("LBFGS-final" if report["lbfgs_calls_per_seed_budget"] else "Adam-final")))
    except BaseException:
        audit.update(state="FAILED", traceback=traceback.format_exc())
        raise
    finally:
        reference_path = args.output/(args.scope+"-independent-references.npz")
        np.savez_compressed(reference_path, **refs)
        audit["independent_reference_raw_sha256"] = file_hash(reference_path)
        if validation_raw:
            path = args.output/(args.scope+"-physical-raw.npz")
            np.savez_compressed(path, **validation_raw)
            audit["physical_raw_sha256"] = file_hash(path)
        audit["audit_runner_sha256"] = file_hash(__file__)
        write_json(audit_path, audit)


if __name__ == "__main__":
    main()
