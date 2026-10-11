"""Recompute the complete CUDA proof comparisons from saved NumPy arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_proof(directory):
    record = json.loads((directory / "proof.json").read_text())
    raw = directory / "raw-proof.npz"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["raw_sha256"]
    with np.load(raw, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    return record, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-proof", type=Path, required=True)
    parser.add_argument("--cpu-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    gpu_record, gpu = read_proof(args.gpu_proof)
    cpu_record, cpu = read_proof(args.cpu_proof)
    assert gpu_record["status"] == "PASS"
    assert cpu_record["state"] == "EXPERIMENTAL_POTENTIAL_SUBSET_COMPLETE_PARAMETER_GRADIENT_PROOF_PASS"
    assert gpu_record["cpu_reference_sha256"] == cpu_record["raw_sha256"]
    assert gpu_record["atol"] == cpu_record["atol"] == 1e-11
    assert gpu_record["rtol"] == cpu_record["rtol"] == 1e-9
    assert gpu_record["N"] == cpu_record["N"] == 17
    assert gpu_record["selected_interior_indices"] == cpu_record["selected_interior_indices"]
    for name in ["training_cache_key", "normalization_key"]:
        assert gpu_record[name] == cpu_record[name]
    seeds = (361901, 361902)
    fixed = {"points", "force", "force_gradient", "weights",
             "geometry_input_states", "geometry_gate_states"}
    outputs = {f"seed{seed}_{name}" for seed in seeds for name in
               ["theta", "C36_state", "C36_loss", "C36_gradient", "native_state", "native_loss", "native_gradient"]}
    assert set(gpu) == set(cpu) == fixed | outputs
    checks = []

    def compare(name, actual, expected):
        assert actual.dtype == expected.dtype == np.float64
        assert actual.shape == expected.shape
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        ratio = float(np.max(np.abs(actual - expected) / (1e-11 + 1e-9 * np.abs(expected))))
        assert ratio <= 1, (name, ratio)
        checks.append({"name": name, "entries": actual.size, "maximum_error_over_original_budget": ratio})

    for name in ["points", "force", "force_gradient", "weights"]:
        assert np.array_equal(gpu[name], cpu[name]), name
    for name in ["geometry_input_states", "geometry_gate_states"]:
        compare(name + "_vs_cpu", gpu[name], cpu[name])
    for seed in seeds:
        assert gpu[f"seed{seed}_C36_gradient"].shape == (50692,)
        assert gpu[f"seed{seed}_native_gradient"].shape == (50692,)
        assert np.array_equal(gpu[f"seed{seed}_theta"], cpu[f"seed{seed}_theta"])
        for name in ["theta", "C36_state", "C36_loss", "C36_gradient",
                     "native_state", "native_loss", "native_gradient"]:
            key = f"seed{seed}_{name}"
            compare(key + "_vs_cpu", gpu[key], cpu[key])
        for name in ["state", "loss", "gradient"]:
            compare(f"seed{seed}_{name}_gpu_vs_native", gpu[f"seed{seed}_C36_{name}"],
                    gpu[f"seed{seed}_native_{name}"])
    assert len(checks) == 22
    producer = {row["name"]: row for row in gpu_record["checks"]}
    assert set(producer) == {row["name"] for row in checks}
    for row in checks:
        assert producer[row["name"]] == row, row["name"]
    result = {"status": "PASS", "audit_backend": "NumPy on CPU; producer comparison function not imported",
              "raw_arrays": len(gpu), "comparisons": len(checks), "complete_gradient_entries": 101384,
              "atol": 1e-11, "rtol": 1e-9, "gpu_raw_sha256": gpu_record["raw_sha256"],
              "cpu_raw_sha256": cpu_record["raw_sha256"], "checks": checks}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "comparisons": len(checks), "raw_arrays": len(gpu)}))


if __name__ == "__main__":
    main()
