"""Independently recompute expanded CUDA comparisons from saved arrays using NumPy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--cpu-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    record = json.loads((args.proof / "proof.json").read_text())
    assert record["status"] == "PASS" and record["atol"] == 1e-11 and record["rtol"] == 1e-9
    assert record["seeds"] == [361901, 361902] and record["parameter_count"] == 50692
    assert record["interior_points"] == 4096 and record["native_reference_points"] == 131
    assert record["geometry_chunk"] == 256
    assert record["candidate_chunks"] == [128, 64] and record["native_chunk"] == 66
    raw = args.proof / "raw-proof.npz"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["raw_sha256"]
    source = args.inputs / "training-source.npz"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record["source_file_sha256"]
    with np.load(raw, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    with np.load(source, allow_pickle=False) as archive:
        for name in ["points", "force", "force_gradient", "weights"]:
            assert np.array_equal(arrays[name], archive["interior_" + name])
    indices = np.unique(np.r_[np.linspace(0, 4095, 129, dtype=np.int64), [1965, 2489]])
    assert len(indices) == 131 and record["regression_source_indices"] == [1965, 2489]
    anchor_indices = np.linspace(0, 4095, 17, dtype=np.int64)
    cpu_record = json.loads((args.cpu_proof / "proof.json").read_text())
    cpu_path = args.cpu_proof / "raw-proof.npz"
    assert cpu_record["state"] == "EXPERIMENTAL_POTENTIAL_SUBSET_COMPLETE_PARAMETER_GRADIENT_PROOF_PASS"
    assert hashlib.sha256(cpu_path.read_bytes()).hexdigest() == record["cpu_reference_sha256"] == cpu_record["raw_sha256"]
    with np.load(cpu_path, allow_pickle=False) as archive:
        cpu_reference = {key: archive[key].copy() for key in archive.files}
    for key in ["points", "force", "force_gradient", "weights"]:
        assert np.array_equal(arrays[key][anchor_indices], cpu_reference[key]), key
    assert np.array_equal(arrays["native_indices"], indices)
    assert record["selected_interior_indices"] == indices.tolist()
    expected = {"points", "force", "force_gradient", "weights", "native_indices",
                "geometry_input_states", "geometry_gate_states"}
    checks = []

    def check(name, a, b):
        assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
        assert np.isfinite(a).all() and np.isfinite(b).all()
        ratio = float(np.max(np.abs(a - b) / (1e-11 + 1e-9 * np.abs(b))))
        assert ratio <= 1, (name, ratio)
        checks.append({"name": name, "entries": a.size, "maximum_error_over_original_budget": ratio})

    for key in ["geometry_input_states", "geometry_gate_states"]:
        assert np.isfinite(arrays[key]).all()
        check(key + "_anchor_vs_cpu", arrays[key][anchor_indices], cpu_reference[key])
    assert np.array_equal(arrays["geometry_gate_states"][2489], np.zeros((1, 36)))
    for seed in (361901, 361902):
        expected.add(f"seed{seed}_theta")
        assert arrays[f"seed{seed}_theta"].shape == (50692,)
        assert np.array_equal(arrays[f"seed{seed}_theta"], cpu_reference[f"seed{seed}_theta"])
        for label, points, chunks in [("full128", 4096, 32), ("full64", 4096, 64),
                                      ("subset", 131, 2), ("native", 131, 2), ("anchor", 17, 1)]:
            for suffix, shape in [("state", (points, 4, 17)), ("chunk_losses", (chunks,)),
                                  ("chunk_gradients", (chunks, 50692))]:
                key = f"seed{seed}_{label}_{suffix}"
                expected.add(key)
                assert arrays[key].shape == shape and arrays[key].dtype == np.float64
                assert np.isfinite(arrays[key]).all()
        for name, suffix in [("state", "state"), ("loss", "chunk_losses"), ("gradient", "chunk_gradients")]:
            a = arrays[f"seed{seed}_anchor_{suffix}"]
            if name != "state":
                a = a.sum(axis=0)
            check(f"seed{seed}_anchor_{name}_vs_cpu", a, cpu_reference[f"seed{seed}_C36_{name}"])
        for name, suffix in [("state", "state"), ("loss", "chunk_losses"), ("gradient", "chunk_gradients")]:
            a, b = (arrays[f"seed{seed}_full{chunk}_{suffix}"] for chunk in (128, 64))
            if name != "state":
                a, b = a.sum(axis=0), b.sum(axis=0)
            check(f"seed{seed}_full_partition_{name}", a, b)
        for name, suffix in [("state", "state"), ("loss", "chunk_losses"), ("gradient", "chunk_gradients")]:
            a, b = (arrays[f"seed{seed}_{label}_{suffix}"] for label in ("subset", "native"))
            check(f"seed{seed}_native_chunk_{name}", a, b)
            if name != "state":
                check(f"seed{seed}_native_total_{name}", a.sum(axis=0), b.sum(axis=0))
        check(f"seed{seed}_subset_state_vs_full", arrays[f"seed{seed}_subset_state"],
              arrays[f"seed{seed}_full128_state"][indices])
    assert set(arrays) == expected and len(checks) == 26
    assert checks == record["checks"]
    result = {"status": "PASS", "audit_backend": "NumPy; no producer import",
              "raw_sha256": record["raw_sha256"], "raw_arrays": len(arrays), "comparisons": len(checks),
              "source_arrays_exact": True, "checks": checks}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))


if __name__ == "__main__":
    main()
