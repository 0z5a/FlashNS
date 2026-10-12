"""NumPy-only reconstruction of the normalized potential-objective CUDA proof."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FAMILIES = ("interior", "initial", "boundary", "gauge")
COEFFICIENTS = np.array([1., 5., 5., 5.])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proof", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    record = json.loads((args.proof / "proof.json").read_text())
    raw = args.proof / "raw-proof.npz"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["raw_sha256"]
    with np.load(raw, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    assert record["status"] == "PASS" and record["atol"] == 1e-11 and record["rtol"] == 1e-9
    assert record["seeds"] == [361901, 361902] and record["parameter_count"] == 50692
    assert record["family_counts"] == [4096, 512, 384, 7]
    native_all = record["full_interior_native_reference_checked"]
    assert isinstance(native_all, bool)
    native_interior = 4096 if native_all else 131
    native_points = native_interior + 512 + 384 + 7
    assert record["fresh_native_family_counts"] == [native_interior, 512, 384, 7]
    assert record["full_candidate_points"] == 4999 and record["fresh_native_points"] == native_points
    assert record["checked_families"] == list(FAMILIES) and record["diagnostic_family_counts"] == [17] * 4
    assert record["geometry_chunk"] == 256 and record["candidate_chunks"] == [128, 64]
    native_chunk = record["native_interior_chunk"]
    assert native_chunk in (32, 66, 128, 256) and record["optimizer_updates"] == 0
    assert record["tolerance_relaxed"] is False
    for flag in ("full_candidate_partition_consistency_checked", "fresh_native_parameter_gradient_checked",
                 "nonzero_value_family_gradients_checked", "caller_mutation_isolated"):
        assert record[flag] is True
    source = args.inputs / "training-source.npz"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == record["source_file_sha256"] == "d378c28991dd9a852297f22c94f0727d4c0c1e5d4140fe718324859c09b78bc2"
    assert record["training_cache_key"] == "6f6121bd890390ed89d20790797a36288b5f855bd59ece191e4d5a8cf85b8dc7"
    calibration = json.loads((args.inputs / "calibration.json").read_text())
    payload = {"calibration_raw_sha256": calibration["raw_sha256"], "source_nu": calibration["source_nu"],
               "source_eta": calibration["source_eta"], "scales": calibration["candidate_fixed_physical_scales"]}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert key == record["normalization_key"] == "46710e73f2b699396512563fce4fca39b0ae1906cae4b9573e14f809647a9164"
    assert np.array_equal(arrays["coefficients"], COEFFICIENTS)
    indices = np.unique(np.r_[np.linspace(0, 4095, 129, dtype=np.int64), [1965, 2489]])
    if native_all:
        indices = np.arange(4096, dtype=np.int64)
    assert np.array_equal(arrays["native_indices"], indices)
    selected = arrays["diagnostic_indices"]
    assert selected.shape == (17,) and selected.dtype == np.int64 and len(np.unique(selected)) == 17
    assert selected.min() >= 0 and selected.max() < 4096
    expected = {"coefficients", "native_indices", "diagnostic_indices"}
    with np.load(source, allow_pickle=False) as archive:
        for label in ("canonical", "subset", "diagnostic"):
            denominators = arrays[label + "_denominators"]
            expected.add(label + "_denominators")
            assert denominators.dtype == np.float64 and denominators.shape == (4,)
            assert np.isfinite(denominators).all() and np.all(denominators > 0)
            for i, name in enumerate(FAMILIES):
                for field in ("points", "weights", "target") + (("force", "force_gradient") if name == "interior" else ()):
                    key = label + "_" + name + "_" + field
                    value = arrays[key]
                    expected.add(key)
                    assert value.dtype == np.float64 and np.isfinite(value).all()
                    if label == "diagnostic":
                        if field == "weights":
                            reference = np.linspace(.3, 1.7, 17) * (i + 1)
                            assert np.allclose(value, reference, atol=1e-15, rtol=0)
                        elif field == "target":
                            reference = np.arange(1, 18)[:, None] * .002 * np.array([1., -1., 1., -1.])
                            assert np.array_equal(value, reference)
                        else:
                            assert np.array_equal(value, archive["interior_" + field][selected])
                    else:
                        reference = archive[name + "_" + field]
                        if label == "subset" and name == "interior":
                            reference = reference[indices]
                        assert np.array_equal(value, reference), key
                if label == "diagnostic":
                    assert abs(denominators[i] - arrays[label + "_" + name + "_weights"].sum()) < 1e-13
                else:
                    assert abs(denominators[i] - archive[name + "_weights"].sum()) < 1e-13
    checks = []

    def comparison(name, a, b, append=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype == np.float64
        assert np.isfinite(a).all() and np.isfinite(b).all()
        ratio = float(np.max(np.abs(a - b) / (1e-11 + 1e-9 * np.abs(b))))
        assert ratio <= 1, (name, ratio)
        if append:
            checks.append({"name": name, "entries": int(a.size), "maximum_error_over_original_budget": ratio})

    totals = {}
    for seed in (361901, 361902):
        expected.add(f"seed{seed}_theta")
        assert arrays[f"seed{seed}_theta"].shape == (50692,) and arrays[f"seed{seed}_theta"].dtype == np.float64
        for label, counts, chunks, data_label in (
                ("full128", [4096, 512, 384, 7], [128] * 4, "canonical"),
                ("full64", [4096, 512, 384, 7], [64] * 4, "canonical"),
                ("subset", [native_interior, 512, 384, 7], [64] * 4, "subset"),
                ("subset_native", [native_interior, 512, 384, 7], [native_chunk, 128, 128, 128], "subset"),
                ("diagnostic", [17] * 4, [7] * 4, "diagnostic"),
                ("diagnostic_native", [17] * 4, [native_chunk, 128, 128, 128], "diagnostic")):
            prefix = f"seed{seed}_" + label
            for suffix in ("family", "chunk_numerators", "chunk_gradients", "loss", "gradient", "numerators", "family_gradients"):
                expected.add(prefix + "_" + suffix)
            family = arrays[prefix + "_family"]
            reference = np.repeat(np.arange(4), [(n + c - 1) // c for n, c in zip(counts, chunks)])
            assert family.dtype == np.int64 and np.array_equal(family, reference)
            numerator = arrays[prefix + "_chunk_numerators"]
            gradient = arrays[prefix + "_chunk_gradients"]
            assert numerator.shape == (len(family),) and gradient.shape == (len(family), 50692)
            n = np.array([numerator[family == i].sum() for i in range(4)])
            fg = np.stack([gradient[family == i].sum(0) for i in range(4)])
            loss = (n * COEFFICIENTS / arrays[data_label + "_denominators"]).sum()
            totals[label] = (loss, gradient.sum(0), n)
            comparison(prefix + "_family_gradient_reconstruction", arrays[prefix + "_family_gradients"], fg, append=False)
            if label.endswith("native"):
                for name, reference in zip(("loss", "gradient", "numerators"), totals[label]):
                    comparison(prefix + "_reconstruction_" + name, arrays[prefix + "_" + name], reference, append=False)
            elif label.startswith("full"):
                for name, reference in zip(("loss", "gradient", "numerators"), totals[label]):
                    comparison(prefix + "_aggregate_" + name, arrays[prefix + "_" + name], reference)
        for name in ("loss", "gradient", "numerators"):
            comparison(f"seed{seed}_full_partition_" + name, arrays[f"seed{seed}_full128_" + name], arrays[f"seed{seed}_full64_" + name])
        for label in ("subset", "diagnostic"):
            prefix = f"seed{seed}_" + label
            for name, reference in zip(("loss", "gradient", "numerators"), totals[label]):
                comparison(prefix + "_aggregate_" + name, arrays[prefix + "_" + name], reference)
            for name in ("loss", "gradient", "numerators", "family_gradients"):
                comparison(prefix + "_fresh_native_" + name, arrays[prefix + "_" + name], arrays[prefix + "_native_" + name])
            if label == "diagnostic":
                assert np.all(np.linalg.norm(arrays[prefix + "_native_family_gradients"], axis=1) > 0)
                for name in ("loss", "gradient", "numerators"):
                    key = prefix + "_frozen_" + name
                    expected.add(key)
                    comparison(prefix + "_caller_mutation_" + name, arrays[key], arrays[prefix + "_" + name])
    assert set(arrays) == expected
    assert checks == record["checks"] and len(checks) == 52
    result = {"status": "PASS", "reader": "independent NumPy recomputation", "raw_arrays": len(arrays),
              "comparisons": len(checks), "raw_sha256": record["raw_sha256"], "checks": checks,
              "full_candidate_points": 4999, "fresh_native_points": native_points,
              "full_interior_native_reference_checked": native_all, "diagnostic_points": 68}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}))


if __name__ == "__main__":
    main()
