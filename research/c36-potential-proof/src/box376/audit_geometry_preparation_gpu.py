"""NumPy-only reader of the bounded geometry and downstream gradient proof."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    record = json.loads((path / "proof.json").read_text())
    raw = path / "raw-proof.npz"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["raw_sha256"]
    with np.load(raw, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return record, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proof", "inputs", "cpu-proof", "scale-proof", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    record, arrays = load(args.proof)
    prior, previous = load(args.scale_proof)
    cpu_record, anchor = load(args.cpu_proof)
    assert record["status"] == prior["status"] == "PASS"
    assert record["atol"] == prior["atol"] == 1e-11 and record["rtol"] == prior["rtol"] == 1e-9
    assert record["scale_reference_sha256"] == prior["raw_sha256"]
    assert record["cpu_reference_sha256"] == cpu_record["raw_sha256"]
    assert record["seeds"] == [361901, 361902] and record["parameter_count"] == 50692
    assert record["interior_points"] == 4096 and record["fresh_native_geometry_points"] == 131
    assert record["geometry_chunk"] == 256 and record["candidate_chunk"] == 128
    source = args.inputs / "training-source.npz"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record["source_file_sha256"] == prior["source_file_sha256"]
    with np.load(source, allow_pickle=False) as archive:
        assert np.array_equal(arrays["points"], archive["interior_points"])
    assert np.array_equal(arrays["points"], previous["points"])
    indices = np.unique(np.r_[np.linspace(0, 4095, 129, dtype=np.int64), [1965, 2489]])
    anchors = np.linspace(0, 4095, 17, dtype=np.int64)
    assert np.array_equal(arrays["native_indices"], indices)
    assert np.array_equal(arrays["points"][anchors], anchor["points"])
    checks = []

    def check(name, a, b):
        assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
        assert np.isfinite(a).all() and np.isfinite(b).all()
        ratio = float(np.max(np.abs(a - b) / (1e-11 + 1e-9 * np.abs(b))))
        assert ratio <= 1, (name, ratio)
        checks.append({"name": name, "entries": a.size, "maximum_error_over_original_budget": ratio})

    expected = {"points", "native_indices", "inputs", "gate", "native_inputs", "native_gate", "chunk13_inputs", "chunk13_gate"}
    for name, old, width in (("inputs", "geometry_input_states", 4), ("gate", "geometry_gate_states", 1)):
        assert arrays[name].shape == (4096, width, 36)
        assert arrays["native_" + name].shape == arrays["chunk13_" + name].shape == (131, width, 36)
        check(name + "_full_vs_frozen_native", arrays[name], previous[old])
        check(name + "_anchors_vs_cpu", arrays[name][anchors], anchor[old])
    flat = (arrays["points"][:, 3] <= .25) | (arrays["points"][:, 3] >= .75)
    assert np.array_equal(arrays["gate"][flat], np.zeros_like(arrays["gate"][flat]))
    assert np.array_equal(arrays["gate"][2489], np.zeros((1, 36)))
    for name in ("inputs", "gate"):
        check(name + "_fresh_native", arrays[name][indices], arrays["native_" + name])
    for name in ("inputs", "gate"):
        check(name + "_partition", arrays["chunk13_" + name], arrays[name][indices])
    for seed in (361901, 361902):
        key = f"seed{seed}_theta"
        expected.add(key)
        assert arrays[key].shape == (50692,) and np.array_equal(arrays[key], previous[key])
        for suffix, shape in (("state", (4096, 4, 17)), ("chunk_losses", (32,)), ("chunk_gradients", (32, 50692))):
            key = f"seed{seed}_" + suffix
            expected.add(key)
            assert arrays[key].shape == shape
            check(key + "_vs_frozen", arrays[key], previous[f"seed{seed}_full128_" + suffix])
            if suffix != "state":
                check(key + "_total_vs_frozen", arrays[key].sum(0), previous[f"seed{seed}_full128_" + suffix].sum(0))
    assert set(arrays) == expected
    assert checks == record["checks"] and len(checks) == 18
    result = {"status": "PASS", "reader": "independent NumPy recomputation", "raw_arrays": len(arrays),
              "comparisons": len(checks), "raw_sha256": record["raw_sha256"], "checks": checks}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}))


if __name__ == "__main__":
    main()
