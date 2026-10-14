"""Audit every packed-gather comparison using saved arrays and NumPy only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    record = json.loads((args.proof / "proof.json").read_text())
    assert record["state"] == "PASS"
    assert record["points"] == 4999 and record["parameters_per_seed"] == 50692
    assert record["seeds"] == [361901, 361902]
    assert record["atol"] == 1e-11 and record["rtol"] == 1e-9
    assert record["tolerance_relaxed"] is False
    assert record["reference_sha256"] == "172174ea49a82a21e0616571ae303560d275317d8b5319c22ea49dbd628d8b66"
    raw = args.proof / "raw-proof.npz"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["raw_sha256"]
    variants = ("explicit128-native", "explicit64-native", "parameterAD-native",
                "nonzero-value-families", "explicit-graph-native", "explicit-graph-changed",
                "parameterAD-graph-native", "parameterAD-graph-changed")
    expected = {f"seed{seed}-{variant}_{kind}" for seed in record["seeds"]
                for variant in variants for kind in ("loss", "gradient", "numerators")}
    producer = {row["name"]: row for row in record["checks"]}
    assert len(record["checks"]) == len(producer) == len(expected) == 48
    assert set(producer) == expected
    checks = []
    with np.load(raw, allow_pickle=False) as arrays:
        required = {prefix + "_" + side + "_" + kind
                    for name in expected for prefix, kind in [name.rsplit("_", 1)]
                    for side in ("actual", "reference")}
        assert set(arrays.files) == required and len(arrays.files) == 96
        for name in sorted(expected):
            prefix, kind = name.rsplit("_", 1)
            actual, reference = arrays[prefix + "_actual_" + kind], arrays[prefix + "_reference_" + kind]
            assert actual.shape == reference.shape
            assert actual.dtype == reference.dtype == np.float64
            assert actual.size == {"loss": 1, "gradient": 50692, "numerators": 4}[kind]
            assert np.isfinite(actual).all() and np.isfinite(reference).all()
            ratio = float(np.max(np.abs(actual - reference) / (1e-11 + 1e-9 * np.abs(reference))))
            assert ratio <= 1, (name, ratio)
            row = {"name": name, "entries": actual.size, "maximum_error_over_original_budget": ratio}
            assert producer[name] == row
            checks.append(row)
    result = {"state": "PASS", "reader": "NumPy only; producer comparison function not imported",
              "checks": checks, "raw_arrays": 96, "raw_sha256": record["raw_sha256"],
              "training_converged": False, "physical_acceptance_evaluated": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"state": "PASS", "comparisons": len(checks), "raw_arrays": 96}))


if __name__ == "__main__":
    main()
