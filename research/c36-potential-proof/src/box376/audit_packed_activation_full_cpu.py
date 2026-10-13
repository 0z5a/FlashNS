"""NumPy-only audit of canonical packed/compiled/analytic C36 CPU proofs."""
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
    report = json.loads((args.proof / "proof.json").read_text())
    assert report["state"] == "PASS" and report["device"] == "cpu"
    assert report["points"] == 4999 and report["parameters_per_seed"] == 50692
    assert report["seeds"] == [361901, 361902]
    assert report["atol"] == 1e-11 and report["rtol"] == 1e-9 and not report["tolerance_relaxed"]
    assert report["reference_provenance"]["complete_native_GPU_raw_sha256"] == "172174ea49a82a21e0616571ae303560d275317d8b5319c22ea49dbd628d8b66"
    assert report["reference_provenance"]["preserved_gather_raw_sha256"] == "7d1db3ce8fb6dbaeeeabff46e4e981633b1761c5dd7cc8c82eaf79c0c939320b"
    raw_path = args.proof / "raw-proof.npz"
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == report["raw_sha256"]
    variants = ["packed128-frozen-native", "packed64-frozen-native",
                "changed-packed128-parameterAD", "changed-packed64-parameterAD"]
    if len(report["checks"]) == 36:
        variants += ["dense128-frozen-native", "parameterAD128-frozen-native"]
    else:
        assert len(report["checks"]) == 42
        variants += ["packedParameterAD128-frozen-native", "compiledParameterAD128-frozen-native",
                     "originalParameterAD128-frozen-native"]
    expected = {f"seed{seed}-{variant}_{kind}" for seed in report["seeds"] for variant in variants
                for kind in ("loss", "gradient", "numerators")}
    assert {row["name"] for row in report["checks"]} == expected
    assert len(report["checks"]) == len(expected)
    required = {prefix + "_" + side + "_" + kind for name in expected
                for prefix, kind in [name.rsplit("_", 1)] for side in ("actual", "reference")}
    errors = []
    with np.load(raw_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == required
        for name in expected:
            prefix, kind = name.rsplit("_", 1)
            a, b = arrays[prefix + "_actual_" + kind], arrays[prefix + "_reference_" + kind]
            assert a.shape == b.shape and a.dtype == b.dtype == np.float64
            assert a.size == {"loss": 1, "gradient": 50692, "numerators": 4}[kind]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            error = float(np.max(np.abs(a - b) / (1e-11 + 1e-9 * np.abs(b))))
            assert error <= 1
            errors.append(error)
    result = {"state": "PASS", "reader": "NumPy only; saved canonical CPU arrays",
              "comparisons": len(errors), "arrays": len(required),
              "maximum_error_over_original_budget": max(errors)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
