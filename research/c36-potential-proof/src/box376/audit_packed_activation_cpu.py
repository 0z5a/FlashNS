"""NumPy-only rereading of the opt-in packed activation's CPU proof."""
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
    raw_path = args.proof / "raw-proof.npz"
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == report["raw_sha256"]
    errors = []
    with np.load(raw_path, allow_pickle=False) as arrays:
        for check in report["checks"]:
            key = f"{check['seed']}_{check['case']}_chunk{check['chunk']}_{check['quantity']}"
            for reference in ("dense", "parameter_ad"):
                a, b = arrays[key], arrays[key + "_" + reference]
                assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
                assert np.isfinite(a).all() and np.isfinite(b).all()
                error = float(np.max(np.abs(a - b) / (1e-11 + 1e-9 * np.abs(b))))
                assert error <= 1
                errors.append(error)
    result = {"state": "PASS", "reader": "NumPy only; saved CPU arrays",
              "comparisons": len(errors), "maximum_error_over_original_budget": max(errors)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
