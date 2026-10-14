"""NumPy-only verification of folded constants and original normalization."""
import hashlib
import json

import numpy as np


def audit_folding(proof, inputs, reference, report):
    assert inputs is not None and report["zero_gate_folding"]
    path = proof / "zero-gate-proof.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["folding_raw_sha256"]
    assert hashlib.sha256((inputs / "training-source.npz").read_bytes()).hexdigest() == "d378c28991dd9a852297f22c94f0727d4c0c1e5d4140fe718324859c09b78bc2"
    assert hashlib.sha256((inputs / "calibration.npz").read_bytes()).hexdigest() == "5aea51c47816fa006ab0124874ea12d1111fd3a436d70f9d05f29bc242316063"
    metadata = json.loads((inputs / "training-source.json").read_text())
    scales = json.loads((inputs / "calibration.json").read_text())["candidate_fixed_physical_scales"]
    families = ["interior", "initial", "boundary", "gauge"]
    counts, numerators = {}, []
    with np.load(path, allow_pickle=False) as raw, np.load(reference, allow_pickle=False) as original:
        required = {"constant_numerators", "denominators", "coefficients", "residual_inverse_square",
            "gradient_inverse_square", "velocity_inverse_square", "pressure_inverse_square"}
        required |= {name + suffix for name in families for suffix in ["_gate", "_active"]}
        assert set(raw.files) == required
        assert np.array_equal(raw["denominators"], [metadata["global_weight_sums"][n] for n in families])
        assert np.array_equal(raw["coefficients"], [1., 5., 5., 5.])
        residual_scale = np.array([1 / scales["momentum_residual"]**2] * 3 + [1 / scales["continuity_residual"]**2])
        derivative_scale = np.array([[1 / scales["momentum_residual_gradient"]**2] * 3] * 3 + [[1 / scales["continuity_residual_gradient"]**2] * 3])
        assert np.array_equal(raw["residual_inverse_square"], residual_scale)
        assert np.array_equal(raw["gradient_inverse_square"], derivative_scale)
        assert raw["velocity_inverse_square"] == 1 / scales["velocity"]**2
        assert raw["pressure_inverse_square"] == 1 / scales["pressure"]**2
        for name, full in zip(families, [4096, 512, 384, 7]):
            gate, active = raw[name + "_gate"], raw[name + "_active"]
            assert gate.dtype == np.float64 and gate.shape == (full, 1, 36) and np.isfinite(gate).all()
            assert active.dtype == np.bool_ and active.shape == (full,)
            assert np.array_equal(active, np.any(gate != 0, axis=(1, 2)))
            count = {"full": full, "active": int(active.sum()), "folded": int((~active).sum()),
                "original_W": metadata["global_weight_sums"][name]}
            assert count == report["folding_family_counts"][name]
            counts[name] = count
            prefix = "canonical_" + name + "_"
            weights = original[prefix + "weights"][~active]
            if name == "interior":
                force, derivative = original[prefix + "force"][~active], original[prefix + "force_gradient"][~active]
                terms = (force**2 * residual_scale[:3]).sum(1)
                terms += .2 * (derivative**2 * derivative_scale[:3]).sum((1, 2))
            else:
                components = slice(3, 4) if name == "gauge" else slice(0, 3)
                scale = 1 / scales["pressure" if name == "gauge" else "velocity"]**2
                terms = (original[prefix + "target"][~active, components]**2).sum(1) * scale
            numerators.append(float((weights * terms).sum()))
        actual = raw["constant_numerators"]
        assert actual.dtype == np.float64 and actual.shape == (4,) and np.isfinite(actual).all()
        assert np.allclose(actual, numerators, atol=1e-11, rtol=1e-9)
    return {"state": "PASS", "reader": "NumPy only", "saved_arrays": len(required),
        "counts": counts, "constant_numerators": numerators, "normalization_unchanged": True,
        "every_original_row_accounted_for": True, "zero_threshold": 0.}
