"""Keep a frozen force binding across cross-platform analytic recomputation."""
import unittest
from unittest.mock import patch

import numpy as np

from box376.audit_potential_run import close, physical_metrics, validated_saved_source


class FrozenPhysicalSourceTests(unittest.TestCase):
    def setUp(self):
        self.exact = {"U": np.zeros((2, 3)), "f": np.full((2, 3), 1e8),
                      "df": np.full((2, 3, 3), 1e8)}
        self.raw = {"holdout_points": np.zeros((2, 4))}
        self.raw.update({"holdout_source_" + k: v for k, v in self.exact.items()})

    def test_last_bit_source_rounding_does_not_change_saved_residual(self):
        rounded = {k: np.nextafter(v, np.inf) if k != "U" else v for k, v in self.exact.items()}
        state = np.zeros((2, 4, 17))
        state[:, 3, 1:10] = 1e8
        scales = {k: 1. for k in ("velocity", "continuity_residual", "continuity_residual_gradient",
                                  "momentum_residual", "momentum_residual_gradient", "pressure")}
        args = (np.zeros(2), np.ones(2), scales)
        expected = physical_metrics(state, self.exact, *args)[1]
        freshly_rounded = physical_metrics(state, rounded, *args)[1]
        with self.assertRaises(AssertionError):
            close(freshly_rounded[0], expected[0])
        with patch("box376.audit_potential_run.source", return_value=rounded):
            fixed = validated_saved_source(self.raw, "holdout")
        actual = physical_metrics(state, fixed, *args)[1]
        for a, b in zip(actual, expected):
            close(a, b)

    def test_material_source_change_still_fails_original_tolerance(self):
        wrong = {k: v.copy() for k, v in self.exact.items()}
        wrong["df"][0, 0, 0] += 1.
        with patch("box376.audit_potential_run.source", return_value=wrong):
            with self.assertRaises(AssertionError):
                validated_saved_source(self.raw, "holdout")


if __name__ == "__main__":
    unittest.main()
