"""Regression checks for gate derivatives at near-zero spatial cutoffs."""
import unittest

import torch

from box376.c36_coordinate_oracle import native_c36
from box376.potential_geometry import features_and_gate


class PotentialGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        # Fixed source indices 1965 and 2489 exposed nonfinite high-order prod derivatives.
        points = torch.tensor([
            [0.4781737422195985, 0.25099227537987967, 0.369032400605715, 0.3505152014577218],
            [0.3726633665152011, 0.48347012883637397, 0.4999507584979306, 0.144075913319894],
        ], dtype=torch.float64, requires_grad=True)
        cls.states = native_c36(lambda p: features_and_gate(p)[1], points).detach()

    def test_cutoff_derivatives_are_finite(self):
        self.assertTrue(torch.isfinite(self.states).all().item())

    def test_temporally_flat_gate_has_zero_derivatives(self):
        # t < 1/4 is inside the prescribed motion's flat interval.
        self.assertTrue(torch.equal(self.states[1], torch.zeros_like(self.states[1])))


if __name__ == "__main__":
    unittest.main()
