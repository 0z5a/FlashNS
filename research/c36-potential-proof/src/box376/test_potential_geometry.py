"""Regression checks for gate derivatives at near-zero spatial cutoffs."""
import unittest

import torch

from box376.c36_coordinate_oracle import native_c36
from box376.potential_geometry import features_and_gate
from box376.potential_model import prepare_geometry


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


class PreparedGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.points = torch.tensor([
            [.31, .47, .53, .1], [.31, .47, .53, .25],
            [.42, .51, .36, .4], [.51, .43, .37, .5],
            [.63, .43, .48, .6], [.31, .47, .53, .75],
            [.31, .47, .53, .9],
        ], dtype=torch.float64, requires_grad=True)
        cls.reference = tuple(native_c36(lambda p, i=i: features_and_gate(p)[i], cls.points).detach()
                              for i in (0, 1))

    def test_chunked_states_match_coordinate_ad_and_preserve_input(self):
        before = self.points.detach().clone()
        # A noncontiguous coordinate view also has to retain its point order.
        storage = torch.stack((self.points.detach(), self.points.detach()), -1)
        points = storage[..., 0].requires_grad_(True)
        for chunk in (3, 7):
            with self.subTest(chunk=chunk), torch.no_grad():
                actual = prepare_geometry(points, chunk_size=chunk)
                for value, reference in zip(actual, self.reference):
                    torch.testing.assert_close(value, reference, atol=1e-11, rtol=1e-9)
                    self.assertFalse(value.requires_grad)
                self.assertTrue(torch.equal(points, before))
                self.assertIsNone(points.grad)

    def test_empty_geometry_preserves_dtype_and_shapes(self):
        inputs, gate = prepare_geometry(torch.empty((0, 4), dtype=torch.float64))
        self.assertEqual(inputs.shape, (0, 4, 36))
        self.assertEqual(gate.shape, (0, 1, 36))
        self.assertEqual(inputs.dtype, torch.float64)
        self.assertEqual(gate.dtype, torch.float64)

    def test_invalid_preparation_options_are_rejected(self):
        for chunk in (0, -1, True, 1.5):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                prepare_geometry(self.points, chunk_size=chunk)
        for eta in (0, -1, float("nan"), float("inf")):
            with self.subTest(eta=eta), self.assertRaises(ValueError):
                prepare_geometry(self.points, eta=eta)
        for points in (torch.zeros(4), torch.zeros(2, 3), torch.zeros(2, 4, dtype=torch.int64)):
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                prepare_geometry(points)


if __name__ == "__main__":
    unittest.main()
