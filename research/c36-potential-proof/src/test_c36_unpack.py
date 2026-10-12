"""Packed gathers preserve the old symmetric expansion and its full transpose."""
import itertools
import unittest
from unittest.mock import patch

import torch
import torch_c36 as c36


def slice_unpack(state):
    h = state.new_empty(state.shape[:-1] + (3, 3))
    q, k = torch.empty_like(h), torch.empty_like(h)
    t = state.new_empty(state.shape[:-1] + (3, 3, 3))
    for slot, (i, j) in enumerate(c36.PAIRS):
        h[..., i, j] = h[..., j, i] = state[..., 4 + slot]
        q[..., i, j] = q[..., j, i] = state[..., 20 + slot]
        k[..., i, j] = k[..., j, i] = state[..., 30 + slot]
    for slot, triple in enumerate(c36.TRIPLES):
        for indices in set(itertools.permutations(triple)):
            t[..., indices[0], indices[1], indices[2]] = state[..., 10 + slot]
    return state[..., 0], state[..., 1:4], h, t, q, state[..., 26], state[..., 27:30], k


class PackedGatherTests(unittest.TestCase):
    def test_values_and_all_input_gradients(self):
        generator = torch.Generator().manual_seed(795913)
        values = torch.randn((2, 3, 36), generator=generator, dtype=torch.float64)
        for a, b in zip(c36.unpack(values), slice_unpack(values)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        cotangent = torch.randn(values.shape, generator=generator, dtype=torch.float64)
        left = torch.randn((2, 1, 36), generator=generator, dtype=torch.float64)
        for function in (c36.activate, lambda x: c36.multiply(left, x)):
            current = values.clone().requires_grad_(True)
            actual = function(current)
            actual_gradient = torch.autograd.grad(actual, current, cotangent)[0]
            previous = values.clone().requires_grad_(True)
            with patch.object(c36, "unpack", slice_unpack):
                expected = function(previous)
                expected_gradient = torch.autograd.grad(expected, previous, cotangent)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-9, atol=1e-11)

    def test_noncontiguous_empty_and_zero_preactivation(self):
        state = torch.zeros((2, 4, 72), dtype=torch.float64)[..., ::2].requires_grad_(True)
        assert not state.is_contiguous()
        result = c36.activate(state)
        gradient = torch.autograd.grad(result.sum(), state)[0]
        assert torch.isfinite(gradient).all()
        for a, b in zip(c36.unpack(state), slice_unpack(state)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(c36.unpack(state[:0]), slice_unpack(state[:0])):
            assert a.shape == b.shape


if __name__ == "__main__":
    unittest.main()
