"""Check native states/parameter gradients and immediate cache lifetime."""
import gc
import unittest
import weakref
from unittest.mock import patch

import torch
from box376 import c36_coordinate_oracle as oracle


def recursive_table(function, points):
    values = function(points)
    if values.ndim == 1:
        values = values[:, None]
    cache = {(): values}

    def derivative(indices):
        indices = tuple(sorted(indices))
        if indices in cache:
            return cache[indices]
        parent = derivative(indices[:-1])
        columns = []
        for component in range(parent.shape[1]):
            value = parent[:, component]
            gradient = torch.autograd.grad(value.sum(), points, create_graph=True,
                                           retain_graph=True, allow_unused=True)[0] if value.requires_grad else None
            columns.append(points[:, indices[-1]] * 0 if gradient is None else gradient[:, indices[-1]])
        cache[indices] = torch.stack(columns, 1)
        return cache[indices]

    return values, derivative


class CoordinateOracleTests(unittest.TestCase):
    def test_states_and_complete_parameter_gradients(self):
        rng = torch.Generator().manual_seed(795914)
        coordinates = torch.randn((5, 8), generator=rng, dtype=torch.float64)[:, ::2]
        initial = torch.randn((6, 4), generator=rng, dtype=torch.float64)

        def evaluate(builder, parameters, points):
            def function(x):
                features = torch.stack((x[:, 0]**4, x[:, 1]**3*x[:, 2], x.prod(1),
                                        x[:, 2]**2*x[:, 3]**2, x[:, 0]*x[:, 3], torch.ones_like(x[:, 0])), 1)
                return features @ parameters
            return builder(function, points)

        for builder in (oracle.native_c17, oracle.native_c36):
            parameters = initial.clone().requires_grad_(True)
            actual = evaluate(builder, parameters, coordinates.detach().requires_grad_(True))
            cotangent = torch.randn(actual.shape, generator=rng, dtype=torch.float64)
            actual_gradient = torch.autograd.grad(actual, parameters, cotangent)[0]
            previous = initial.clone().requires_grad_(True)
            with patch.object(oracle, 'derivative_table', recursive_table):
                expected = evaluate(builder, previous, coordinates.detach().requires_grad_(True))
                expected_gradient = torch.autograd.grad(expected, previous, cotangent)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    def test_derivative_function_released_without_cyclic_collection(self):
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            points = torch.ones((2, 4), dtype=torch.float64, requires_grad=True)
            _, derivative = oracle.derivative_table(lambda x: x[:, 0]**4, points)
            derivative((0, 0, 0, 0))
            reference = weakref.ref(derivative)
            del derivative
            self.assertIsNone(reference())
        finally:
            if was_enabled:
                gc.enable()


if __name__ == '__main__':
    unittest.main()
