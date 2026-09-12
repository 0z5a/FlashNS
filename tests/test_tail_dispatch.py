"""CPU contract checks for the optional packed tail dgrad/VJP backend."""

import pytest
import torch

from flashns.jet_packed import TensorActivation
from flashns.pinn_packed import PackedPINN


@pytest.mark.parametrize("mode", ["dense", "split", "compact"])
@pytest.mark.parametrize("ni,nb", [(5, 3), (0, 3), (5, 0)])
def test_tail_backend_preserves_weighted_full_gradient(mode, ni, nb):
    generator = torch.Generator().manual_seed(9212)

    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64) * 0.2

    x = random(ni + nb, 2)
    pde = torch.linspace(0.1, 0.7, ni, dtype=torch.float64)
    boundary = random(nb, 3).square()
    target = random(nb, 3)
    weights = [random(7, 2), random(5, 7), random(3, 5)]
    biases = [random(7), random(5), random(3)]
    calls = []

    def tail(layout, d, weight, hidden, aux):
        # The callback sees the complete packed segment, including scalar rows.
        assert d.shape == (layout.rows, 3)
        assert hidden.shape == (layout.rows, 5)
        assert aux.shape == (layout.points, 5)
        assert weight is weights[-1]
        calls.append((layout.full_points, layout.value_points))
        return TensorActivation(layout).vjp(hidden, aux, d @ weight)

    reference = PackedPINN(x, ni, pde, boundary, target, layout=mode)
    candidate = PackedPINN(x, ni, pde, boundary, target, layout=mode,
                           tail_dgrad_vjp=tail)
    expected_segments = ({"dense": [(ni + nb, 0)],
                          "split": [(ni, 0), (0, nb)],
                          "compact": [(ni, nb)]})[mode]
    for _ in range(3):
        calls.clear()
        loss_a, grads_a = reference(weights, biases)
        loss_p, grads_p = candidate(weights, biases)
        torch.testing.assert_close(loss_p, loss_a, rtol=0, atol=0)
        for actual, expected in zip(grads_p, grads_a):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert calls == expected_segments
        # A backend must consume updated parameters, not a cached first weight.
        with torch.no_grad():
            for parameter, gradient in zip(weights + biases, grads_a):
                parameter.add_(gradient, alpha=-1e-3)


def test_linear_network_never_calls_activation_vjp():
    x = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float64)

    def unexpected(*args):
        raise AssertionError("a linear output has no preceding activation")

    step = PackedPINN(x, 1, torch.ones(1, dtype=torch.float64),
                      torch.ones(1, 3, dtype=torch.float64),
                      torch.zeros(1, 3, dtype=torch.float64),
                      tail_dgrad_vjp=unexpected)
    loss, gradients = step([torch.ones(3, 2, dtype=torch.float64)],
                           [torch.zeros(3, dtype=torch.float64)])
    assert torch.isfinite(loss)
    assert [g.shape for g in gradients] == [(3, 2), (3,)]
