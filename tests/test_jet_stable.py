import math

import mpmath as mp
import pytest

torch = pytest.importorskip("torch")

from flashns.jet_spec import JetSpec
from flashns.jet_stable import stable_tanh, tanh_jet, tanh_vjp


@pytest.mark.parametrize("dim", [2, 3])
def test_stable_jet_vjp_against_differentiable_reference(dim):
    torch.manual_seed(7410)
    spec = JetSpec(dim)
    z = (torch.randn(3, spec.q, 5, dtype=torch.float64) * 0.2).requires_grad_()
    z.data[1, 0] += 20
    z.data[2, 0] -= 20
    h, aux = tanh_jet(z, dim)
    seed = torch.randn_like(h)
    (expected,) = torch.autograd.grad((h * seed).sum(), z)
    actual = tanh_vjp(h.detach(), aux.detach(), seed, dim)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
    assert (actual[1:, 1:].abs().sum() > 0).item()


@pytest.mark.parametrize("center", [0.0, 5.0, 20.0, -20.0, 100.0, 350.0])
def test_nested_stable_tanh_derivatives_vs_100_digit_reference(center):
    x = torch.tensor(center, dtype=torch.float64, requires_grad=True)
    derivative = stable_tanh(x)
    with mp.workdps(100):
        for order in range(1, 5):
            (derivative,) = torch.autograd.grad(derivative, x, create_graph=True)
            # mp.diff avoids cancellation from finite working precision in tanh.
            expected = float(mp.diff(mp.tanh, mp.mpf(center), order, addprec=1200))
            actual = float(derivative.detach())
            assert abs(actual - expected) <= 1e-11 * abs(expected) + 8 * math.ulp(
                expected
            )
