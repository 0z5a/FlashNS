"""First-affine/activation dispatch, full parameter VJP and real Adam updates."""

import pytest
import torch

from flashns.jet_packed import TensorActivation
from flashns.pinn_packed import PackedPINN
from test_coordinate_affine import coordinate_reference


@pytest.mark.parametrize("mode", ["dense", "split", "compact"])
@pytest.mark.parametrize("ni,nb", [(7, 3), (0, 3), (7, 0), (0, 0)])
def test_first_fused_full_gradients_and_three_adam_updates(mode, ni, nb):
    generator = torch.Generator().manual_seed(92641)

    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64)*0.2

    x = random(ni+nb, 2)
    pde = torch.arange(1, ni+1, dtype=torch.float64)/max(ni, 1)
    boundary, target = random(nb, 3).square(), random(nb, 3)
    originals = [random(7, 2), random(5, 7), random(3, 5), random(7), random(5), random(3)]
    reference_params = [p.clone().requires_grad_() for p in originals]
    candidate_params = [p.clone().requires_grad_() for p in originals]
    reference_optimizer = torch.optim.Adam(reference_params, lr=1e-3)
    candidate_optimizer = torch.optim.Adam(candidate_params, lr=1e-3)
    calls = []

    def fused(layout, coordinates, weight, bias):
        assert weight is candidate_params[0] and bias is candidate_params[3]
        calls.append((layout.full_points, layout.value_points))
        return TensorActivation(layout).forward(coordinate_reference(layout, coordinates, weight, bias))

    def unused_affine(*args):
        raise AssertionError("fused first hidden layer must bypass its materialized affine")

    reference = PackedPINN(x, ni, pde, boundary, target, layout=mode,
                           coordinate_affine=coordinate_reference)
    candidate = PackedPINN(x, ni, pde, boundary, target, layout=mode,
                           coordinate_affine=unused_affine, first_affine_activation=fused)
    x.add_(100)  # Both schedules own the fixed coordinate input prepared earlier.
    expected_calls = {"dense": [(ni+nb, 0)], "split": [(ni, 0), (0, nb)],
                      "compact": [(ni, nb)]}[mode]
    for _ in range(3):
        calls.clear()
        expected_loss, expected_gradients = reference(reference_params[:3], reference_params[3:])
        actual_loss, actual_gradients = candidate(candidate_params[:3], candidate_params[3:])
        torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert calls == expected_calls
        for parameter, gradient in zip(reference_params, expected_gradients):
            parameter.grad = gradient
        for parameter, gradient in zip(candidate_params, actual_gradients):
            parameter.grad = gradient
        reference_optimizer.step()
        candidate_optimizer.step()
        for actual, expected in zip(candidate_params, reference_params):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["dense", "split", "compact"])
def test_single_linear_keeps_coordinate_affine_without_activation(mode):
    x = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float64)
    calls = []

    def affine(layout, *args):
        calls.append((layout.full_points, layout.value_points))
        return coordinate_reference(layout, *args)

    def unexpected(*args):
        raise AssertionError("linear output must never acquire a hidden activation")

    args = (x, 1, torch.ones(1, dtype=x.dtype), torch.ones(1, 3, dtype=x.dtype),
            torch.zeros(1, 3, dtype=x.dtype))
    baseline = PackedPINN(*args, layout=mode, coordinate_affine=coordinate_reference)
    candidate = PackedPINN(*args, layout=mode, coordinate_affine=affine,
                           first_affine_activation=unexpected)
    weights, biases = [torch.ones(3, 2, dtype=x.dtype)], [torch.zeros(3, dtype=x.dtype)]
    expected_loss, expected_gradients = baseline(weights, biases)
    actual_loss, actual_gradients = candidate(weights, biases)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(calls) == (2 if mode == "split" else 1)


@pytest.mark.parametrize("mode", ["dense", "split", "compact"])
def test_first_fused_matches_independent_nested_ad(mode):
    from test_pinn_v8 import reference

    x, ws, bs, ni, pw, bw, target = reference.make_case(5, 3, width=7, seed=642)
    expected_loss, expected_gradients = reference.nested_step(x, ws, bs, ni, pw, bw, target)

    def fused(layout, *args):
        return TensorActivation(layout).forward(coordinate_reference(layout, *args))

    # The new hook also works without selecting the standalone coordinate hook.
    candidate = PackedPINN(torch.tensor(x), ni, torch.tensor(pw), torch.tensor(bw),
                           torch.tensor(target), layout=mode, first_affine_activation=fused)
    actual_loss, actual_gradients = candidate([torch.tensor(w) for w in ws], [torch.tensor(b) for b in bs])
    reference.close(actual_loss.detach().numpy(), expected_loss)
    for actual, expected in zip(actual_gradients, expected_gradients):
        reference.close(actual.detach().numpy(), expected)
