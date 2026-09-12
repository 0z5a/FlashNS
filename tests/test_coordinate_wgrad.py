"""Actual C++ helper vs dense BLAS, full parameter gradients and Adam updates."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from flashns.jet_packed import PackedLayout
from flashns.pinn_packed import PackedPINN

HERE = Path(__file__).resolve().parents[1]/"experiments/pinn_v8"


@pytest.fixture(scope="session")
def host_wgrad(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("coordinate_wgrad_host", HERE/"coordinate_wgrad_host.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = (Path(os.environ["FLASHNS_HOST_WGRAD_OUTPUT"]) if "FLASHNS_HOST_WGRAD_OUTPUT" in os.environ
              else tmp_path_factory.mktemp("coordinate_wgrad")/"build")
    return module.build_host(output)


def dense_contraction(layout, x, derivative):
    return derivative.T @ layout.coordinates(x), derivative[layout.zero_rows(x.device)].sum(0)


def assert_patterns(actual, expected):
    for observed, target in zip(actual, expected):
        for predicate in (torch.isnan, torch.isposinf, torch.isneginf):
            assert torch.equal(predicate(observed), predicate(target))
        finite = torch.isfinite(target)
        torch.testing.assert_close(observed[finite], target[finite], atol=2e-12, rtol=2e-11)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("full,value", [(0, 0), (0, 3), (1, 0), (17, 5), (129, 31)])
@pytest.mark.parametrize("channels", [7, 64])
def test_actual_host_helper_against_dense_blas(host_wgrad, dimension, full, value, channels):
    layout = PackedLayout(full, value, dimension)
    generator = torch.Generator().manual_seed(89917)
    x = torch.randn(layout.points, dimension, generator=generator, dtype=torch.float64)*0.3
    derivative = torch.randn(layout.rows, channels, generator=generator, dtype=torch.float64)*0.3
    # Nonzero highest-order derivatives are intentional; this is not a zero-tail test.
    expected = dense_contraction(layout, x, derivative)
    actual = host_wgrad(layout, x, derivative)
    for observed, target in zip(actual, expected):
        torch.testing.assert_close(observed, target, atol=2e-12, rtol=2e-11)


@pytest.mark.parametrize("dimension", [2, 3])
def test_each_nonfinite_slot_has_axis_specific_propagation(host_wgrad, dimension):
    layout = PackedLayout(1, 1, dimension)
    x = torch.ones(layout.points, dimension, dtype=torch.float64)
    for slot in range(layout.rows):
        for value in (float("nan"), float("inf"), -float("inf")):
            derivative = torch.zeros(layout.rows, 3, dtype=torch.float64)
            derivative[slot, 1] = value
            assert_patterns(host_wgrad(layout, x, derivative), dense_contraction(layout, x, derivative))
    # Higher-order nonfinite values contaminate dW through structural zeros but
    # do not enter db; a selected unit derivative must keep its matching Inf.
    derivative = torch.zeros(layout.rows, 1, dtype=torch.float64)
    derivative[dimension, 0] = float("inf")
    dw, db = host_wgrad(layout, x, derivative)
    assert torch.isposinf(dw[0, 0]) and bool(torch.isnan(dw[0, 1:]).all())
    assert db.item() == 0.


@pytest.mark.parametrize("dimension", [2, 3])
def test_host_partials_and_outputs_cover_every_element_and_respect_canaries(host_wgrad, dimension):
    layout = PackedLayout(17, 5, dimension)
    channels = 7

    def guarded(shape):
        size = 1
        for item in shape: size *= item
        holder = torch.full((size+16,), 8123.5, dtype=torch.float64)
        view = holder[8:-8].view(shape)
        view.fill_(float("nan"))
        return view, holder

    x = torch.full((layout.points, dimension), 0.25, dtype=torch.float64)
    derivative = torch.full((layout.rows, channels), 0.5, dtype=torch.float64)
    partial, ph = guarded(((layout.points+7)//8, channels, dimension+1))
    dw, wh = guarded((channels, dimension))
    db, bh = guarded((channels,))
    host_wgrad.call_into(layout, x, derivative, partial, dw, db)
    assert bool(torch.isfinite(partial).all())
    for tensor in (ph, wh, bh):
        assert bool((tensor[:8] == 8123.5).all()) and bool((tensor[-8:] == 8123.5).all())
    assert_patterns((dw, db), dense_contraction(layout, x, derivative))


@pytest.mark.parametrize("mode", ["dense", "split", "compact"])
@pytest.mark.parametrize("ni,nb", [(7, 3), (0, 3), (7, 0), (0, 0)])
@pytest.mark.parametrize("single_linear", [False, True])
def test_full_gradients_three_real_adam_updates_and_coordinate_snapshot(host_wgrad, mode, ni, nb, single_linear):
    generator = torch.Generator().manual_seed(977319)

    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64)*0.2

    x = random(ni+nb, 2)
    args = (x, ni, torch.arange(1, ni+1, dtype=x.dtype)/max(ni, 1), random(nb, 3).square(), random(nb, 3))
    originals = ([random(3, 2), random(3)] if single_linear else
                 [random(7, 2), random(5, 7), random(3, 5), random(7), random(5), random(3)])
    count = len(originals)//2
    baseline_params = [t.clone().requires_grad_() for t in originals]
    candidate_params = [t.clone().requires_grad_() for t in originals]
    optimizers = [torch.optim.Adam(params, lr=1e-3) for params in (baseline_params, candidate_params)]
    baseline = PackedPINN(*args, layout=mode)
    calls = []

    def terminal(layout, coordinates, derivative):
        calls.append((layout.full_points, layout.value_points))
        return host_wgrad(layout, coordinates, derivative)

    candidate = PackedPINN(*args, layout=mode, coordinate_wgrad=terminal)
    assert candidate.metadata["coordinate_wgrad_enabled"] and not baseline.metadata["coordinate_wgrad_enabled"]
    x.add_(100)  # The terminal hook must own its own fixed coordinate snapshot.
    expected_calls = {"dense": [(ni+nb, 0)], "split": [(ni, 0), (0, nb)], "compact": [(ni, nb)]}[mode]
    for _ in range(3):
        calls.clear()
        expected_loss, expected_gradients = baseline(baseline_params[:count], baseline_params[count:])
        actual_loss, actual_gradients = candidate(candidate_params[:count], candidate_params[count:])
        torch.testing.assert_close(actual_loss, expected_loss, atol=2e-12, rtol=2e-11)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
        assert calls == expected_calls
        for optimizer, params, grads in zip(optimizers, (baseline_params, candidate_params),
                                            (expected_gradients, actual_gradients)):
            for parameter, gradient in zip(params, grads): parameter.grad = gradient
            optimizer.step()
        for actual, expected in zip(candidate_params, baseline_params):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)


@pytest.mark.parametrize("mode", ["dense", "split", "compact"])
def test_actual_host_helper_full_vjp_against_independent_nested_ad(host_wgrad, mode):
    from test_pinn_v8 import reference
    x, ws, bs, ni, pw, bw, target = reference.make_case(5, 3, width=7, seed=9121)
    expected_loss, expected_gradients = reference.nested_step(x, ws, bs, ni, pw, bw, target)
    candidate = PackedPINN(torch.tensor(x), ni, torch.tensor(pw), torch.tensor(bw), torch.tensor(target),
                           layout=mode, coordinate_wgrad=host_wgrad)
    actual_loss, actual_gradients = candidate([torch.tensor(w) for w in ws], [torch.tensor(b) for b in bs])
    reference.close(actual_loss.detach().numpy(), expected_loss)
    for actual, expected in zip(actual_gradients, expected_gradients):
        reference.close(actual.detach().numpy(), expected)
