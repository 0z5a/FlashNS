"""Complete C++ stable VJP fused into coordinate contraction and PINN updates."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from flashns.jet_packed import PackedLayout, TensorActivation
from flashns.pinn_packed import PackedPINN
from test_coordinate_wgrad import dense_contraction, assert_patterns


@pytest.fixture(scope="session")
def host_activation_wgrad(tmp_path_factory):
    path = Path(__file__).resolve().parents[1]/"experiments/pinn_v8/coordinate_activation_wgrad_host.py"
    spec = importlib.util.spec_from_file_location("coordinate_activation_wgrad_host", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = (Path(os.environ["FLASHNS_HOST_FIRST_VJP_OUTPUT"]) if "FLASHNS_HOST_FIRST_VJP_OUTPUT" in os.environ
              else tmp_path_factory.mktemp("coordinate_activation_wgrad")/"build")
    return module.build_host(output)


def exact_matched(actual, expected):
    assert_patterns(actual, expected)
    for observed, target in zip(actual, expected):
        finite = torch.isfinite(target)
        assert torch.equal(observed.view(torch.int64)[finite], target.view(torch.int64)[finite])


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("full,value", [(0, 0), (0, 3), (17, 5), (129, 31)])
@pytest.mark.parametrize("channels", [7, 64])
def test_g2_actual_helper_against_materialized_g1_and_independent_tensor_vjp(host_activation_wgrad, dimension, full, value, channels):
    layout = PackedLayout(full, value, dimension)
    generator = torch.Generator().manual_seed(736119)
    x = torch.randn(layout.points, dimension, generator=generator, dtype=torch.float64)*0.3
    z = torch.randn(layout.rows, channels, generator=generator, dtype=torch.float64)*0.3
    seed = torch.randn(layout.rows, channels, generator=generator, dtype=torch.float64)*0.3
    ops = TensorActivation(layout)
    h, aux = ops.forward(z)
    expected = dense_contraction(layout, x, ops.vjp(h, aux, seed))
    g1 = host_activation_wgrad(layout, x, h, aux, seed, fused=False)
    g2 = host_activation_wgrad(layout, x, h, aux, seed)
    exact_matched(g2, g1)
    for observed, target in zip(g2, expected):
        torch.testing.assert_close(observed, target, atol=2e-12, rtol=2e-11)


@pytest.mark.parametrize("dimension", [2, 3])
def test_g2_preserves_complete_nonfinite_vjp_and_internal_overflow(host_activation_wgrad, dimension):
    layout = PackedLayout(1, 1, dimension)
    x = torch.ones(layout.points, dimension, dtype=torch.float64)
    base = torch.full((layout.rows, 2), 0.1, dtype=x.dtype)
    for location in ("hidden", "seed"):
        for slot in range(layout.rows):
            for value in (float("nan"), float("inf"), -float("inf")):
                h, seed, aux = base.clone(), base.clone(), torch.ones(layout.points, 2, dtype=x.dtype)
                (h if location == "hidden" else seed)[slot, 1] = value
                exact_matched(host_activation_wgrad(layout, x, h, aux, seed),
                              host_activation_wgrad(layout, x, h, aux, seed, fused=False))
    for value in (float("nan"), float("inf"), -float("inf")):
        aux = torch.ones(layout.points, 2, dtype=x.dtype)
        aux[0, 1] = value
        exact_matched(host_activation_wgrad(layout, x, base, aux, base),
                      host_activation_wgrad(layout, x, base, aux, base, fused=False))
    # Inputs can all be finite while the full VJP overflows. G2 must feed all
    # resulting bar_Z slots into the same zero-product propagation as G1.
    h, seed = torch.ones_like(base), torch.full_like(base, 1e308)
    aux = torch.ones(layout.points, 2, dtype=x.dtype)
    actual = host_activation_wgrad(layout, x, h, aux, seed)
    expected = host_activation_wgrad(layout, x, h, aux, seed, fused=False)
    exact_matched(actual, expected)
    assert any(bool((~torch.isfinite(t)).any()) for t in expected)


@pytest.mark.parametrize("dimension", [2, 3])
def test_g2_keeps_saturated_nonzero_a1_tail(host_activation_wgrad, dimension):
    layout = PackedLayout(1, 1, dimension)
    x = torch.ones(layout.points, dimension, dtype=torch.float64)
    for center in (20., -20., 100., -100., 350., -350.):
        z = torch.full((layout.rows, 1), 0.1, dtype=x.dtype)
        z[layout.zero_rows(x.device)] = center
        h, aux = TensorActivation(layout).forward(z)
        seed = torch.zeros_like(h)
        seed[layout.zero_rows(x.device)] = 1.
        actual = host_activation_wgrad(layout, x, h, aux, seed)
        expected = host_activation_wgrad(layout, x, h, aux, seed, fused=False)
        exact_matched(actual, expected)
        assert bool((aux>0).all()) and bool((actual[1]>0).all())


@pytest.mark.parametrize("mode", ["dense", "split", "compact"])
@pytest.mark.parametrize("ni,nb", [(7, 3), (0, 3), (7, 0), (0, 0)])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_g2_full_gradients_three_adam_updates_dispatch_and_snapshot(host_activation_wgrad, mode, ni, nb, depth):
    generator = torch.Generator().manual_seed(625913)

    def random(*shape): return torch.randn(shape, generator=generator, dtype=torch.float64)*0.2

    x = random(ni+nb, 2)
    args = (x, ni, torch.arange(1, ni+1, dtype=x.dtype)/max(ni, 1), random(nb, 3).square(), random(nb, 3))
    sizes = {1: [2, 3], 2: [2, 7, 3], 3: [2, 7, 5, 3]}[depth]
    originals = [random(b, a) for a, b in zip(sizes, sizes[1:])] + [random(b) for b in sizes[1:]]
    baseline_params = [t.clone().requires_grad_() for t in originals]
    candidate_params = [t.clone().requires_grad_() for t in originals]
    optimizers = [torch.optim.Adam(params, lr=1e-3) for params in (baseline_params, candidate_params)]
    baseline = PackedPINN(*args, layout=mode)
    calls = []

    def fused(layout, coordinates, h, aux, seed):
        calls.append((layout.full_points, layout.value_points))
        return host_activation_wgrad(layout, coordinates, h, aux, seed)

    # Only the G2 hook is selected, so its own coordinate snapshot and ordinary
    # single-affine fallback cannot rely on any forward or G1 hook.
    candidate = PackedPINN(*args, layout=mode, first_activation_wgrad=fused)
    x.add_(100)
    expected_calls = ({"dense": [(ni+nb, 0)], "split": [(ni, 0), (0, nb)],
                       "compact": [(ni, nb)]}[mode] if depth>1 else [])
    for _ in range(3):
        calls.clear()
        expected_loss, expected_gradients = baseline(baseline_params[:depth], baseline_params[depth:])
        actual_loss, actual_gradients = candidate(candidate_params[:depth], candidate_params[depth:])
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
def test_g2_full_parameter_vjp_against_independent_nested_ad(host_activation_wgrad, mode):
    from test_pinn_v8 import reference
    x, ws, bs, ni, pw, bw, target = reference.make_case(5, 3, width=7, seed=1883)
    expected_loss, expected_gradients = reference.nested_step(x, ws, bs, ni, pw, bw, target)
    candidate = PackedPINN(torch.tensor(x), ni, torch.tensor(pw), torch.tensor(bw), torch.tensor(target),
                           layout=mode, first_activation_wgrad=host_activation_wgrad)
    actual_loss, actual_gradients = candidate([torch.tensor(w) for w in ws], [torch.tensor(b) for b in bs])
    reference.close(actual_loss.detach().numpy(), expected_loss)
    for actual, expected in zip(actual_gradients, expected_gradients):
        reference.close(actual.detach().numpy(), expected)
