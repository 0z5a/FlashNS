"""Physical-coordinate sparse affine contracts, including parameter updates."""

import pytest
import torch

from flashns.jet_packed import PackedLayout
from flashns.jet_spec import JetSpec
from flashns.pinn_packed import PackedPINN


def coordinate_reference(layout, x, weight, bias):
    result = x.new_zeros((layout.rows, len(weight)))
    result[layout.zero_rows(x.device)] = x @ weight.T + bias
    full = result[:layout.full_rows].view(layout.full_points, layout.q, len(weight))
    indices = JetSpec(layout.dimension).coefficient_order
    for axis in range(layout.dimension):
        unit = tuple(int(j == axis) for j in range(layout.dimension))
        full[:, indices.index(unit)] = weight[:, axis]
    return result


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("ni,nb", [(0, 0), (0, 3), (1, 0), (17, 5)])
def test_physical_coordinate_coefficient_order(dimension, ni, nb):
    generator = torch.Generator().manual_seed(6421)
    layout = PackedLayout(ni, nb, dimension)
    x = torch.randn(ni + nb, dimension, generator=generator, dtype=torch.float64)
    weight = torch.randn(7, dimension, generator=generator, dtype=torch.float64)
    bias = torch.randn(7, generator=generator, dtype=torch.float64)
    expected = layout.coordinates(x) @ weight.T
    expected[layout.zero_rows(x.device)] += bias
    torch.testing.assert_close(coordinate_reference(layout, x, weight, bias), expected,
                               rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("mode", ["dense", "split", "compact"])
@pytest.mark.parametrize("ni,nb", [(7, 3), (0, 3), (7, 0)])
def test_coordinate_hook_preserves_all_gradients_and_fixed_inputs(mode, ni, nb):
    generator = torch.Generator().manual_seed(6422)

    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64) * 0.2

    x = random(ni + nb, 2)
    weights = [random(7, 2), random(5, 7), random(3, 5)]
    biases = [random(7), random(5), random(3)]
    pde = torch.arange(1, ni + 1, dtype=torch.float64) / max(ni, 1)
    boundary, target = random(nb, 3).square(), random(nb, 3)
    calls = []

    def candidate(layout, coordinates, weight, bias):
        assert weight is weights[0] and bias is biases[0]
        calls.append((layout.full_points, layout.value_points))
        return coordinate_reference(layout, coordinates, weight, bias)

    baseline = PackedPINN(x, ni, pde, boundary, target, layout=mode)
    changed = PackedPINN(x, ni, pde, boundary, target, layout=mode,
                         coordinate_affine=candidate)
    # Both implementations prepare fixed data at construction time.
    x.add_(100)
    expected_calls = {"dense": [(ni + nb, 0)], "split": [(ni, 0), (0, nb)],
                      "compact": [(ni, nb)]}[mode]
    for _ in range(3):
        calls.clear()
        loss_a, gradients_a = baseline(weights, biases)
        loss_p, gradients_p = changed(weights, biases)
        torch.testing.assert_close(loss_p, loss_a, rtol=2e-11, atol=2e-12)
        for actual, expected in zip(gradients_p, gradients_a):
            torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
        assert calls == expected_calls
        with torch.no_grad():
            for parameter, gradient in zip(weights + biases, gradients_a):
                parameter.add_(gradient, alpha=-1e-3)
