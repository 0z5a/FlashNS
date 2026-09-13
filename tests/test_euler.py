import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import BSpline, NdBSpline

from flashns.euler import (
    computational_jet,
    load_points,
    load_surfaces,
    local_coefficient_jet,
    physical_jet,
    points_hash,
    residuals_from_jets,
    sample_points,
)


def polynomial_surface():
    # Independent polynomial f=x^2+2*z^3+x*z/2 represented in cubic splines.
    knots = np.array([-2] * 4 + [2] * 4, dtype=float)
    sites = np.linspace(-2, 2, 4)
    basis = BSpline.design_matrix(sites, knots, 3).toarray()
    x, z = np.meshgrid(sites, sites, indexing="ij")
    values = x * x + 2 * z**3 + x * z / 2
    coefficients = np.linalg.solve(basis, np.linalg.solve(basis, values).T).T
    return NdBSpline((knots, knots), coefficients, (3, 3), extrapolate=False)


@pytest.mark.parametrize("evaluate", [computational_jet, local_coefficient_jet])
def test_independent_derivatives_and_nonlinear_coordinate_chain_rule(evaluate):
    points = np.array([[-2, -2], [0, 0], [0.3, -0.2], [2, 2]])
    x, z = points.T
    jet = evaluate(polynomial_surface(), points, second=True)
    expected = {
        (0, 0): x * x + 2 * z**3 + x * z / 2,
        (1, 0): 2 * x + z / 2,
        (0, 1): 6 * z * z + x / 2,
        (2, 0): np.full(len(x), 2),
        (0, 2): 12 * z,
    }
    for key, value in expected.items():
        np.testing.assert_allclose(jet[key], value, atol=4e-13)
    physical = physical_jet(jet, points)
    np.testing.assert_allclose(
        physical["RR"], (2 - (2 * x + z / 2) * np.tanh(x)) / np.cosh(x) ** 2, atol=4e-13
    )
    np.testing.assert_allclose(
        physical["ZZ"],
        (12 * z - (6 * z * z + x / 2) * np.tanh(z)) / np.cosh(z) ** 2,
        atol=4e-13,
    )


def test_column_major_loader_and_degree_contract(tmp_path):
    surface = polynomial_surface()
    data = {
        "degree_x": 3,
        "degree_y": 3,
        "nctrl_x": 4,
        "nctrl_y": 4,
        "coeffs_layout": "column_major",
        "knots_x": surface.t[0].tolist(),
        "knots_y": surface.t[1].tolist(),
        "coeffs": surface.c.flatten(order="F").tolist(),
    }
    for name in ["U", "Omega", "Psi"]:
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    (tmp_path / "meta.json").write_text(
        json.dumps({"degree": 3, "lambda": 0.5, "C_raw": 10})
    )
    _, loaded = load_surfaces(tmp_path)
    np.testing.assert_allclose(loaded["U"].c, surface.c)
    (tmp_path / "meta.json").write_text(
        json.dumps({"degree": 4, "lambda": 0.5, "C_raw": 10})
    )
    with pytest.raises(ValueError, match="degree"):
        load_surfaces(tmp_path)


def test_spline_bounds_and_nonfinite_inputs():
    for points in [[[3, 0]], [[float("nan"), 0]]]:
        with pytest.raises(ValueError):
            computational_jet(polynomial_surface(), points)


def test_mpmath_differentiation_matches_known_polynomial():
    from flashns.spline_reference import high_precision_jet

    point = [0.3, -0.2]
    jet = high_precision_jet(polynomial_surface(), point, second=True)
    assert abs(float(jet[1, 0]) - 0.5) < 1e-13
    assert abs(float(jet[0, 1]) - 0.39) < 1e-13
    assert abs(float(jet[2, 0]) - 2) < 1e-13
    assert abs(float(jet[0, 2]) + 2.4) < 1e-13


def test_axis_removable_singularity_convention():
    points = np.array([[0, 0], [1e-12, 0], [1e-7, 0]])
    # Psi=R^2 => Psi_RR+(3/R)Psi_R=2+6=8, including R=0.
    R = np.sinh(points[:, 0])
    zeros = np.zeros(3)
    jets = {
        "U": {"value": zeros, "R": zeros, "Z": zeros},
        "Omega": {"value": -np.full(3, 8.0), "R": zeros, "Z": zeros},
        "Psi": {
            "value": R**2,
            "R": 2 * R,
            "Z": zeros,
            "RR": np.full(3, 2.0),
            "ZZ": zeros,
        },
    }
    residual = residuals_from_jets(points, jets, {"lambda": 0.5, "C_raw": 0})
    np.testing.assert_allclose(residual[:, 2], 0, atol=1e-14)


def test_frozen_points_reproducible_and_invalid_domain_rejected(tmp_path):
    points = sample_points(33)
    assert points_hash(points) == points_hash(sample_points(33))
    assert points_hash(points) != points_hash(sample_points(33, seed=24))
    path = tmp_path / "points.npz"
    np.savez(path, **points)
    assert points_hash(points) == points_hash(load_points(path))
    points["axis"][0, 0] = 0.1
    np.savez(path, **points)
    with pytest.raises(ValueError, match="axis"):
        load_points(path)


@pytest.mark.parametrize("mode", ["eager", "vectorized"])
def test_torch_local_coefficients_against_analytic_polynomial(mode):
    pytest.importorskip("torch")
    from flashns.euler import scipy_residuals, upstream_torch_checker
    from flashns.torch_spline import make_local_checker

    root = Path(__file__).resolve().parents[1]
    if not (root / "sources/vendor/eulerRepo/splineEvaluator.py").is_file():
        pytest.skip("requires pinned upstream evaluator; run fetch-sources")
    meta = {"lambda": 0.5, "C_raw": 2.0, "degree": 3}
    surfaces = {name: polynomial_surface() for name in ["U", "Omega", "Psi"]}
    upstream = upstream_torch_checker(root, surfaces, meta, "cpu")
    candidate = make_local_checker(upstream, meta, "cpu", mode=mode)
    points = np.array([[0, 0], [0, 0.3], [0.2, -0.3], [1, 0.1]])
    expected = scipy_residuals(surfaces, meta, points)
    np.testing.assert_allclose(
        candidate(points, batch_size=3), expected, atol=2e-12, rtol=2e-12
    )
    with pytest.raises(ValueError, match="domain"):
        candidate(np.array([[4, 0]]))
