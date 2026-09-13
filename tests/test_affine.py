import mpmath as mp
import numpy as np
import pytest
from scipy.linalg import expm

from flashns.affine import PrecisionError, SineProfile, evaluate, integrate, rhs
from flashns.reference import differentiated_fields, field_error
from flashns.symbolic import identities


def test_generic_primitive_identities_and_sign_mutation():
    assert all(value == 0 for value in identities().values())
    # The independent differentiator must reject the wrong buoyancy sign.
    assert identities(-1)["vorticity_residual_increment"] != 0


def test_analytic_growing_mode():
    times = np.linspace(0, 1, 17)
    actual, _ = integrate([0.6, 0.8, -0.25, -1], times, np.zeros((2, 2)), [0, -4], 8)
    np.testing.assert_allclose(
        actual[:, :2], np.tile([0.6, 0.8], (len(times), 1)), atol=1e-14
    )
    np.testing.assert_allclose(actual[:, 2], -0.25 * np.exp(1.2 * times), rtol=1e-11)
    np.testing.assert_allclose(actual[:, 3], -np.exp(1.2 * times), rtol=1e-11)


def test_phase_transport_matches_matrix_exponential():
    D = np.array([[0.2, -0.3], [0.4, -0.2]])
    times = np.linspace(0, 2, 15)
    actual, _ = integrate([1, 0.5, 0.1, -0.2], times, D, [0, -1], 7)
    expected = np.array([expm(-t * D.T) @ np.array([1, 0.5]) for t in times])
    np.testing.assert_allclose(actual[:, :2], expected, rtol=2e-11, atol=2e-12)


def test_time_dependent_background_phase():
    # D(t)=diag(t,-t) => zeta=(exp(-t^2/2), 2 exp(t^2/2)).
    times = np.linspace(0, 1, 9)
    actual, _ = integrate(
        [1, 2, 0, 0], times, lambda t: [[t, 0], [0, -t]], lambda t: [np.sin(t), 1], 3
    )
    np.testing.assert_allclose(
        actual[:, :2],
        np.stack([np.exp(-(times**2) / 2), 2 * np.exp(times**2 / 2)], axis=1),
        rtol=1e-11,
    )


@pytest.mark.parametrize("coefficients", [(1.0,), (1.0, -0.125, 0.0625)])
def test_fields_against_independent_streamfunction_differentiation(coefficients):
    point = [0.21, -0.15]
    state = [0.8, 0.6, -0.3, 0.7]
    actual = evaluate([point], state, 11, profile=SineProfile(coefficients))
    reference = differentiated_fields(point, state, 11, coefficients)
    for name, value in reference.items():
        assert field_error(actual[name][0], value) < 2e-13, name


@pytest.mark.parametrize(
    "state,frequency,error",
    [
        ([0, 0, 1, 1], 1, PrecisionError),
        ([1e-200, 0, 1, 1], 1, PrecisionError),
        ([1, 0, 1, 1], 0, ValueError),
        ([1, 0, 1, 1], -1, ValueError),
        ([1, 0, 1, 1], float("inf"), ValueError),
        ([float("nan"), 1, 1, 1], 1, ValueError),
        ([1, 0, 1e308, 1e308], 1e100, PrecisionError),
    ],
)
def test_invalid_or_unrepresentable_inputs_fail(state, frequency, error):
    with pytest.raises(error):
        evaluate([[1, 1]], state, frequency)


def test_large_phase_and_underflow_are_explicit():
    with pytest.raises(PrecisionError, match="phase"):
        evaluate([[1, 1]], [1, 0, 1, 1], 1e16)
    with pytest.raises(PrecisionError):
        evaluate([[0.1, 0.2]], [1, 0, 1e-310, 1e-310], 1)
    with pytest.raises(PrecisionError, match="cancellation"):
        evaluate([[0.01, -0.02]], [0.6, 0.8, 1e-60, 1e60], 100)


def test_background_trace_and_time_grid_rejected():
    with pytest.raises(ValueError, match="trace-free"):
        rhs(0, [1, 0, 1, 1], np.eye(2), [0, 1], 1)
    for times in [[0, 0], [1, 0], [0, float("nan")], [0]]:
        with pytest.raises(ValueError):
            integrate([1, 0, 1, 1], times, np.zeros((2, 2)), [0, 1], 1)


def test_zero_amplitude_fields_are_finite_and_zero():
    fields = evaluate([[0, 0], [0.2, -0.3]], [1, 2, 0, 0], 5)
    for key in [
        "theta",
        "velocity",
        "vorticity",
        "scalar_residual_increment",
        "vorticity_residual_increment",
    ]:
        assert np.all(fields[key] == 0)


def test_tiny_nonzero_amplitude_is_preserved_in_relative_error():
    point = [0.125, -0.25]
    state = [0.6, 0.8, 1e-120, -1e-100]
    actual = evaluate([point], state, 8)
    reference = differentiated_fields(point, state, 8)
    with mp.workdps(80):
        assert actual["theta"][0] != 0
        assert abs(mp.mpf(float(actual["theta"][0])) / reference["theta"] - 1) < mp.mpf(
            "1e-14"
        )


def test_comparison_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        field_error([1, 2], [mp.mpf(1)])


def test_finite_superposition_has_cross_residual_and_truncation_error():
    # A finite sum of separately valid local waves need not solve the PDE.
    point = np.array([[0.17, -0.23]])
    waves = [([1, 0.3, 0.4, 0.7], 4), ([0.2, 1, -0.5, 0.6], 7)]
    fields = [evaluate(point, y, lam, G=[0, -1]) for y, lam in waves]
    cross = sum(
        float(a["velocity"][0] @ b["grad_theta"][0])
        for i, a in enumerate(fields)
        for j, b in enumerate(fields)
        if i != j
    )
    with mp.workdps(60):

        def temp(t, x, y):
            value = mp.mpf(0)
            for state, lam in waves:
                rate = rhs(0, state, np.zeros((2, 2)), [0, -1], lam)
                value += (mp.mpf(state[2]) + t * mp.mpf(float(rate[2]))) * mp.sin(
                    lam * (mp.mpf(state[0]) * x + mp.mpf(state[1]) * y)
                )
            return value

        x, y = map(mp.mpf, point[0])
        velocity = sum((f["velocity"][0] for f in fields), np.zeros(2))
        residual = (
            mp.diff(lambda t: temp(t, x, y), 0)
            + velocity[0] * mp.diff(lambda a: temp(0, a, y), x)
            + velocity[1] * mp.diff(lambda b: temp(0, x, b), y)
            - velocity[1]
        )
        assert abs(float(residual) - cross) < 1e-12
    assert abs(cross) > 0.01
    full_theta = sum(f["theta"][0] for f in fields)
    assert abs(full_theta - fields[0]["theta"][0]) > 0.01


def test_compact_streamfunction_cutoff_preserves_divergence_and_changes_curl():
    # Independent compact C-infinity plateau envelope; no claim to replay
    # the paper's transported cutoff hierarchy or its estimates.
    with mp.workdps(60):

        def cutoff(x, y):
            radius2 = x * x + y * y
            if radius2 <= mp.mpf("0.25"):
                return mp.mpf(1)
            if radius2 >= 1:
                return mp.mpf(0)
            z = (1 - radius2) / mp.mpf("0.75")
            a, b = mp.exp(-1 / z), mp.exp(-1 / (1 - z))
            return a / (a + b)

        def psi(x, y):
            return -mp.cos(3 * x + 2 * y) / 13

        def localized(x, y):
            return cutoff(x, y) * psi(x, y)

        point = (mp.mpf("0.7"), mp.mpf("0.2"))

        def velocity1(x, y):
            return -mp.diff(localized, (x, y), (0, 1))

        def velocity2(x, y):
            return mp.diff(localized, (x, y), (1, 0))

        div = mp.diff(velocity1, point, (1, 0)) + mp.diff(velocity2, point, (0, 1))
        # Naively cutting off velocity produces a real divergence defect.
        naive = -mp.diff(
            lambda x, y: cutoff(x, y) * mp.diff(psi, (x, y), (0, 1)), point, (1, 0)
        )
        naive += mp.diff(
            lambda x, y: cutoff(x, y) * mp.diff(psi, (x, y), (1, 0)), point, (0, 1)
        )
        curl = sum(mp.diff(localized, point, nu) for nu in [(2, 0), (0, 2)])
        naive_curl = cutoff(*point) * sum(
            mp.diff(psi, point, nu) for nu in [(2, 0), (0, 2)]
        )
        assert div == 0 and abs(naive) > mp.mpf("0.01")
        assert abs(curl - naive_curl) > mp.mpf("0.01")
        for p in [(mp.mpf("0.1"), mp.mpf("0.2")), (mp.mpf(2), mp.mpf(0))]:
            assert localized(*p) == (psi(*p) if p[0] < 1 else 0)
