"""Boussinesq paper, Lemma 3.1, (3.1)-(3.2).

J(a,b)=(-b,a); curl(v)=d_1 v_2-d_2 v_1. These are residual
INCREMENTS on an affine background, not unforced full-field solutions.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


class PrecisionError(ValueError):
    """Input lies outside the declared FP64 evaluation envelope."""


def finite_array(value, shape, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must have shape {shape} and finite FP64 entries")
    return out


@dataclass(frozen=True)
class SineProfile:
    """F(s)=sum c_n sin(n*s); P is its mean-zero even primitive.

    This includes the sine example and checks the lemma on a non-sine
    profile. It does not implement the paper's affine-near-zero profile.
    """

    coefficients: tuple[float, ...] = (1.0,)

    def __post_init__(self):
        if not self.coefficients or not np.isfinite(self.coefficients).all():
            raise ValueError("finite, nonempty sine coefficients required")

    def jet(self, phase):
        phase = np.asarray(phase)
        f = np.zeros_like(phase)
        fp = np.zeros_like(phase)
        fpp = np.zeros_like(phase)
        primitive = np.zeros_like(phase)
        for n, c in enumerate(self.coefficients, 1):
            f += c * np.sin(n * phase)
            fp += n * c * np.cos(n * phase)
            fpp -= n * n * c * np.sin(n * phase)
            primitive -= c * np.cos(n * phase) / n
        return primitive, f, fp, fpp


def validate(D, G, y, frequency):
    D = finite_array(D, (2, 2), "D")
    G = finite_array(G, (2,), "G")
    y = finite_array(y, (4,), "[zeta1,zeta2,Theta,Omega]")
    # A genuinely trace-free matrix is required, rather than silently
    # subtracting the trace of an invalid background supplied by the caller.
    if D[0, 0] != -D[1, 1]:
        raise ValueError("D must be trace-free (D11 == -D22)")
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("lambda must be finite and positive")
    radius = np.hypot(*y[:2])
    if not 1e-100 <= radius <= 1e100:
        raise PrecisionError("|zeta| outside supported FP64 range [1e-100, 1e100]")
    return D, G, y


def rhs(t, y, D, G, frequency):
    """D and G may be arrays or explicit functions of time."""
    D, G, y = validate(
        D(t) if callable(D) else D, G(t) if callable(G) else G, y, frequency
    )
    z, theta, omega = y[:2], y[2], y[3]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            q = z @ z
            jz = np.array([-z[1], z[0]])
            value = np.r_[
                -D.T @ z, -((jz @ G) / q / frequency) * omega, frequency * z[0] * theta
            ]
    except FloatingPointError as exc:
        raise PrecisionError(f"ODE arithmetic outside FP64 envelope: {exc}") from exc
    if not np.isfinite(value).all():
        raise PrecisionError("nonfinite ODE derivative")
    return value


def integrate(y0, times, D, G, frequency, *, rtol=2e-12, atol=1e-14):
    times = np.asarray(times, dtype=np.float64)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("times must be finite and strictly increasing")
    validate(
        D(times[0]) if callable(D) else D,
        G(times[0]) if callable(G) else G,
        y0,
        frequency,
    )
    if (
        not np.isfinite(rtol)
        or rtol <= 0
        or np.any(np.asarray(atol) <= 0)
        or not np.isfinite(atol).all()
    ):
        raise ValueError("positive finite solver tolerances required")
    sol = solve_ivp(
        lambda t, y: rhs(t, y, D, G, frequency),
        (times[0], times[-1]),
        y0,
        t_eval=times,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if (
        not sol.success
        or sol.y.shape != (4, len(times))
        or not np.isfinite(sol.y).all()
    ):
        raise PrecisionError(f"integration failed: {sol.message}")
    return sol.y.T, {
        "method": "DOP853",
        "nfev": sol.nfev,
        "rtol": rtol,
        "atol": np.asarray(atol).tolist(),
    }


def evaluate(points, y, frequency, *, D=None, G=None, profile=None):
    """Evaluate the wave and analytic spatial/time derivatives in FP64.

    Returned derivatives are checked independently with symbolic
    differentiation and mpmath.diff. No finite-difference cancellation is
    hidden inside a residual tolerance. Phase checks are diagnostics, not
    rigorous interval bounds. Underflow, overflow and excessive phase
    sensitivity cause explicit failure.
    """
    D = np.zeros((2, 2)) if D is None else D
    G = np.zeros(2) if G is None else G
    D, G, y = validate(D, G, y, frequency)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite array of shape (n,2)")
    profile = SineProfile() if profile is None else profile
    z, theta, omega = y[:2], y[2], y[3]
    rate = rhs(0, y, D, G, frequency)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            k = frequency * z
            phase = points @ k
            phase_roundoff_estimate = (
                8
                * np.finfo(float).eps
                * len(profile.coefficients)
                * (np.abs(points) @ np.abs(k))
            )
            if np.any(phase_roundoff_estimate > 1e-6):
                raise PrecisionError(
                    "estimated harmonic phase error exceeds 1e-6 radians; use higher precision"
                )
            q = z @ z
            jz = np.array([-z[1], z[0]])
            a = (omega / frequency) * (jz / q)
            p, f, fp, fpp = profile.jet(phase)
            phase_t = points @ (frequency * rate[:2])
            velocity = f[:, None] * a
            grad_theta = (theta * fp)[:, None] * k
            grad_vorticity = (omega * fpp)[:, None] * k
            grad_velocity = fp[:, None, None] * np.outer(a, k)
            theta_t = rate[2] * f + theta * fp * phase_t
            vorticity_t = rate[3] * fp + omega * fpp * phase_t
            background_velocity = points @ D.T
            # v is analytically perpendicular to both wave gradients. A
            # direct FP64 dot product can lose that cancellation entirely
            # when Omega^2 dwarfs the remaining PDE terms. Reject such a
            # residual evaluation instead of disguising it with a huge norm.
            for grad, derivative, source in [
                (grad_theta, theta_t, velocity @ G),
                (grad_vorticity, vorticity_t, -grad_theta[:, 0]),
            ]:
                self_scale = np.sum(np.abs(velocity * grad), axis=1)
                other_scale = (
                    1
                    + np.abs(derivative)
                    + np.abs(source)
                    + np.sum(np.abs(background_velocity * grad), axis=1)
                )
                if np.any(8 * np.finfo(float).eps * self_scale > 1e-8 * other_scale):
                    raise PrecisionError(
                        "self-advection cancellation exceeds residual precision budget"
                    )
            u = background_velocity + velocity
            scalar_terms = np.stack(
                [theta_t, np.einsum("ni,ni->n", u, grad_theta), velocity @ G]
            )
            vort_terms = np.stack(
                [
                    vorticity_t,
                    np.einsum("ni,ni->n", u, grad_vorticity),
                    -grad_theta[:, 0],
                ]
            )
            result = {
                "theta": theta * f,
                "streamfunction": (omega / frequency / frequency / q) * p,
                "velocity": velocity,
                "vorticity": omega * fp,
                "grad_theta": grad_theta,
                "grad_velocity": grad_velocity,
                "grad_vorticity": grad_vorticity,
                "divergence": grad_velocity[:, 0, 0] + grad_velocity[:, 1, 1],
                "curl_error": grad_velocity[:, 1, 0]
                - grad_velocity[:, 0, 1]
                - omega * fp,
                "scalar_residual_increment": scalar_terms.sum(axis=0),
                "vorticity_residual_increment": vort_terms.sum(axis=0),
                "scalar_term_scale": np.abs(scalar_terms).sum(axis=0),
                "vorticity_term_scale": np.abs(vort_terms).sum(axis=0),
                "phase_roundoff_estimate": phase_roundoff_estimate,
            }
    except FloatingPointError as exc:
        raise PrecisionError(f"field arithmetic outside FP64 envelope: {exc}") from exc
    if any(not np.isfinite(a).all() for a in result.values()):
        raise PrecisionError("nonfinite field output")
    return result
