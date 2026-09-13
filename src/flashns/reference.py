"""Independent arbitrary-precision ODE and field checks (not interval bounds)."""

from itertools import pairwise

import mpmath as mp
import numpy as np


def high_precision_trajectory(y0, times, D, G, frequency, *, dps=80, substeps=64):
    """RK4 in mpmath; callers compare substeps and 2*substeps for truncation.

    Floats are converted exactly: comparisons isolate evaluation/integration
    error on the same FP64 inputs, not decimal-to-binary representation error.
    """
    if dps < 30 or substeps < 1:
        raise ValueError("at least 30 digits and one substep required")
    with mp.workdps(dps):
        freq = mp.mpf(float(frequency))
        y = mp.matrix([mp.mpf(float(v)) for v in y0])
        matrix = mp.matrix([[mp.mpf(float(v)) for v in row] for row in D])
        gradient = [mp.mpf(float(v)) for v in G]

        def fun(state):
            z1, z2, th, om = state
            return mp.matrix(
                [
                    -matrix[0, 0] * z1 - matrix[1, 0] * z2,
                    -matrix[0, 1] * z1 - matrix[1, 1] * z2,
                    -(-z2 * gradient[0] + z1 * gradient[1])
                    * om
                    / (freq * (z1 * z1 + z2 * z2)),
                    freq * z1 * th,
                ]
            )

        out = [list(y)]
        for left, right in pairwise(times):
            step = (mp.mpf(float(right)) - mp.mpf(float(left))) / substeps
            for _ in range(substeps):
                k1 = fun(y)
                k2 = fun(y + step * k1 / 2)
                k3 = fun(y + step * k2 / 2)
                k4 = fun(y + step * k3)
                y += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            out.append(list(y))
        return out


def trajectory_error(actual, reference):
    if np.asarray(actual).shape != np.asarray(reference, dtype=object).shape:
        raise ValueError("trajectory comparison shape mismatch")
    if not np.isfinite(actual).all():
        raise ValueError("nonfinite trajectory")
    with mp.workdps(80):
        return max(
            float(abs(mp.mpf(float(a)) - r) / (1 + abs(r)))
            for row, ref in zip(actual, reference)
            for a, r in zip(row, ref)
        )


def reference_delta(coarse, fine):
    if np.asarray(coarse, dtype=object).shape != np.asarray(fine, dtype=object).shape:
        raise ValueError("reference comparison shape mismatch")
    with mp.workdps(80):
        return max(
            float(abs(a - b) / (1 + abs(b)))
            for row, ref in zip(coarse, fine)
            for a, b in zip(row, ref)
        )


def differentiated_fields(point, state, frequency, coefficients=(1.0,), *, dps=80):
    """Differentiate the streamfunction and scalar field with mpmath.diff."""
    with mp.workdps(dps):
        x = tuple(mp.mpf(float(v)) for v in point)
        z1, z2, th, om = [mp.mpf(float(v)) for v in state]
        lam = mp.mpf(float(frequency))
        coeff = [mp.mpf(float(v)) for v in coefficients]

        def phase(a, b):
            return lam * (z1 * a + z2 * b)

        def psi(a, b):
            p = sum(-c * mp.cos(n * phase(a, b)) / n for n, c in enumerate(coeff, 1))
            return om * p / (lam**2 * (z1 * z1 + z2 * z2))

        def temp(a, b):
            return th * sum(c * mp.sin(n * phase(a, b)) for n, c in enumerate(coeff, 1))

        def v1(a, b):
            return -mp.diff(psi, (a, b), (0, 1))

        def v2(a, b):
            return mp.diff(psi, (a, b), (1, 0))

        def vort(a, b):
            return mp.diff(psi, (a, b), (2, 0)) + mp.diff(psi, (a, b), (0, 2))

        return {
            "theta": temp(*x),
            "streamfunction": psi(*x),
            "velocity": [v1(*x), v2(*x)],
            "vorticity": vort(*x),
            "grad_theta": [mp.diff(temp, x, (1, 0)), mp.diff(temp, x, (0, 1))],
            "grad_velocity": [
                [mp.diff(v, x, (1, 0)), mp.diff(v, x, (0, 1))] for v in [v1, v2]
            ],
            "grad_vorticity": [mp.diff(vort, x, (1, 0)), mp.diff(vort, x, (0, 1))],
        }


def field_error(actual, reference, *, dps=80):
    if np.asarray(actual).shape != np.asarray(reference, dtype=object).shape:
        raise ValueError("field comparison shape mismatch")
    if not np.isfinite(actual).all():
        raise ValueError("nonfinite field")
    with mp.workdps(dps):
        return max(
            float(abs(mp.mpf(float(a)) - r) / (1 + abs(r)))
            for a, r in zip(
                np.asarray(actual).flat, np.asarray(reference, dtype=object).flat
            )
        )
