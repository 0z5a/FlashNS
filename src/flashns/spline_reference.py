"""Independent mpmath differentiation of the tensor-product spline value.

Only point checks. Neither mpmath nor the sampled residuals enclose a domain.
"""

import mpmath as mp
import numpy as np


def high_precision_jet(surface, point, *, second=False, dps=80):
    with mp.workdps(dps):
        axes = []
        for axis in [0, 1]:
            knots = surface.t[axis]
            degree = surface.k[axis]
            x = mp.mpf(float(point[axis]))
            span = int(
                np.clip(
                    np.searchsorted(knots, float(x), side="right") - 1,
                    degree,
                    surface.c.shape[axis] - 1,
                )
            )
            exact_knots = [
                mp.mpf(float(k)) for k in knots[span - degree : span + degree + 1]
            ]

            def local_values(value, degree=degree, exact_knots=exact_knots):
                weights = [mp.mpf(1)]
                for level in range(1, degree + 1):
                    next_weights = [mp.mpf(0)] * (level + 1)
                    for j, weight in enumerate(weights):
                        lo = exact_knots[degree + 1 - level + j]
                        hi = exact_knots[degree + 1 + j]
                        if hi == lo:
                            continue
                        next_weights[j] += (hi - value) * weight / (hi - lo)
                        next_weights[j + 1] += (value - lo) * weight / (hi - lo)
                    weights = next_weights
                return weights

            values = local_values(x)
            first = [
                mp.diff(lambda v, i=i: local_values(v)[i], x) for i in range(degree + 1)
            ]
            derivs = [values, first]
            if second:
                derivs.append(
                    [
                        mp.diff(lambda v, i=i: local_values(v)[i], x, 2)
                        for i in range(degree + 1)
                    ]
                )
            axes.append((span - degree, derivs))
        i0, bx = axes[0]
        j0, bz = axes[1]
        coefficients = [
            [mp.mpf(float(surface.c[i0 + i, j0 + j])) for j in range(surface.k[1] + 1)]
            for i in range(surface.k[0] + 1)
        ]
        requested = [(0, 0), (1, 0), (0, 1)] + ([(2, 0), (0, 2)] if second else [])
        return {
            nu: mp.fsum(
                coefficients[i][j] * bx[nu[0]][i] * bz[nu[1]][j]
                for i in range(len(coefficients))
                for j in range(len(coefficients[0]))
            )
            for nu in requested
        }


def high_precision_residual(surfaces, meta, point, *, dps=80):
    with mp.workdps(dps):
        x, z = [mp.mpf(float(v)) for v in point]
        R, Z = mp.sinh(x), mp.sinh(z)
        jets = {}
        for name, surface in surfaces.items():
            c = high_precision_jet(surface, point, second=name == "Psi", dps=dps)
            out = {"v": c[0, 0], "r": c[1, 0] / mp.cosh(x), "z": c[0, 1] / mp.cosh(z)}
            if name == "Psi":
                out["rr"] = (c[2, 0] - c[1, 0] * mp.tanh(x)) / mp.cosh(x) ** 2
                out["zz"] = (c[0, 2] - c[0, 1] * mp.tanh(z)) / mp.cosh(z) ** 2
            jets[name] = out
        u, w, p = [jets[k] for k in ["U", "Omega", "Psi"]]
        lam, C = mp.mpf(float(meta["lambda"])), mp.mpf(float(meta["C_raw"]))
        vr = R * (lam - p["z"])
        vz = C + lam * Z + 2 * p["v"] + R * p["r"]
        radial = 3 * p["rr"] if abs(R) <= mp.mpf(1e-8) else 3 * p["r"] / R
        return [
            u["v"] + vr * u["r"] + vz * u["z"] - 2 * u["v"] * p["z"],
            (1 + lam) * w["v"] + vr * w["r"] + vz * w["z"] - 2 * u["v"] * u["z"],
            w["v"] + p["rr"] + radial + p["zz"],
        ]
