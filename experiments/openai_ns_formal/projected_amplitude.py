"""Finite-coefficient adaptation of paper (7.3)--(7.6), (7.13).

This evaluates one local wave-amplitude equation, not the global construction.
Complex vectors are stored as two real vectors to retain explicit FP64 policy.
Inputs are [n(3), n'(3), F, RF_R, G_R, delta, k, m, Re(t)(3), Im(t)(3),
Re(f)(3), Im(f)(3)]; outputs are [Re(t')(3), Im(t')(3), Re(pi), Im(pi)].
The discrete frequency k*p is fixed before numerical differentiation.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

COMMIT = "8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538"
PAPER_SHA256 = "0e779481c4da40bd28d1e642e1d8ca57447d129610df28dfa5a11e9af8ae228f"
ROOT = Path(__file__).resolve().parents[2]


def contract():
    return {
        "workload_source_id": "openai_ns_formal",
        "object": "local_projected_complex_wave_amplitude",
        "repository_commit": COMMIT,
        "paper_sha256": PAPER_SHA256,
        "paper_equations": ["7.3", "7.4", "7.5", "7.6", "7.13"],
        "lean_file": "NavierStokes/TangentProjection.lean",
        "lean_definitions": ["tangentProj", "projectedRhs", "pressureCoefficient"],
        "lean_theorems": [
            "normal_projectedRhs",
            "pressure_cancellation",
            "tangency_preserved",
            "complex_pressure_sign",
        ],
        "numbering_note": "Pinned Lean comments refer to older equation (27), Lemma 8.4; algebra matches this PDF's (7.13).",
        "precision": "float64, complex128 represented by real and imaginary lanes",
        "epsilon_values": [2.0**-i for i in (8, 12, 16, 24, 32)],
        "local_parameters": "F in [0.5,2], a in [2.25,5], b in [-2,2], R in [0.5,2]; g=F*(-a,b), u*=1, Ls in {16,32,64,128}; F_Z=G_Z=0",
        "discrete_frequency": "k=ceil(epsilon^-1/2); k*p nearest nonzero integer, exact tie chooses larger integer, zero tie chooses +1",
        "scope_limit": "Synthetic finite local coefficients satisfying the displayed local algebra and positive lambda0; not a numerical realization of Theorem 4.6 profiles, q*, all cutoffs, all correction stages, or global blowup.",
        "operator_tolerance": {"atol": 1e-11, "rtol": 5e-13},
        "trajectory_tolerance": {"atol": 3e-9, "rtol": 3e-8},
    }


def make_case(batch, seed=730613):
    generator = np.random.default_rng(seed)
    fbase = generator.uniform(0.5, 2, batch)
    a = generator.uniform(2.25, 5, batch)
    b = generator.uniform(-2, 2, batch)
    b[::17] = 0  # Exercise nearest-nonzero angular frequency.
    radius = generator.uniform(0.5, 2, batch)
    g = fbase[:, None] * np.stack((-a, b), axis=-1)
    length = np.linalg.norm(g, axis=-1)
    normal = g / length[:, None]
    transverse = np.stack((-normal[:, 1], normal[:, 0]), axis=-1)
    rate = np.sqrt(-2 * fbase * normal[:, 0] * (2 * fbase * normal[:, 0] + length))
    eps = np.array(contract()["epsilon_values"])[np.arange(batch) % 5]
    k = np.ceil(eps**-0.5)
    m = np.array([-3, -1, 1, 2])[np.arange(batch) % 4].astype(float)
    sign = np.where(np.arange(batch) % 2, -1.0, 1.0)
    ls = np.array([16, 32, 64, 128])[np.arange(batch) % 4].astype(float)
    bs = np.sqrt(rate / (eps * k * k * 2**1.5))
    wave = bs[:, None] * (
        transverse - sign[:, None] / ls[:, None] * g / length[:, None] ** 2
    )
    kp_unrounded = k * radius * wave[:, 0]
    kp = np.floor(kp_unrounded + 0.5)
    kp = np.where(kp == 0, np.where(kp_unrounded < 0, -1, 1), kp)
    p, pz = kp / k, wave[:, 1]
    fr = g[:, 0] / radius
    npulse = np.stack(
        (-p * fr - pz * g[:, 1], np.zeros(batch), np.zeros(batch)), axis=-1
    )
    nzero = np.stack((sign * bs / 2, p / radius, pz), axis=-1)
    pulse = generator.random(batch) * ls
    n = nzero + pulse[:, None] * npulse
    damping_scale = eps * k * k * m * m
    delta = damping_scale * np.sum(n * n, axis=-1)
    t = generator.normal(size=(batch, 2, 3))
    t[: batch // 2] -= (
        n[: batch // 2, None, :]
        * (
            np.sum(n[: batch // 2, None, :] * t[: batch // 2], axis=-1)
            / np.sum(n[: batch // 2] ** 2, axis=-1)[:, None]
        )[:, :, None]
    )
    source = generator.normal(size=(batch, 2, 3))
    data = np.concatenate(
        (
            n,
            npulse,
            fbase[:, None],
            g,
            delta[:, None],
            k[:, None],
            m[:, None],
            t.reshape(batch, 6),
            source.reshape(batch, 6),
        ),
        axis=-1,
    )
    if data.shape != (batch, 24) or not np.isfinite(data).all() or np.any(kp == 0):
        raise RuntimeError("invalid coefficient batch")
    metadata = {
        "batch": batch,
        "seed": seed,
        "input_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "angular_integer_sha256": hashlib.sha256(
            kp.astype(np.int64).tobytes()
        ).hexdigest(),
        "minimum_normal_squared": float(np.min(np.sum(n * n, axis=-1))),
        "minimum_lambda0": float(rate.min()),
        "nonzero_angular_frequency": bool(np.all(kp != 0)),
    }
    geometry = {
        "nzero": nzero,
        "nprime": npulse,
        "damping_scale": damping_scale,
        "Ls": ls,
        "pulse": pulse,
    }
    return np.ascontiguousarray(data), metadata, geometry


def numpy_operator(data):
    n, nprime = data[:, :3], data[:, 3:6]
    t, source = data[:, 12:18].reshape(-1, 2, 3), data[:, 18:24].reshape(-1, 2, 3)
    kt = np.stack(
        (
            -2 * data[:, 6, None] * t[:, :, 1],
            (2 * data[:, 6, None] + data[:, 7, None]) * t[:, :, 0],
            data[:, 8, None] * t[:, :, 0],
        ),
        axis=-1,
    )
    coefficient = (
        np.sum(n[:, None, :] * (kt + source), axis=-1)
        - np.sum(nprime[:, None, :] * t, axis=-1)
    ) / np.sum(n * n, axis=-1)[:, None]
    derivative = (
        -kt
        - data[:, 9, None, None] * t
        - source
        + n[:, None, :] * coefficient[:, :, None]
    )
    pressure = (
        np.stack((-coefficient[:, 1], coefficient[:, 0]), axis=-1)
        / (data[:, 10] * data[:, 11])[:, None]
    )
    return np.concatenate((derivative.reshape(-1, 6), pressure), axis=-1)


def torch_operator(data):
    import torch

    n, nprime = data[:, :3], data[:, 3:6]
    t, source = data[:, 12:18].reshape(-1, 2, 3), data[:, 18:24].reshape(-1, 2, 3)
    kt = torch.stack(
        (
            -2 * data[:, 6, None] * t[:, :, 1],
            (2 * data[:, 6, None] + data[:, 7, None]) * t[:, :, 0],
            data[:, 8, None] * t[:, :, 0],
        ),
        dim=-1,
    )
    coefficient = (
        (n[:, None, :] * (kt + source)).sum(-1) - (nprime[:, None, :] * t).sum(-1)
    ) / (n * n).sum(-1)[:, None]
    derivative = (
        -kt
        - data[:, 9, None, None] * t
        - source
        + n[:, None, :] * coefficient[:, :, None]
    )
    pressure = (
        torch.stack((-coefficient[:, 1], coefficient[:, 0]), dim=-1)
        / (data[:, 10] * data[:, 11])[:, None]
    )
    return torch.cat((derivative.reshape(-1, 6), pressure), dim=-1)


def saddle_reference(data, precision=100):
    """Solve (7.5) plus differentiated tangency, without using (7.13)."""
    import mpmath as mp

    results = []
    with mp.workdps(precision):
        for row in data:
            values = [mp.mpf(float(value)) for value in row]
            n, npulse = values[:3], values[3:6]
            fbase, rfr, gr, delta, k, m = values[6:12]
            matrix = mp.eye(4)
            for i in range(3):
                matrix[i, 3] = n[i]
                matrix[3, i] = n[i]
            matrix[3, 3] = 0
            output, pressure = [], []
            for component in range(2):
                t = values[12 + 3 * component : 15 + 3 * component]
                source = values[18 + 3 * component : 21 + 3 * component]
                kt = [-2 * fbase * t[1], (2 * fbase + rfr) * t[0], gr * t[0]]
                right = mp.matrix(
                    [-kt[i] - delta * t[i] - source[i] for i in range(3)]
                    + [-sum(npulse[i] * t[i] + delta * n[i] * t[i] for i in range(3))]
                )
                solution = mp.lu_solve(matrix, right)
                output.extend(float(solution[i]) for i in range(3))
                pressure.append(-solution[3] / (k * m))
            output.extend((-float(pressure[1]), float(pressure[0])))
            results.append(output)
    return np.array(results)


def compare(actual, expected, atol=1e-11, rtol=5e-13):
    error = np.abs(actual - expected)
    ratio = float(np.max(error / (atol + rtol * np.abs(expected))))
    result = {
        "max_abs": float(error.max()),
        "max_tolerance_ratio": ratio,
        "atol": atol,
        "rtol": rtol,
        "passed": bool(np.isfinite(ratio) and ratio <= 1),
    }
    if not result["passed"]:
        raise AssertionError(result)
    return result


def trajectory_validation():
    """Independent Radau saddle-system evolution versus projected DOP853."""
    from scipy.integrate import solve_ivp

    data, _, geometry = make_case(8, seed=731300)
    results = []
    for index in range(8):
        row = data[index].copy()
        nzero, nprime = geometry["nzero"][index], geometry["nprime"][index]
        scale = geometry["damping_scale"][index]
        duration = 8.0
        sample_times = np.linspace(0, duration, 129)
        initial = np.zeros(6)
        if index % 2:
            initial[:3] = nzero / np.dot(nzero, nzero) * 0.25

        def at_time(time, state):
            current = row.copy()
            current[:3] = nzero + time * nprime
            current[9] = scale * np.dot(current[:3], current[:3])
            current[12:18] = state
            current[18:24] = np.sin(time * np.arange(1, 7) / 9) * np.exp(-time / 4)
            return current

        def projected(time, state):
            return numpy_operator(at_time(time, state)[None])[0, :6]

        def saddle(time, state):
            current = at_time(time, state)
            n, npulse = current[:3], current[3:6]
            t, source = current[12:18].reshape(2, 3), current[18:24].reshape(2, 3)
            matrix = np.eye(4)
            matrix[:3, 3] = n
            matrix[3, :3] = n
            matrix[3, 3] = 0
            kmatrix = np.array(
                [
                    [0, -2 * current[6], 0],
                    [2 * current[6] + current[7], 0, 0],
                    [current[8], 0, 0],
                ]
            )
            right = np.vstack(
                (
                    (-t @ kmatrix.T - current[9] * t - source).T,
                    -t @ npulse - current[9] * (t @ n),
                )
            )
            return np.linalg.solve(matrix, right)[:3].T.reshape(6)

        first = solve_ivp(
            projected,
            (0, duration),
            initial,
            method="DOP853",
            rtol=1e-13,
            atol=2e-15,
            t_eval=sample_times,
        )
        second = solve_ivp(
            saddle,
            (0, duration),
            initial,
            method="Radau",
            rtol=2e-12,
            atol=2e-13,
            t_eval=sample_times,
        )
        if not first.success or not second.success:
            raise RuntimeError("trajectory integrator failed")
        agreement = compare(first.y, second.y, atol=3e-9, rtol=3e-8)
        normals = nzero + sample_times[:, None] * nprime
        defect = np.sum(normals[:, None, :] * first.y.T.reshape(-1, 2, 3), axis=-1)
        primitive = scale * (
            np.dot(nzero, nzero) * sample_times
            + np.dot(nzero, nprime) * sample_times**2
            + np.dot(nprime, nprime) * sample_times**3 / 3
        )
        exact_defect = (
            np.dot(initial.reshape(2, 3), nzero)[None, :] * np.exp(-primitive)[:, None]
        )
        defect_check = compare(defect, exact_defect, atol=3e-10, rtol=3e-8)
        results.append(
            {
                "label": index,
                "initial_tangent": index % 2 == 0,
                "duration": duration,
                "agreement": agreement,
                "defect_decay": defect_check,
                "dop853_nfev": first.nfev,
                "radau_nfev": second.nfev,
            }
        )
    return results


def source_hashes():
    paths = [
        *Path(__file__).parent.glob("*.py"),
        ROOT
        / "sources/openai_ns_formal/repository/NavierStokes/TangentProjection.lean",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if path.exists()
    }


if __name__ == "__main__":
    data, metadata, _ = make_case(96)
    print(
        json.dumps(
            {
                "contract": contract(),
                "case": metadata,
                "saddle_100_digits": compare(
                    numpy_operator(data), saddle_reference(data)
                ),
                "trajectories": trajectory_validation(),
            },
            indent=2,
        )
    )
