"""FP64 local-coefficient Torch prototype, executable on CPU or CUDA.

One local coefficient gather per surface and point batch; derivative
coefficients are reused without constructing autodiff graphs. The pinned
upstream local basis evaluator remains the baseline basis implementation.
"""


def vectorized_basis(x, knots, degree, control_count):
    """Cox-de Boor recurrence with one tensor operation group per degree.

    Caller validates the complete point set on CPU before dispatch. All
    inputs, weights and divisions retain FP64; no runtime bounds reduction
    forces a CUDA-to-CPU synchronization inside each recurrence.
    """
    import torch

    span = torch.searchsorted(knots, x, right=True) - 1
    span = span.clamp(min=degree, max=control_count - 1)
    weights = torch.ones_like(x)[:, None]
    zero = torch.zeros_like(x)[:, None]
    for level in range(1, degree + 1):
        offset = torch.arange(level, device=x.device)
        lo = knots[span[:, None] + 1 - level + offset]
        hi = knots[span[:, None] + 1 + offset]
        gap = hi - lo
        safe = torch.where(gap != 0, gap, torch.ones_like(gap))
        scaled = torch.where(gap != 0, weights / safe, torch.zeros_like(weights))
        left = (hi - x[:, None]) * scaled
        right = (x[:, None] - lo) * scaled
        weights = torch.cat([left, zero], dim=1) + torch.cat([zero, right], dim=1)
    indices = span[:, None] - degree + torch.arange(degree + 1, device=x.device)
    return weights, indices


def make_local_checker(upstream, meta, device, *, mode="eager"):
    import numpy as np
    import torch

    if mode not in {"eager", "vectorized", "compiled"}:
        raise ValueError("unknown Torch spline mode")
    surfaces = upstream.surfaces
    local_basis = upstream.local_basis if mode == "eager" else vectorized_basis
    bounds = [
        (
            float(s.knots_x[s.degree_x].cpu()),
            float(s.knots_x[-s.degree_x - 1].cpu()),
            float(s.knots_y[s.degree_y].cpu()),
            float(s.knots_y[-s.degree_y - 1].cpu()),
        )
        for s in surfaces.values()
    ]

    def jet(surface, x, z, second):
        basis = {}
        indices = {}
        degree = (surface.degree_x, surface.degree_y)
        knots = (surface.knots_x, surface.knots_y)
        for axis, coordinate in enumerate([x, z]):
            for order in range(3 if second else 2):
                trimmed = (
                    knots[axis][order : len(knots[axis]) - order]
                    if order
                    else knots[axis]
                )
                basis[axis, order], indices[axis, order] = local_basis(
                    coordinate,
                    trimmed,
                    degree[axis] - order,
                    surface.coeffs.shape[axis] - order,
                )
        tile = surface.coeffs[indices[0, 0][:, :, None], indices[1, 0][:, None, :]]

        def contract(c, dx=0, dz=0):
            return torch.sum(
                basis[0, dx][:, :, None] * c * basis[1, dz][:, None, :], dim=(1, 2)
            )

        out = {(0, 0): contract(tile)}
        for axis in [0, 1]:
            p = degree[axis]
            t = knots[axis]
            i = indices[axis, 0][:, :-1]
            factor = p / (t[i + p + 1] - t[i + 1])
            first = torch.diff(tile, dim=axis + 1) * (
                factor[:, :, None] if axis == 0 else factor[:, None, :]
            )
            out[(1, 0) if axis == 0 else (0, 1)] = contract(first, 1 - axis, axis)
            if second:
                i = i[:, :-1]
                factor = (p - 1) / (t[i + p + 1] - t[i + 2])
                coeff = torch.diff(first, dim=axis + 1) * (
                    factor[:, :, None] if axis == 0 else factor[:, None, :]
                )
                out[(2, 0) if axis == 0 else (0, 2)] = contract(
                    coeff, 2 * (1 - axis), 2 * axis
                )
        cx, cz = torch.cosh(x), torch.cosh(z)
        physical = {"v": out[0, 0], "r": out[1, 0] / cx, "z": out[0, 1] / cz}
        if second:
            physical["rr"] = (out[2, 0] - out[1, 0] * torch.tanh(x)) / cx**2
            physical["zz"] = (out[0, 2] - out[0, 1] * torch.tanh(z)) / cz**2
        return physical

    def tensor_residual(batch):
        x, z = batch[:, 0].contiguous(), batch[:, 1].contiguous()
        U, W, P = [
            jet(surfaces[name], x, z, name == "Psi") for name in ["U", "Omega", "Psi"]
        ]
        R, Z = torch.sinh(x), torch.sinh(z)
        lam, C = meta["lambda"], meta["C_raw"]
        vr = R * (lam - P["z"])
        vz = C + lam * Z + 2 * P["v"] + R * P["r"]
        axis = R.abs() <= 1e-8
        safe = torch.where(axis, torch.ones_like(R), R)
        radial = torch.where(axis, 3 * P["rr"], 3 * P["r"] / safe)
        return torch.stack(
            [
                U["v"] + vr * U["r"] + vz * U["z"] - 2 * U["v"] * P["z"],
                (1 + lam) * W["v"] + vr * W["r"] + vz * W["z"] - 2 * U["v"] * U["z"],
                W["v"] + P["rr"] + radial + P["zz"],
            ],
            dim=1,
        )

    dispatch = (
        torch.compile(tensor_residual, fullgraph=True, dynamic=False)
        if mode == "compiled"
        else tensor_residual
    )

    def run(points, batch_size=4096):
        points = np.asarray(points, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) == 0
            or not np.isfinite(points).all()
        ):
            raise ValueError("expected a finite, nonempty (n,2) point array")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        lo, hi = points.min(axis=0), points.max(axis=0)
        if any(
            lo[0] < a or hi[0] > b or lo[1] < c or hi[1] > d for a, b, c, d in bounds
        ):
            raise ValueError("point outside valid spline domain")
        results = []
        with torch.no_grad():
            for start in range(0, len(points), batch_size):
                batch = torch.as_tensor(
                    points[start : start + batch_size],
                    dtype=torch.float64,
                    device=device,
                )
                results.append(dispatch(batch).cpu())
        result = torch.cat(results).numpy()
        if not np.isfinite(result).all():
            raise ValueError("nonfinite Torch residuals")
        return result

    run.tensor_residual = tensor_residual
    run.tensor_jet = jet
    return run
