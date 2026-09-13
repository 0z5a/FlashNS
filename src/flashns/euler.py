"""Frozen official Euler spline artifact: sampled FP64 residuals only.

SciPy provides an independent analytic-derivative CPU baseline. The Torch
comparison executes selected, hash-verified upstream definitions unchanged.
No training code, Arb certificate, or stability proof is present in this ZIP.
"""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import BSpline, NdBSpline

from .provenance import run_metadata, verify_sources

DOMAINS = ("full_computational_box", "physical_box_radius3", "axis", "far_field")


def sample_points(count=1024, seed=23):
    if count < 2:
        raise ValueError("at least two points per domain required")
    rng = np.random.Generator(np.random.PCG64(seed))
    return {
        "full_computational_box": np.column_stack(
            [rng.uniform(0, 30, count), rng.uniform(-30, 30, count)]
        ),
        "physical_box_radius3": np.arcsinh(
            np.column_stack([rng.uniform(0, 3, count), rng.uniform(-3, 3, count)])
        ),
        "axis": np.column_stack([np.zeros(count), rng.uniform(-30, 30, count)]),
        "far_field": np.column_stack(
            [
                rng.uniform(10, 30, count),
                np.r_[
                    rng.uniform(-30, -10, count // 2),
                    rng.uniform(10, 30, count - count // 2),
                ],
            ]
        ),
    }


def points_hash(points):
    h = hashlib.sha256()
    for name in DOMAINS:
        a = np.asarray(points[name], dtype="<f8")
        h.update(name.encode())
        h.update(np.asarray(a.shape, dtype="<i8").tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def load_points(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(DOMAINS):
            raise ValueError(
                "frozen points must contain exactly the four named domains"
            )
        result = {key: archive[key].copy() for key in DOMAINS}
    for name, points in result.items():
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.isfinite(points).all()
        ):
            raise ValueError(f"invalid points in {name}")
        x, z = points.T
        if not (np.all((x >= 0) & (x <= 30)) and np.all(np.abs(z) <= 30)):
            raise ValueError(f"point outside computational box in {name}")
        if name == "axis" and np.any(x != 0):
            raise ValueError("axis points require xR=0")
        if name == "physical_box_radius3" and (
            np.any(x > np.arcsinh(3)) or np.any(np.abs(z) > np.arcsinh(3))
        ):
            raise ValueError("point outside physical box")
        if name == "far_field" and (np.any(x < 10) or np.any(np.abs(z) < 10)):
            raise ValueError("point outside far-field subdomains")
    return result


def load_surfaces(directory):
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    if not all(np.isfinite(meta[k]) for k in ["lambda", "C_raw"]):
        raise ValueError("nonfinite profile parameters")
    surfaces = {}
    for name in ["U", "Omega", "Psi"]:
        data = json.loads((directory / f"{name}.json").read_text())
        if data["coeffs_layout"] != "column_major":
            raise ValueError("unsupported coefficient layout")
        degrees = (int(data["degree_x"]), int(data["degree_y"]))
        if degrees != (meta["degree"], meta["degree"]):
            raise ValueError("metadata degree disagrees with spline")
        knots = tuple(
            np.asarray(data[k], dtype=np.float64) for k in ["knots_x", "knots_y"]
        )
        shape = (data["nctrl_x"], data["nctrl_y"])
        coefficients = (
            np.asarray(data["coeffs"], dtype=np.float64)
            .reshape(shape, order="F")
            .copy(order="C")
        )
        if not np.isfinite(coefficients).all() or any(
            not np.isfinite(k).all() for k in knots
        ):
            raise ValueError("nonfinite spline data")
        surfaces[name] = NdBSpline(knots, coefficients, degrees, extrapolate=False)
        del data
    return meta, surfaces


def computational_jet(surface, points, second=False):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be finite (n,2)")
    derivatives = [(0, 0), (1, 0), (0, 1)] + ([(2, 0), (0, 2)] if second else [])
    out = {nu: surface(points, nu=nu) for nu in derivatives}
    if any(not np.isfinite(v).all() for v in out.values()):
        raise ValueError(
            "nonfinite spline evaluation or point outside valid knot domain"
        )
    return out


def local_coefficient_jet(surface, points, second=False):
    """Reuse a local coefficient tile and difference it before contraction.

    This reduces cancellation of large coefficients against basis derivatives.
    No complete derivative coefficient arrays are materialized; arithmetic
    remains FP64. SciPy's C implementation evaluates the one-dimensional bases.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be finite (n,2)")
    basis = {}
    indices = {}
    for axis in [0, 1]:
        degree = surface.k[axis]
        if degree < (2 if second else 1):
            raise ValueError("spline degree too low for requested jet")
        for derivative in range(3 if second else 2):
            knots = (
                surface.t[axis][derivative : len(surface.t[axis]) - derivative]
                if derivative
                else surface.t[axis]
            )
            matrix = BSpline.design_matrix(
                points[:, axis], knots, degree - derivative, extrapolate=False
            )
            width = degree - derivative + 1
            basis[axis, derivative] = matrix.data.reshape(len(points), width)
            indices[axis, derivative] = matrix.indices.reshape(len(points), width)
    tile = surface.c[indices[0, 0][:, :, None], indices[1, 0][:, None, :]]

    def contract(coeff, dx=0, dz=0):
        return np.einsum(
            "ni,nij,nj->n", basis[0, dx], coeff, basis[1, dz], optimize=True
        )

    out = {(0, 0): contract(tile)}
    for axis in [0, 1]:
        degree = surface.k[axis]
        knots = surface.t[axis]
        i = indices[axis, 0][:, :-1]
        first_factor = degree / (knots[i + degree + 1] - knots[i + 1])
        first = np.diff(tile, axis=axis + 1) * (
            first_factor[:, :, None] if axis == 0 else first_factor[:, None, :]
        )
        out[(1, 0) if axis == 0 else (0, 1)] = contract(first, 1 - axis, axis)
        if second:
            i = i[:, :-1]
            second_factor = (degree - 1) / (knots[i + degree + 1] - knots[i + 2])
            second_coeff = np.diff(first, axis=axis + 1) * (
                second_factor[:, :, None] if axis == 0 else second_factor[:, None, :]
            )
            out[(2, 0) if axis == 0 else (0, 2)] = contract(
                second_coeff, 2 * (1 - axis), 2 * axis
            )
    if any(not np.isfinite(v).all() for v in out.values()):
        raise ValueError("nonfinite local derivative coefficients")
    return out


def physical_jet(computational, points):
    """R=sinh(x), f_R=f_x/cosh(x), f_RR=(f_xx-f_x*tanh(x))/cosh(x)^2."""
    cx, cz = np.cosh(points[:, 0]), np.cosh(points[:, 1])
    out = {
        "value": computational[(0, 0)],
        "R": computational[(1, 0)] / cx,
        "Z": computational[(0, 1)] / cz,
    }
    if (2, 0) in computational:
        out["RR"] = (
            computational[(2, 0)] - computational[(1, 0)] * np.tanh(points[:, 0])
        ) / cx**2
        out["ZZ"] = (
            computational[(0, 2)] - computational[(0, 1)] * np.tanh(points[:, 1])
        ) / cz**2
    return out


def residuals_from_jets(points, jets, meta):
    U, W, P = [jets[name] for name in ["U", "Omega", "Psi"]]
    R, Z = np.sinh(points).T
    lam, C = meta["lambda"], meta["C_raw"]
    vr = R * (lam - P["Z"])
    vz = C + lam * Z + 2 * P["value"] + R * P["R"]
    radial = np.empty_like(R)
    axis = np.abs(R) <= 1e-8
    radial[axis] = 3 * P["RR"][axis]
    radial[~axis] = 3 * P["R"][~axis] / R[~axis]
    result = np.column_stack(
        [
            U["value"] + vr * U["R"] + vz * U["Z"] - 2 * U["value"] * P["Z"],
            (1 + lam) * W["value"]
            + vr * W["R"]
            + vz * W["Z"]
            - 2 * U["value"] * U["Z"],
            W["value"] + P["RR"] + radial + P["ZZ"],
        ]
    )
    if not np.isfinite(result).all():
        raise ValueError("nonfinite residual")
    return result


def scipy_residuals(surfaces, meta, points, *, method="direct", batch_size=4096):
    if method not in {"direct", "local_coefficients"}:
        raise ValueError("unknown spline evaluation method")
    if batch_size < 1:
        raise ValueError("positive batch size required")
    if len(points) > batch_size:
        return np.concatenate(
            [
                scipy_residuals(
                    surfaces,
                    meta,
                    points[start : start + batch_size],
                    method=method,
                    batch_size=batch_size,
                )
                for start in range(0, len(points), batch_size)
            ]
        )
    evaluate = computational_jet if method == "direct" else local_coefficient_jet
    jets = {
        name: physical_jet(evaluate(surface, points, second=name == "Psi"), points)
        for name, surface in surfaces.items()
    }
    return residuals_from_jets(points, jets, meta)


def upstream_torch_checker(root, surfaces, meta, device):
    """Use the upstream evaluator and exact upstream getResiduals body.

    Imports only the reviewed evaluator. From the upstream script, executes
    just two equation lambdas and three function definitions. This avoids
    its hard-coded CUDA device, random sampling and top-level million-point
    run. Source hashes are mandatory before executing either module.
    """
    import torch
    from torch.autograd import grad

    root = Path(root)
    verify_sources(root, names=["euler_splineEvaluator", "euler_checkSplineAccuracy"])
    directory = root / "sources/vendor/eulerRepo"
    spec = importlib.util.spec_from_file_location(
        "flashns_pinned_upstream_spline", directory / "splineEvaluator.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Dynamo resolves imported function owners through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    converted = {
        name: module.spline(
            int(s.k[0]),
            int(s.k[1]),
            torch.as_tensor(s.t[0], dtype=torch.float64, device=device),
            torch.as_tensor(s.t[1], dtype=torch.float64, device=device),
            torch.as_tensor(s.c, dtype=torch.float64, device=device),
        )
        for name, s in surfaces.items()
    }
    namespace = {
        "torch": torch,
        "grad": grad,
        "loadedLambda": torch.tensor(
            meta["lambda"], dtype=torch.float64, device=device
        ),
        "loadedC": torch.tensor(meta["C_raw"], dtype=torch.float64, device=device),
    }
    for key, name in [("modelU", "U"), ("modelW", "Omega"), ("modelPsi", "Psi")]:
        namespace[key] = lambda x, z, surface=converted[name]: module.localSplineValue(
            surface, x, z
        )
    tree = ast.parse((directory / "checkSplineAccuracy.py").read_text())
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "chainRule",
            "eq3",
            "getResiduals",
        }:
            selected.append(node)
            found.add(node.name)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"eq1", "eq2"}
        ):
            selected.append(node)
            found.add(node.targets[0].id)
    if found != {"chainRule", "eq1", "eq2", "eq3", "getResiduals"}:
        raise ValueError("pinned upstream checker structure changed")
    exec(  # noqa: S102 -- exact definitions from SHA-256-verified, locally reviewed source
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(directory / "checkSplineAccuracy.py"),
            "exec",
        ),
        namespace,
    )

    def run(points, batch_size=4096):
        result = []
        for start in range(0, len(points), batch_size):
            batch = torch.as_tensor(
                points[start : start + batch_size], dtype=torch.float64, device=device
            )
            x = batch[:, 0:1].contiguous().detach().requires_grad_(True)
            z = batch[:, 1:2].contiguous().detach().requires_grad_(True)
            rows = namespace["getResiduals"](x, z)
            # Every accepted sample returns to the CPU comparison path.
            result.append(torch.cat(rows, dim=1).detach().cpu().numpy())
        return np.concatenate(result)

    run.surfaces = converted
    run.local_basis = module.localBasis
    run.raw_residual = namespace["getResiduals"]
    return run


def sampled_norms(values):
    if not np.isfinite(values).all():
        raise ValueError("nonfinite samples")
    return {
        f"eq{i + 1}": {
            "sample_RMSE": float(np.sqrt(np.mean(values[:, i] ** 2))),
            "sample_max_abs": float(np.max(np.abs(values[:, i]))),
        }
        for i in range(3)
    }


def run_euler(
    root,
    points,
    *,
    device="cpu",
    batch_size=4096,
    repeats=3,
    threads=4,
    torch_mode="eager",
    include_legacy=True,
):
    import torch

    from .torch_spline import make_local_checker

    if device not in ["cpu", "cuda"]:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for this run; no GPU was allocated automatically"
        )
    if min(batch_size, repeats, threads) < 1:
        raise ValueError("batch size, repeats and threads must be positive")
    torch.set_num_threads(threads)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    sources = verify_sources(
        root,
        names=[
            "euler_paper",
            "euler_code",
            "euler_U",
            "euler_Omega",
            "euler_Psi",
            "euler_meta",
        ],
    )
    verified = perf_counter()
    meta, surfaces = load_surfaces(Path(root) / "sources/vendor/eulerRepo/splines")
    loaded = perf_counter()
    checker = upstream_torch_checker(root, surfaces, meta, device)
    local_checker = make_local_checker(checker, meta, device, mode=torch_mode)
    if device == "cuda":
        torch.cuda.synchronize()
    converted = perf_counter()
    outputs = {}
    scipy_times = []
    torch_times = []
    local_cpu_times = []
    local_torch_times = []
    # No discarded warmup: report first and subsequent runs separately.
    for repeat in range(repeats):
        s0 = perf_counter()
        scipy_result = (
            {
                name: scipy_residuals(
                    surfaces, meta, points[name], batch_size=batch_size
                )
                for name in DOMAINS
            }
            if include_legacy
            else None
        )
        s1 = perf_counter()
        torch_result = (
            {name: checker(points[name], batch_size) for name in DOMAINS}
            if include_legacy
            else None
        )
        if device == "cuda":
            torch.cuda.synchronize()
        s2 = perf_counter()
        local_cpu = {
            name: scipy_residuals(
                surfaces,
                meta,
                points[name],
                method="local_coefficients",
                batch_size=batch_size,
            )
            for name in DOMAINS
        }
        s3 = perf_counter()
        local_torch = {
            name: local_checker(points[name], batch_size) for name in DOMAINS
        }
        if device == "cuda":
            torch.cuda.synchronize()
        s4 = perf_counter()
        if include_legacy:
            scipy_times.append(s1 - s0)
            torch_times.append(s2 - s1)
        local_cpu_times.append(s3 - s2)
        local_torch_times.append(s4 - s3)
        outputs[repeat] = (scipy_result, torch_result, local_cpu, local_torch)
    compare_start = perf_counter()
    atol, rtol = 2e-9, 2e-9
    domains = {}
    all_ok = True
    for name in DOMAINS:
        ratios = []
        abs_errors = []
        local_ratios = []
        local_errors = []
        for ref, actual, local_cpu, local_torch in outputs.values():
            if include_legacy:
                delta = np.abs(actual[name] - ref[name])
                if not np.isfinite(delta).all():
                    raise ValueError("nonfinite comparison")
                ratios.append(float(np.max(delta / (atol + rtol * np.abs(ref[name])))))
                abs_errors.append(float(np.max(delta)))
            local_delta = np.abs(local_torch[name] - local_cpu[name])
            if not np.isfinite(local_delta).all():
                raise ValueError("nonfinite local coefficient comparison")
            local_ratios.append(
                float(np.max(local_delta / (atol + rtol * np.abs(local_cpu[name]))))
            )
            local_errors.append(float(np.max(local_delta)))
        ok = max(ratios) <= 1 if include_legacy else None
        local_ok = max(local_ratios) <= 1
        all_ok &= local_ok
        domains[name] = {
            "sample_count": len(points[name]),
            "scipy": sampled_norms(outputs[0][0][name]) if include_legacy else None,
            "upstream_torch": sampled_norms(outputs[0][1][name])
            if include_legacy
            else None,
            "max_abs_backend_difference": max(abs_errors) if include_legacy else None,
            "max_tolerance_ratio_all_repeats": max(ratios) if include_legacy else None,
            "backend_agreement": ok,
        }
        domains[name].update(
            {
                "local_coefficient_cpu": sampled_norms(outputs[0][2][name]),
                "local_coefficient_torch": sampled_norms(outputs[0][3][name]),
                "local_coefficient_backend_agreement": local_ok,
                "local_coefficient_max_abs_difference": max(local_errors),
                "local_coefficient_max_tolerance_ratio": max(local_ratios),
            }
        )
    finished = perf_counter()
    memory_bytes = sum(
        s.c.nbytes + sum(t.nbytes for t in s.t) for s in surfaces.values()
    )
    return {
        "schema_version": 1,
        "adapter": "official_euler_frozen_splines",
        "metadata": run_metadata(root),
        "sources": sources,
        "torch_version": torch.__version__,
        "torch_device": device,
        "torch_local_mode": torch_mode,
        "legacy_baselines_executed": include_legacy,
        "compile_policy": "first run includes any torch.compile cost; subsequent runs reuse compiled kernels"
        if torch_mode == "compiled"
        else "not compiled",
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch_cpu_threads": threads,
        "batch_size": batch_size,
        "profile_parameters": meta,
        "points_sha256": points_hash(points),
        "domains": domains,
        "sampling": {
            "space": "computational xR,xZ; physical R,Z=sinh(xR),sinh(xZ)",
            "measure": "empirical equal-weight samples; physical_box uses uniform physical coordinates",
            "seed_generator": "NumPy PCG64, separate from original Torch CUDA RNG; freeze and compare identical points",
            "axis_rule": "upstream |R|<=1e-8 uses 3*Psi_RR; parity is not an axis-regularity certificate",
        },
        "legacy_backend_agreement": all(
            v["backend_agreement"] for v in domains.values()
        )
        if include_legacy
        else None,
        "acceptance": {
            "scope": "local-coefficient Torch/CPU agreement of sampled FP64 residuals on pinned profile and identical frozen points; legacy disagreement reported separately",
            "norm": "max over points, 3 residuals, all repeats of abs(torch_local-cpu_local)/(atol+rtol*abs(cpu_local))",
            "atol": atol,
            "rtol": rtol,
            "units": "dimensionless profile residuals",
            "rounding": "FP64 round-to-nearest; no outward rounding",
            "criteria_met": bool(all_ok),
        },
        "timing_seconds": {
            "source_hash_check": verified - start,
            "load_json_and_construct_scipy": loaded - verified,
            "upstream_load_and_device_transfer": converted - loaded,
            "scipy_runs": scipy_times,
            "upstream_torch_runs_including_return_to_cpu": torch_times,
            "local_coefficient_cpu_runs": local_cpu_times,
            "local_coefficient_torch_runs_including_return_to_cpu": local_torch_times,
            "backend_comparison": finished - compare_start,
            "total_measured_workflow": finished - start,
            "single_scipy_path_from_sources": (verified - start)
            + (loaded - verified)
            + scipy_times[0]
            if include_legacy
            else None,
            "single_torch_path_from_sources": (verified - start)
            + (loaded - verified)
            + (converted - loaded)
            + torch_times[0]
            if include_legacy
            else None,
            "single_local_cpu_path_from_sources": (verified - start)
            + (loaded - verified)
            + local_cpu_times[0],
            "single_local_torch_path_from_sources": (verified - start)
            + (loaded - verified)
            + (converted - loaded)
            + local_torch_times[0],
            "single_local_crosschecked_path_from_sources": (converted - start)
            + local_cpu_times[0]
            + local_torch_times[0]
            + (finished - compare_start),
        },
        "resident_spline_bytes": memory_bytes,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        "status": {
            "reproduction": "sampled_residual_comparison_completed",
            "training": "not_available_in_archive",
            "gpu_benchmark": "measured" if device == "cuda" else "not_run",
            "arb_certificate": "not_available_in_archive",
            "proof_replay": "not_run",
            "global_blowup": "not_verified",
        },
        "performance_scope": "Frozen-profile sampled residual execution; CPU comparison included in total workflow and crosschecked path, excluded from individual execution paths. Excludes training and rigorous verification; not a publication speedup claim",
    }
