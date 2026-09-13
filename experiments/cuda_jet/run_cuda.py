"""Compile and validate the user-supplied FP64 jet kernels on a CUDA GPU.

Independent experiment: these PINN kernels are not the official Euler spline
artifact. Source inputs are kept unchanged. No GEMM epilogue fusion is claimed.
"""

import argparse
import ctypes
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from functools import partial
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import mpmath as mp
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "input"))
import jet_reference as ref


class NativeJet:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        self.launch = self.library.flashns_launch_jet_activation
        self.launch.argtypes = [
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int

    def __call__(self, value, plan, adjoint=None):
        tensors = [value] + ([] if adjoint is None else [adjoint])
        if (
            plan.dim not in (2, 3)
            or plan.order != 3
            or plan.indices != ref.JetPlan.dense(plan.dim, 3).indices
        ):
            raise ValueError("native kernel requires dense 2-D/3-D order-3 jets")
        for t in tensors:
            if (
                t.ndim != 3
                or t.shape != value.shape
                or t.shape[1] != plan.q
                or t.dtype != torch.float64
                or not t.is_cuda
                or t.device != value.device
                or not t.is_contiguous()
            ):
                raise ValueError("expected matching contiguous CUDA FP64 [B,Q,C]")
        output = torch.empty_like(value)
        with torch.cuda.device(value.device):
            error = self.launch(
                plan.dim,
                adjoint is not None,
                value.data_ptr(),
                adjoint.data_ptr() if adjoint is not None else None,
                output.data_ptr(),
                value.shape[0],
                value.shape[2],
                torch.cuda.current_stream().cuda_stream,
            )
        if error:
            raise RuntimeError(f"CUDA jet launch failed with cudaError_t={error}")
        return output


def comparison(actual, expected, *, atol=2e-12, rtol=2e-11):
    a, b = actual.detach(), expected.detach()
    diff = (a - b).abs()
    ratio = float((diff / (atol + rtol * b.abs())).max())
    result = {
        "max_abs": float(diff.max()),
        "max_tolerance_ratio": ratio,
        "atol": atol,
        "rtol": rtol,
        "passed": math.isfinite(ratio) and ratio <= 1,
    }
    if not result["passed"]:
        raise AssertionError(result)
    return result


def network_forward(x, weights, biases, plan, activation):
    h = ref.seed_coordinates(x, plan)
    checkpoints = [h]
    for layer, (w, b) in enumerate(zip(weights, biases)):
        z = h @ w.T
        z[:, 0] += b
        h = activation(z) if layer + 1 < len(weights) else z
        checkpoints.append(h)
    return h, checkpoints


def network_backward(seed, checkpoints, weights, vjp):
    dws, dbs = [None] * len(weights), [None] * len(weights)
    adjoint = seed
    for layer in reversed(range(len(weights))):
        d = (
            vjp(checkpoints[layer + 1], adjoint)
            if layer + 1 < len(weights)
            else adjoint
        )
        dws[layer] = d.flatten(0, 1).T @ checkpoints[layer].flatten(0, 1)
        dbs[layer] = d[:, 0].sum(0)
        if layer:
            adjoint = d @ weights[layer]
    return dws + dbs


def step(x, weights, biases, plan, activation, vjp):
    with torch.no_grad():
        jets, checkpoints = network_forward(x, weights, biases, plan, activation)
    leaf = jets.detach().requires_grad_()
    loss = ref.residual_loss(leaf, plan)
    (seed,) = torch.autograd.grad(loss, leaf)
    with torch.no_grad():
        gradients = network_backward(seed, checkpoints, weights, vjp)
    return loss.detach(), gradients


def validate(native):
    torch.manual_seed(7309)
    results = []
    for dim in (2, 3):
        plan = ref.JetPlan.dense(dim, 3)
        for batch, channels, scale in [
            (1, 1, 0.2),
            (3, 5, 0.2),
            (17, 33, 0.5),
            (129, 7, 2.0),
            (5, 129, 0.05),
        ]:
            z = (
                torch.randn(batch, plan.q, channels, device="cuda", dtype=torch.float64)
                * scale
            ).requires_grad_()
            bar = torch.randn_like(z)
            h_ref = ref.tanh_jet(z, plan)
            (vjp_ref,) = torch.autograd.grad((h_ref * bar).sum(), z)
            h = native(z, plan)
            vjp = native(h, plan, bar)
            results.append(
                {
                    "dimension": dim,
                    "shape": list(z.shape),
                    "scale": scale,
                    "forward": comparison(h, h_ref),
                    "vjp_vs_autograd": comparison(vjp, vjp_ref),
                }
            )
        # Exercise the supplied stream, odd sizes and an empty launch.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            z = torch.randn(23, plan.q, 35, device="cuda", dtype=torch.float64) * 0.2
            bar = torch.randn_like(z)
            h = native(z, plan)
            b = native(h, plan, bar)
            h_ref, b_ref = ref.tanh_jet(z, plan), ref.tanh_jet_vjp(h, bar, plan)
        stream.synchronize()
        results.append(
            {
                "dimension": dim,
                "nondefault_stream": True,
                "forward": comparison(h, h_ref),
                "vjp": comparison(b, b_ref),
            }
        )
        assert native(z[:0], plan).numel() == 0
        try:
            native(z.transpose(0, 2), plan)
        except ValueError:
            pass
        else:
            raise AssertionError("noncontiguous input accepted")
    assert native.launch(4, False, None, None, None, 0, 0, None) != 0
    assert native.launch(2, False, None, None, None, 1, 1, None) != 0

    plan = ref.JetPlan.dense(2, 3)
    network_results = []
    for coordinate_scale in (0.01, 0.4, 3.0):
        widths = [2, 8, 8, 3]
        weights = [
            (
                torch.randn(o, i, device="cuda", dtype=torch.float64) * 0.2
            ).requires_grad_()
            for i, o in pairwise(widths)
        ]
        biases = [
            (torch.randn(o, device="cuda", dtype=torch.float64) * 0.1).requires_grad_()
            for o in widths[1:]
        ]
        x = (
            torch.randn(5, 2, device="cuda", dtype=torch.float64) * coordinate_scale
        ).requires_grad_()
        activation = lambda z: native(z, plan)
        vjp = lambda h, b: native(h, plan, b)
        with torch.no_grad():
            jets, _ = network_forward(x, weights, biases, plan, activation)
        outputs = ref.mlp(x, weights, biases)
        expected = torch.stack(
            [
                torch.stack(
                    [
                        ref.nested_derivative(outputs[:, c], x, a)
                        / math.prod(math.factorial(v) for v in a)
                        for c in range(3)
                    ],
                    -1,
                )
                for a in plan.indices
            ],
            1,
        )
        loss, gradients = step(x, weights, biases, plan, activation, vjp)
        expected_loss = ref.nested_loss(x, weights, biases)
        expected_grad = torch.autograd.grad(expected_loss, weights + biases)
        network_results.append(
            {
                "widths": widths,
                "batch": 5,
                "coordinate_scale": coordinate_scale,
                "all_output_jets": comparison(jets, expected),
                "loss": comparison(loss, expected_loss),
                "parameter_gradients": [
                    comparison(a, b) for a, b in zip(gradients, expected_grad)
                ],
            }
        )
    return {
        "activation_cases": results,
        "network_cases": network_results,
        "empty_invalid_layout_and_null_launch_checks": True,
        "acceptance_scope": "listed finite FP64 inputs; full third-order output jets and first parameter gradients of 2-D steady NS residual plus residual-gradient loss",
        "passed": True,
    }


def saturation_diagnostics(native):
    plan = ref.JetPlan.dense(2, 3)
    lookup = plan.lookup
    rows = []
    with mp.workdps(100):
        for center in (0, 5, 10, 15, 18, 19, 20, 30, -15, -20, -30):
            z = torch.zeros(1, plan.q, 1, device="cuda", dtype=torch.float64)
            z[:, 0] = center
            z[:, lookup[1, 0]] = 1
            h = native(z, plan)
            bar = torch.zeros_like(h)
            bar[:, lookup[3, 0]] = 1
            b = native(h, plan, bar)
            expected_first = mp.diff(mp.tanh, mp.mpf(center), 1)
            expected_vjp0 = mp.diff(mp.tanh, mp.mpf(center), 4) / 6
            first = float(h[0, lookup[1, 0], 0])
            vjp0 = float(b[0, 0, 0])
            first_relative = float(
                abs(mp.mpf(first) - expected_first) / abs(expected_first)
            )
            rows.append(
                {
                    "z0": center,
                    "forward_first": first,
                    "reference_first_100_digits": str(expected_first),
                    "first_derivative_relative_error": first_relative,
                    "vjp_z0_for_third_jet_seed": vjp0,
                    "reference_vjp0_100_digits": str(expected_vjp0),
                    "vjp0_absolute_error": float(abs(mp.mpf(vjp0) - expected_vjp0)),
                    "tail_relative_1e-8_met": first_relative <= 1e-8,
                }
            )
    return {
        "cases": rows,
        "all_tail_relative_criteria_met": all(
            r["tail_relative_1e-8_met"] for r in rows
        ),
        "scope": "diagnostic only; tail failures do not pass the nonsaturated-input acceptance contract",
        "limitation": "H[0] rounded to +/-1 cannot reconstruct the lost tiny derivative. A robust implementation needs a stable additional checkpoint or recomputation from Z, followed by separate high-precision validation.",
    }


def measure(fn, repeats=7):
    for _ in range(3):
        _value = fn()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, device = [], []
    for _ in range(repeats):
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start = perf_counter()
        begin.record()
        _value = fn()
        end.record()
        end.synchronize()
        wall.append((perf_counter() - start) * 1000)
        device.append(begin.elapsed_time(end))
    return {
        "wall_ms": wall,
        "cuda_event_ms": device,
        "median_wall_ms": statistics.median(wall),
        "median_cuda_event_ms": statistics.median(device),
        "resident_before_bytes": resident,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "warmup_calls": 3,
        "last_result_kept_until_next_call": True,
    }


def paired_measure(baseline, candidate):
    """Alternate execution order to reduce order and GPU clock bias."""
    functions = {"compiled_torch": baseline, "native_cuda": candidate}
    for _ in range(3):
        for fn in functions.values():
            _value = fn()
    torch.cuda.synchronize()
    pairs = []
    for pair in range(9):
        order = ["compiled_torch", "native_cuda"]
        if pair % 2:
            order.reverse()
        row = {"execution_order": order}
        for name in order:
            torch.cuda.synchronize()
            start = perf_counter()
            _value = functions[name]()
            torch.cuda.synchronize()
            row[name + "_wall_ms"] = (perf_counter() - start) * 1000
        row["speedup"] = row["compiled_torch_wall_ms"] / row["native_cuda_wall_ms"]
        pairs.append(row)
    return {
        "pairs": pairs,
        "median_speedup": statistics.median(row["speedup"] for row in pairs),
        "min_speedup": min(row["speedup"] for row in pairs),
        "max_speedup": max(row["speedup"] for row in pairs),
        "compiled_torch_median_wall_ms": statistics.median(
            row["compiled_torch_wall_ms"] for row in pairs
        ),
        "native_cuda_median_wall_ms": statistics.median(
            row["native_cuda_wall_ms"] for row in pairs
        ),
    }


def benchmark(native):
    result = {"activation": [], "network": []}
    compiled = {}
    for dim in (2, 3):
        plan = ref.JetPlan.dense(dim, 3)
        eager_f = lambda z, p=plan: ref.tanh_jet(z, p)
        eager_b = lambda h, b, p=plan: ref.tanh_jet_vjp(h, b, p)
        f = torch.compile(eager_f, fullgraph=True, dynamic=False)
        vjp = torch.compile(eager_b, fullgraph=True, dynamic=False)
        compiled[dim] = (f, vjp)
        for batch in (4096, 16384):
            z = torch.randn(batch, plan.q, 64, device="cuda", dtype=torch.float64) * 0.2
            bar = torch.randn_like(z)
            start = perf_counter()
            h = f(z)
            b = vjp(h, bar)
            torch.cuda.synchronize()
            compile_seconds = perf_counter() - start
            nh = native(z, plan)
            nb = native(nh, plan, bar)
            row = {
                "dimension": dim,
                "shape": list(z.shape),
                "compiled_first_call_seconds": compile_seconds,
                "compiled_forward_vs_native": comparison(nh, h),
                "compiled_vjp_vs_native": comparison(nb, b),
            }
            for label, forward, backward in [
                ("torch_eager", eager_f, eager_b),
                ("torch_compiled", f, vjp),
                (
                    "native_cuda",
                    lambda z, p=plan: native(z, p),
                    lambda h, b, p=plan: native(h, p, b),
                ),
            ]:
                row[label] = measure(
                    lambda f=forward, g=backward, a=z, s=bar: g(f(a), s)
                )
            result["activation"].append(row)
            print(
                f"jet activation benchmark complete: dim={dim}, B={batch}", flush=True
            )

    plan = ref.JetPlan.dense(2, 3)
    f, vjp = compiled[2]
    for batch in (4096, 16384):
        widths = [2, 64, 64, 3]
        weights = [
            torch.randn(o, i, device="cuda", dtype=torch.float64) * (0.7 / math.sqrt(i))
            for i, o in pairwise(widths)
        ]
        biases = [
            torch.randn(o, device="cuda", dtype=torch.float64) * 0.1 for o in widths[1:]
        ]
        x = torch.randn(batch, 2, device="cuda", dtype=torch.float64) * 0.4
        native_step = partial(
            step,
            x,
            weights,
            biases,
            plan,
            lambda z: native(z, plan),
            lambda h, adjoint: native(h, plan, adjoint),
        )
        compiled_step = partial(step, x, weights, biases, plan, f, vjp)
        start = perf_counter()
        c_loss, c_grad = compiled_step()
        torch.cuda.synchronize()
        first_seconds = perf_counter() - start
        n_loss, n_grad = native_step()
        row = {
            "batch": batch,
            "widths": widths,
            "compiled_first_network_step_seconds": first_seconds,
            "loss_agreement": comparison(n_loss, c_loss),
            "parameter_gradient_agreement": [
                comparison(a, b) for a, b in zip(n_grad, c_grad)
            ],
            "torch_compiled_activations_plus_library_gemm": measure(compiled_step),
            "native_activations_plus_library_gemm": measure(native_step),
            "includes": "dense input seeding, library GEMM forward/dgrad/wgrad, activation/VJP, residual-gradient loss and its output seed, bias reductions; no optimizer update",
        }
        row["paired_comparison"] = paired_measure(compiled_step, native_step)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            native_step()
            torch.cuda.synchronize()
        events = [
            {
                "operator": e.key,
                "count": e.count,
                "self_cpu_us": e.self_cpu_time_total,
                "self_device_us": getattr(e, "self_device_time_total", 0),
            }
            for e in profiler.key_averages()
        ]
        row["top_device_events"] = sorted(
            events, key=lambda e: e["self_device_us"], reverse=True
        )[:20]
        result["network"].append(row)
        print(f"jet full parameter-gradient step complete: B={batch}", flush=True)
    result["performance_scope"] = (
        "synthetic finite PINN workload; microkernel and full first parameter-gradient step; no official training checkpoint, scientific convergence, optimizer, GEMM fusion, or Rubin claim"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=HERE / "artifacts/cuda_validation.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7309)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    artifacts = HERE / "artifacts"
    artifacts.mkdir(exist_ok=True)
    capability = torch.cuda.get_device_capability()
    library = artifacts / "libflashns_jet.so"
    command = [
        "nvcc",
        "-O3",
        "-std=c++17",
        f"-arch=sm_{capability[0]}{capability[1]}",
        "-shared",
        "-Xcompiler=-fPIC",
        "--ptxas-options=-v",
        "-lineinfo",
        str(HERE / "input/jet_activation_kernels.cu"),
        "-o",
        str(library),
    ]
    start = perf_counter()
    build = subprocess.run(command, text=True, capture_output=True, check=False)
    (artifacts / "nvcc_build.log").write_text(build.stdout + build.stderr)
    build.check_returncode()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": 7309,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "build_command": command,
        "build_seconds": perf_counter() - start,
        "fast_math": False,
        "matmul_dtype": "torch.float64",
        "matmul_backend": "unchanged PyTorch library dispatch; hardware instruction path not independently audited",
        "source_hashes": {
            str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted((HERE / "input").glob("*"))]
            if p.is_file()
        },
        "scope": "user-supplied independent PINN Taylor-jet experiment",
    }
    native = NativeJet(library)
    report["validation"] = validate(native)
    print("CUDA forward, VJP and full parameter-gradient validation passed", flush=True)
    report["saturation"] = saturation_diagnostics(native)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.benchmark:
        try:
            report["benchmark"] = benchmark(native)
        except Exception as exc:  # noqa: BLE001 -- preserve compiler and experiment failures
            report["benchmark_failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        {
            "output": str(args.output),
            "validation": report["validation"]["passed"],
            "all_saturation_cases_pass": report["saturation"][
                "all_tail_relative_criteria_met"
            ],
            "benchmark_failure": report.get("benchmark_failure"),
        },
        flush=True,
    )
    return 1 if "benchmark_failure" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
