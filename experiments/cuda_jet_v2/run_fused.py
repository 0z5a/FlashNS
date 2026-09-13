"""One-variable FP64 dgrad/jet-VJP fusion experiment; no automatic dispatch."""

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

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "cuda_jet"))
import run_cuda as v1

from flashns.jet_spec import JetSpec

ref = v1.ref
compare = v1.comparison
CUTLASS_COMMIT = "cb4247394dd82148787aed73e5dc7cef33cbf862"


class FusedDgrad:
    """Manual first parameter VJP; inputs stay alive on the current stream.

    As for ordinary PyTorch ops, callers must establish input readiness when
    crossing streams. record_stream handles allocation lifetime, not readiness.
    """

    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        self.launch = self.library.flashns_dgrad_jet_vjp_fp64
        self.launch.argtypes = (
            [ctypes.c_int, ctypes.c_bool]
            + [ctypes.c_void_p] * 4
            + [ctypes.c_int] * 3
            + [ctypes.c_void_p]
        )
        self.launch.restype = ctypes.c_int
        self.library.flashns_dgrad_shared_bytes.restype = ctypes.c_int
        self.shared_bytes = self.library.flashns_dgrad_shared_bytes()

    def __call__(self, d, weight, hidden, plan, *, fused=True):
        if (
            plan.dim not in (2, 3)
            or plan.order != 3
            or plan.indices != ref.JetPlan.dense(plan.dim, 3).indices
        ):
            raise ValueError("requires complete canonical 2-D/3-D order-3 jets")
        tensors = (d, weight, hidden)
        for value in tensors:
            if (
                value.dtype != torch.float64
                or not value.is_cuda
                or value.device != d.device
                or not value.is_contiguous()
            ):
                raise ValueError("requires contiguous FP64 tensors on one CUDA device")
            if torch.is_grad_enabled() and value.requires_grad:
                raise NotImplementedError(
                    "manual VJP only; double backward/HVP unsupported"
                )
        if (
            d.ndim != 3
            or d.shape[1] != plan.q
            or weight.ndim != 2
            or weight.shape[0] != d.shape[2]
            or hidden.shape != (d.shape[0], plan.q, weight.shape[1])
            or d.shape[2] <= 0
            or d.shape[0] > (2**31 - 1) // 20
            or weight.shape[1] > 64 * 65535
            or weight.shape[0] > 2**31 - 17
        ):
            raise ValueError("expected D[B,Q,Cout], W[Cout,Cin], H[B,Q,Cin]")
        output = torch.empty_like(hidden)
        with torch.cuda.device(d.device):
            stream = torch.cuda.current_stream()
            error = self.launch(
                plan.dim,
                fused,
                d.data_ptr(),
                weight.data_ptr(),
                hidden.data_ptr(),
                output.data_ptr(),
                d.shape[0],
                weight.shape[0],
                weight.shape[1],
                stream.cuda_stream,
            )
            for value in (*tensors, output):
                value.record_stream(stream)
        if error:
            raise RuntimeError(f"dgrad launch failed: cudaError_t={error}")
        return output


def backward(seed, checkpoints, weights, plan, dgrad_vjp):
    dws, dbs = [None] * len(weights), [None] * len(weights)
    d = seed
    for layer in reversed(range(len(weights))):
        # D is still materialized and read by wgrad before the next dgrad.
        dws[layer] = d.flatten(0, 1).T @ checkpoints[layer].flatten(0, 1)
        dbs[layer] = d[:, 0].sum(0)
        if layer:
            d = dgrad_vjp(d, weights[layer], checkpoints[layer], plan)
    return dws + dbs


def step(x, weights, biases, plan, native, dgrad_vjp):
    with torch.no_grad():
        jets, checkpoints = v1.network_forward(
            x, weights, biases, plan, lambda z: native(z, plan)
        )
    leaf = jets.detach().requires_grad_()
    loss = ref.residual_loss(leaf, plan)
    (seed,) = torch.autograd.grad(loss, leaf)
    with torch.no_grad():
        gradients = backward(seed, checkpoints, weights, plan, dgrad_vjp)
    return loss.detach(), gradients


def implementations(native, fused):
    return {
        "library_gemm_plus_native_vjp": lambda d, w, h, p: native(h, p, d @ w),
        "same_cutlass_mainloop_plus_native_vjp": lambda d, w, h, p: native(
            h, p, fused(d, w, h, p, fused=False)
        ),
        "fused_cutlass_dgrad_vjp": fused,
    }


def validate(native, fused):
    torch.manual_seed(7310)
    rows = []
    for dim in (2, 3):
        plan = ref.JetPlan.dense(dim, 3)
        assert tuple(plan.indices) == JetSpec(dim).coefficient_order
        for batch, cout, cin in [(1, 1, 1), (7, 65, 33), (13, 3, 65), (17, 128, 127)]:
            for case in ("random", "zero", "mixed_scale"):
                z = (
                    torch.randn(batch, plan.q, cin, device="cuda", dtype=torch.float64)
                    * 0.2
                )
                d = (
                    torch.randn(batch, plan.q, cout, device="cuda", dtype=torch.float64)
                    * 0.2
                )
                w = torch.randn(cout, cin, device="cuda", dtype=torch.float64) * 0.2
                if case == "zero":
                    z.zero_()
                    d.zero_()
                elif case == "mixed_scale":
                    d[:, 1::2] *= 1e-8
                    w[::2] *= 1e4
                    w[1::2] *= 1e-4
                z.requires_grad_()
                expected_h = ref.tanh_jet(z, plan)
                bar_h = d @ w
                (expected_d,) = torch.autograd.grad((expected_h * bar_h).sum(), z)
                with torch.no_grad():
                    hidden = native(z, plan)
                    candidate = fused(d, w, hidden, plan)
                    unpacked = fused(d, w, hidden, plan, fused=False)
                rows.append(
                    {
                        "dimension": dim,
                        "shape": [batch, cout, cin],
                        "case": case,
                        "gemm": compare(unpacked, bar_h),
                        "vjp_vs_independent_autograd": compare(candidate, expected_d),
                    }
                )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            d = torch.randn(23, plan.q, 65, device="cuda", dtype=torch.float64) * 0.2
            w = torch.randn(65, 35, device="cuda", dtype=torch.float64) * 0.2
            hidden = native(
                torch.randn(23, plan.q, 35, device="cuda", dtype=torch.float64) * 0.2,
                plan,
            )
            expected = native(hidden, plan, d @ w)
            outputs = [fused(d, w, hidden, plan) for _ in range(4)]
        stream.synchronize()
        rows.append(
            {
                "dimension": dim,
                "nondefault_stream_repeated": [compare(a, expected) for a in outputs],
            }
        )
        assert fused(d[:0], w, hidden[:0], plan).numel() == 0
        assert fused(d, w[:, :0], hidden[:, :, :0], plan).numel() == 0
        invalid = [
            (d.float(), w, hidden),
            (d, w.T, hidden),
            (d.transpose(0, 1), w, hidden),
            (d, w, hidden[:, :-1]),
        ]
        for values in invalid:
            try:
                fused(*values, plan)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid tensor accepted")
        try:
            fused(d.requires_grad_(), w, hidden, plan)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("differentiable custom VJP silently accepted")
    networks = []
    plan = ref.JetPlan.dense(2, 3)
    for scale in (0.01, 0.4, 3.0):
        widths = [2, 8, 7, 3]
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
            torch.randn(5, 2, device="cuda", dtype=torch.float64) * scale
        ).requires_grad_()
        output = ref.mlp(x, weights, biases)
        expected_jets = torch.stack(
            [
                torch.stack(
                    [
                        ref.nested_derivative(output[:, c], x, a)
                        / math.prod(math.factorial(v) for v in a)
                        for c in range(3)
                    ],
                    -1,
                )
                for a in plan.indices
            ],
            1,
        )
        with torch.no_grad():
            jets, _ = v1.network_forward(
                x, weights, biases, plan, lambda z: native(z, plan)
            )
        loss, gradients = step(x, weights, biases, plan, native, fused)
        expected_loss = ref.nested_loss(x, weights, biases)
        expected_gradients = torch.autograd.grad(expected_loss, weights + biases)
        networks.append(
            {
                "widths": widths,
                "batch": 5,
                "coordinate_scale": scale,
                "all_output_jets": compare(jets, expected_jets),
                "loss": compare(loss, expected_loss),
                "parameter_gradients": [
                    compare(a, b) for a, b in zip(gradients, expected_gradients)
                ],
            }
        )
    return {
        "cases": rows,
        "network_cases": networks,
        "invalid_inputs_and_unsupported_grad_rejected": True,
        "passed": True,
    }


def paired_measure(functions):
    """Rotate order over nine paired observations; report wall and event time."""
    names = list(functions)
    for _ in range(3):
        for fn in functions.values():
            _value = fn()
    torch.cuda.synchronize()
    pairs = []
    for index in range(9):
        order = names[index % len(names) :] + names[: index % len(names)]
        if index % 2:
            order = list(reversed(order))
        row = {"order": order}
        for name in order:
            torch.cuda.synchronize()
            _value = None
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start = perf_counter()
            begin.record()
            _value = functions[name]()
            end.record()
            end.synchronize()
            row[name] = {
                "wall_ms": (perf_counter() - start) * 1000,
                "event_ms": begin.elapsed_time(end),
                "additional_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                - resident,
            }
        pairs.append(row)
    baseline, ablation, candidate = names
    ratios = [r[baseline]["wall_ms"] / r[candidate]["wall_ms"] for r in pairs]
    fusion_ratios = [r[ablation]["wall_ms"] / r[candidate]["wall_ms"] for r in pairs]
    return {
        "pairs": pairs,
        "warmup_calls_per_variant": 3,
        "median_wall_ms": {
            n: statistics.median(r[n]["wall_ms"] for r in pairs) for n in names
        },
        "median_event_ms": {
            n: statistics.median(r[n]["event_ms"] for r in pairs) for n in names
        },
        "median_additional_peak_bytes": {
            n: statistics.median(r[n]["additional_peak_allocated_bytes"] for r in pairs)
            for n in names
        },
        "fused_vs_library_wall_speedup": {
            "median": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
        },
        "fused_vs_same_mainloop_wall_speedup": {
            "median": statistics.median(fusion_ratios),
            "min": min(fusion_ratios),
            "max": max(fusion_ratios),
        },
    }


def profile(fn):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        fn()
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
    kernel_events = [
        e for e in profiler.events() if str(e.device_type).endswith("CUDA")
    ]
    return {
        "top_events": sorted(events, key=lambda e: e["self_device_us"], reverse=True)[
            :18
        ],
        "cuda_event_count": len(kernel_events),
    }


def optimization_check(native, variants):
    """Fixed ten SGD updates check trajectory parity, not PDE convergence."""
    torch.manual_seed(7311)
    plan = ref.JetPlan.dense(2, 3)
    widths = [2, 32, 32, 3]
    weights = [
        torch.randn(o, i, device="cuda", dtype=torch.float64) * (0.7 / math.sqrt(i))
        for i, o in pairwise(widths)
    ]
    biases = [
        torch.randn(o, device="cuda", dtype=torch.float64) * 0.1 for o in widths[1:]
    ]
    x = torch.randn(128, 2, device="cuda", dtype=torch.float64) * 0.4
    states = {
        name: ([w.clone() for w in weights], [b.clone() for b in biases])
        for name in variants
    }
    losses = {name: [] for name in variants}
    rows = []
    baseline = next(iter(variants))
    for index in range(10):
        gradients = {}
        for name, implementation in variants.items():
            ws, bs = states[name]
            loss, gradients[name] = step(x, ws, bs, plan, native, implementation)
            losses[name].append(loss)
            with torch.no_grad():
                for parameter, gradient in zip(ws + bs, gradients[name]):
                    parameter.add_(gradient, alpha=-1e-3)
        rows.append(
            {
                "step": index,
                "loss": {
                    n: compare(losses[n][-1], losses[baseline][-1])
                    for n in variants
                    if n != baseline
                },
                "all_gradients": {
                    n: [
                        compare(a, b) for a, b in zip(gradients[n], gradients[baseline])
                    ]
                    for n in variants
                    if n != baseline
                },
            }
        )
    expected = states[baseline][0] + states[baseline][1]
    return {
        "steps": 10,
        "optimizer": "SGD",
        "learning_rate": 1e-3,
        "batch": 128,
        "widths": widths,
        "losses": {n: [float(v) for v in values] for n, values in losses.items()},
        "checks": rows,
        "final_parameters": {
            n: [compare(a, b) for a, b in zip(states[n][0] + states[n][1], expected)]
            for n in variants
            if n != baseline
        },
        "scope": "same fixed ten-update schedule; no scientific stopping criterion or convergence claim",
        "passed": True,
    }


def benchmark(native, fused):
    torch.manual_seed(7312)
    variants = implementations(native, fused)
    micro, network = [], []
    for dim in (2, 3):
        plan = ref.JetPlan.dense(dim, 3)
        shapes = [(b, c, c) for b in (32, 4096, 16384) for c in (32, 64, 128)]
        shapes += [(b, 3, 64) for b in (32, 4096, 16384)]
        for batch, cout, cin in shapes:
            d = (
                torch.randn(batch, plan.q, cout, device="cuda", dtype=torch.float64)
                * 0.2
            )
            w = torch.randn(cout, cin, device="cuda", dtype=torch.float64) * 0.2
            h = native(
                torch.randn(batch, plan.q, cin, device="cuda", dtype=torch.float64)
                * 0.2,
                plan,
            )
            functions = {
                name: partial(fn, d, w, h, plan) for name, fn in variants.items()
            }
            expected = next(iter(functions.values()))()
            row = {
                "dimension": dim,
                "batch": batch,
                "cout": cout,
                "cin": cin,
                "parity": {n: compare(f(), expected) for n, f in functions.items()},
                "eliminated_bar_h_payload_bytes": batch * plan.q * cin * 8,
                "eliminated_logical_write_and_read_bytes": 2 * batch * plan.q * cin * 8,
                "timing": paired_measure(functions),
            }
            micro.append(row)
            print(
                f"microbenchmark complete: dim={dim}, B={batch}, Cout={cout}, Cin={cin}",
                flush=True,
            )
    plan = ref.JetPlan.dense(2, 3)
    for batch, channels in [(4096, 32), (4096, 64), (16384, 64), (4096, 128)]:
        widths = [2, channels, channels, 3]
        weights = [
            torch.randn(o, i, device="cuda", dtype=torch.float64) * (0.7 / math.sqrt(i))
            for i, o in pairwise(widths)
        ]
        biases = [
            torch.randn(o, device="cuda", dtype=torch.float64) * 0.1 for o in widths[1:]
        ]
        x = torch.randn(batch, 2, device="cuda", dtype=torch.float64) * 0.4
        functions = {
            name: partial(step, x, weights, biases, plan, native, fn)
            for name, fn in variants.items()
        }
        expected_loss, expected_grad = next(iter(functions.values()))()
        comparisons = {}
        for name, fn in functions.items():
            loss, gradients = fn()
            comparisons[name] = {
                "loss": compare(loss, expected_loss),
                "gradients": [compare(a, b) for a, b in zip(gradients, expected_grad)],
            }
        row = {
            "batch": batch,
            "widths": widths,
            "parity": comparisons,
            "timing": paired_measure(functions),
        }
        # Profile separately, after unprofiled timings, to explain whole-step costs.
        row["profiles"] = {name: profile(fn) for name, fn in functions.items()}
        row["includes"] = (
            "coordinate seeding, unchanged native forward activations, library forward/wgrad, candidate dgrad/VJP, bias reductions, full steady NS residual plus residual-gradient loss and its output seed; no optimizer or independent CPU acceptance in timing"
        )
        network.append(row)
        print(f"full gradient benchmark complete: B={batch}, C={channels}", flush=True)
    return {
        "micro": micro,
        "network": network,
        "optimization_check": optimization_check(native, variants),
        "scope": "same FP64 full Taylor representation and H-only activation; reported logical traffic savings are not measured DRAM bytes; no scientific solve-to-tolerance claim",
    }


def build():
    target = HERE / "artifacts"
    target.mkdir(exist_ok=True)
    cutlass = ROOT / "third_party/cutlass"
    revision = subprocess.check_output(
        ["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != CUTLASS_COMMIT:
        raise RuntimeError(f"CUTLASS commit differs: {revision}")
    subprocess.run(
        ["git", "-C", str(cutlass), "diff", "--exit-code", "HEAD", "--", "include"],
        check=True,
        capture_output=True,
    )
    capability = torch.cuda.get_device_capability()
    library = target / "libdgrad_jet.so"
    command = [
        "nvcc",
        "-O3",
        "-std=c++17",
        f"-arch=sm_{capability[0]}{capability[1]}",
        "--expt-relaxed-constexpr",
        "-shared",
        "-Xcompiler=-fPIC",
        "-lineinfo",
        "--ptxas-options=-v",
        f"-I{cutlass / 'include'}",
        str(HERE / "dgrad_jet_vjp.cu"),
        "-o",
        str(library),
    ]
    start = perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    (target / "build.log").write_text(process.stdout + process.stderr)
    process.check_returncode()
    return library, {
        "command": command,
        "seconds": perf_counter() - start,
        "cutlass_commit": revision,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "fast_math": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=HERE / "artifacts/fused_a5000.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7310)
    library, build_info = build()
    native = v1.NativeJet(HERE.parent / "cuda_jet/artifacts/libflashns_jet.so")
    fused = FusedDgrad(library)
    source_paths = [
        Path(__file__),
        HERE / "dgrad_jet_vjp.cu",
        HERE / "sanitizer_smoke.cu",
        ROOT / "src/flashns/jet_spec.py",
        Path(v1.__file__),
        HERE.parent / "cuda_jet/input/jet_activation_kernels.cu",
        HERE.parent / "cuda_jet/input/jet_primitives.cuh",
        HERE.parent / "cuda_jet/input/jet_reference.py",
    ]
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": 7310,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "build": build_info,
        "shared_bytes_per_cta": fused.shared_bytes,
        "native_activation_library_sha256": hashlib.sha256(
            (HERE.parent / "cuda_jet/artifacts/libflashns_jet.so").read_bytes()
        ).hexdigest(),
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
            if p.is_file()
        },
        "jet_specs": [
            JetSpec(dim).manifest(
                generated_source_hash=hashlib.sha256(
                    (HERE.parent / "cuda_jet/input/jet_primitives.cuh").read_bytes()
                ).hexdigest()
            )
            for dim in (2, 3)
        ],
        "scope": "experimental full-Taylor ordinary-FP64 first parameter VJP; no automatic dispatch, C13, tail-robust activation, HVP or scientific convergence claim",
    }
    try:
        report["validation"] = validate(native, fused)
        print(
            "Fused GPU parity, stream and full parameter-gradient validation passed",
            flush=True,
        )
        report["saturation"] = v1.saturation_diagnostics(native)
        if args.benchmark:
            report["benchmark"] = benchmark(native, fused)
    except Exception as exc:  # noqa: BLE001 -- retain evidence of experimental failure
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print({"output": str(args.output), "failure": report.get("failure")}, flush=True)
    return 1 if "failure" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
