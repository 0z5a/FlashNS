"""Target-device validation, with tail criteria fixed before performance runs."""

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import mpmath as mp
import torch
from common import (
    comparison,
    local_case,
    make_case,
    nested_reference,
    source_hashes,
    step,
)
from gpu import BACKENDS, Native

from flashns.jet_spec import JetSpec
from flashns.jet_stable import tanh_jet


def quantized_tail_comparison(actual, expected):
    """Normal tails: relative 1e-11; subnormals: additionally eight binary64 ulps."""
    expected_float = float(expected)
    absolute = abs(actual - expected_float)
    tolerance = 1e-11 * abs(expected_float) + 8 * math.ulp(expected_float)
    passed = math.isfinite(actual) and absolute <= tolerance
    result = {
        "actual": actual,
        "expected_100_digits": str(expected),
        "expected_rounded_fp64": expected_float,
        "absolute_error_to_rounded": absolute,
        "tolerance": tolerance,
        "subnormal_or_underflow": abs(expected_float) < 2.0**-1022,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(result)
    return result


def tail_diagnostics(native):
    rows = []
    centers = (
        0,
        5,
        10,
        15,
        18,
        19,
        20,
        30,
        100,
        300,
        350,
        360,
        370,
        373,
        374,
        500,
        750,
        -20,
        -100,
        -350,
        -370,
        -374,
    )
    with mp.workdps(100):
        for dim in (2, 3):
            spec = JetSpec(dim)
            lookup = {a: i for i, a in enumerate(spec.coefficient_order)}
            pure = [lookup[(n,) + (0,) * (dim - 1)] for n in range(4)]
            for center in centers:
                z = torch.zeros(1, spec.q, 1, device="cuda", dtype=torch.float64)
                z[:, 0] = center
                z[:, pure[1]] = 1
                with torch.no_grad():
                    h, aux = native.forward(z, dim)
                    bar = torch.zeros_like(h)
                    bar[:, pure[3]] = 1
                    vjp = native.vjp(h, aux, bar, dim)
                derivatives = {
                    order: mp.diff(mp.tanh, mp.mpf(center), order, addprec=3000)
                    for order in range(1, 5)
                }
                row = {
                    "dimension": dim,
                    "z0": center,
                    "forward": [
                        quantized_tail_comparison(
                            float(h[0, pure[n], 0]), derivatives[n] / math.factorial(n)
                        )
                        for n in range(1, 4)
                    ],
                    "aux": quantized_tail_comparison(float(aux[0, 0]), derivatives[1]),
                    "vjp": [
                        quantized_tail_comparison(
                            float(vjp[0, pure[n], 0]),
                            derivatives[4 - n] / math.factorial(3 - n),
                        )
                        for n in range(4)
                    ],
                }
                rows.append(row)
    return {
        "cases": rows,
        "criteria": "relative 1e-11 plus eight ulps of the correctly rounded reference; subnormal/underflow acceptance is absolute quantization accuracy, not relative preservation",
        "passed": True,
    }


def tail_tensor_comparison(actual, expected):
    a = actual.detach().cpu().flatten().tolist()
    b = expected.detach().cpu().flatten().tolist()
    ratios = [abs(x - y) / (1e-10 * abs(y) + 8 * math.ulp(y)) for x, y in zip(a, b)]
    maximum = max(ratios)
    if not math.isfinite(maximum) or maximum > 1:
        raise AssertionError({"tail_network_max_tolerance_ratio": maximum})
    return {
        "max_tolerance_ratio": maximum,
        "relative_tolerance": 1e-10,
        "ulp_allowance": 8,
        "passed": True,
    }


def validate(native):
    torch.manual_seed(7511)
    local = []
    for dim in (2, 3):
        q = JetSpec(dim).q
        for batch, cin, cout in [(1, 1, 1), (7, 33, 65), (13, 65, 3), (17, 32, 32)]:
            for shift in (0.0, 20.0, -20.0):
                z = torch.randn(batch, q, cin, device="cuda", dtype=torch.float64) * 0.2
                z[:, 0] += shift
                z.requires_grad_()
                d = (
                    torch.randn(batch, q, cout, device="cuda", dtype=torch.float64)
                    * 0.2
                )
                w = torch.randn(cout, cin, device="cuda", dtype=torch.float64) * 0.2
                expected_h, expected_aux = tanh_jet(z, dim)
                (expected_vjp,) = torch.autograd.grad((expected_h * (d @ w)).sum(), z)
                with torch.no_grad():
                    h, aux = native.forward(z, dim)
                    results = {
                        backend: comparison(
                            native.dgrad(d, w, h, aux, dim, backend), expected_vjp
                        )
                        for backend in BACKENDS
                    }
                local.append(
                    {
                        "dimension": dim,
                        "shape": [batch, cin, cout],
                        "shift": shift,
                        "forward": comparison(h, expected_h),
                        "aux": comparison(aux, expected_aux),
                        "backend_vjp": results,
                    }
                )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            z = torch.randn(23, q, 35, device="cuda", dtype=torch.float64) * 0.2
            d = torch.randn(23, q, 3, device="cuda", dtype=torch.float64) * 0.2
            w = torch.randn(3, 35, device="cuda", dtype=torch.float64) * 0.2
            h, aux = native.forward(z, dim)
            expected = native.dgrad(d, w, h, aux, dim)
            outputs = {
                backend: native.dgrad(d, w, h, aux, dim, backend)
                for backend in BACKENDS
            }
        stream.synchronize()
        local.append(
            {
                "dimension": dim,
                "nondefault_stream": {
                    backend: comparison(value, expected)
                    for backend, value in outputs.items()
                },
            }
        )
        with torch.no_grad():
            for backend in BACKENDS:
                assert native.dgrad(d[:0], w, h[:0], aux[:0], dim, backend).numel() == 0
        for bad in [z.float(), z.transpose(0, 2), z[:, :-1]]:
            try:
                native.forward(bad, dim)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid input accepted")
        try:
            native.forward(z.requires_grad_(), dim)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("unsupported differentiable native call accepted")
    networks = []
    for scale in (0.01, 0.4, 3.0):
        x, quadrature, weights, biases, meta = local_case(
            make_case(5, 7, coordinate_scale=scale, weighted=True), 1, 0, "cuda"
        )
        loss_ref, grad_ref = nested_reference(
            x, quadrature, meta["global_denominator"], weights, biases
        )
        rows = {}
        for backend in BACKENDS:
            loss, gradients = step(
                x,
                quadrature,
                meta["global_denominator"],
                weights,
                biases,
                native,
                backend,
            )
            rows[backend] = {
                "loss": comparison(loss, loss_ref),
                "all_parameter_gradients": [
                    comparison(a, b) for a, b in zip(gradients, grad_ref)
                ],
            }
        networks.append({"case": meta, "backends": rows})
    tail_networks = []
    for shift in (20.0, -20.0):
        generator = torch.Generator().manual_seed(7513)
        x = (torch.randn(3, 2, generator=generator, dtype=torch.float64) * 0.2).cuda()
        weights = [
            (torch.randn(2, 2, generator=generator, dtype=torch.float64) * 0.2).cuda(),
            (torch.randn(3, 2, generator=generator, dtype=torch.float64) * 0.2).cuda(),
        ]
        biases = [
            torch.full((2,), shift, device="cuda", dtype=torch.float64),
            torch.zeros(3, device="cuda", dtype=torch.float64),
        ]
        q = torch.tensor([0.5, 1.0, 1.5], device="cuda", dtype=torch.float64)
        loss_ref, grad_ref = nested_reference(x, q, 3.0, weights, biases)
        rows = {}
        for backend in BACKENDS:
            loss, gradients = step(x, q, 3.0, weights, biases, native, backend)
            rows[backend] = {
                "loss": tail_tensor_comparison(loss, loss_ref),
                "all_parameter_gradients": [
                    tail_tensor_comparison(a, b) for a, b in zip(gradients, grad_ref)
                ],
            }
        tail_networks.append(
            {
                "shift": shift,
                "widths": [2, 2, 3],
                "weighted_batch": 3,
                "reference": "independent nested AD with scalar stable derivative, separately checked to 100 digits",
                "loss": float(loss_ref),
                "backends": rows,
            }
        )
    return {
        "local": local,
        "networks": networks,
        "tail_networks": tail_networks,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.set_num_threads(2)
    native = Native()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(),
        "source_hashes": source_hashes(),
        "build": native.build,
        "scope": "finite full-jet stable-a1 local and first-parameter-VJP validation; no HVP or scientific convergence",
    }
    try:
        report["high_precision_tails"] = tail_diagnostics(native)
        report["validation"] = validate(native)
    except Exception as error:  # noqa: BLE001 -- preserve experimental failures
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print({"output": str(args.output), "failure": report.get("failure")}, flush=True)
    return 1 if "failure" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
