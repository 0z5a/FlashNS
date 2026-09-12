"""Target-GPU correctness, non-default stream, Graph and actual Adam updates."""

import argparse
import json
import math
import traceback
from pathlib import Path

import mpmath as mp
import numpy as np
import torch

from runtime import (Adam, Engine, NativeCUDA, Problem, device_metadata, frozen_protocol,
                     parameters, snapshot, views)
from backends import nested_step
from flashns.jet_packed import PackedLayout, TensorActivation
from flashns.ns_seed import NSLoss, explicit_seed


def compare(actual, expected, atol=2e-12, rtol=2e-11):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(expected, torch.Tensor):
        expected = expected.detach().cpu().numpy()
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("shape or finite-value mismatch")
    ratio = float(np.max(np.abs(actual-expected)/(atol+rtol*np.abs(expected)), initial=0))
    if ratio > 1:
        raise AssertionError(f"max error/tolerance={ratio}")
    return ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="kernel/Graph smoke for sanitizer runs")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite preflight evidence")
    if not torch.cuda.is_available():
        raise RuntimeError("target CUDA GPU required; GPU validation was not run")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(83109)
    native = NativeCUDA(args.build)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "completed": False, "quick": args.quick,
              "source_hashes": snapshot(args.output), "binary_sha256": native.metadata["library_sha256"],
              "device": device_metadata(), "checks": []}

    def check(name, function):
        try:
            details = function()
            report["checks"].append({"name": name, "passed": True, "detail": details})
        except Exception as error:
            report["checks"].append({"name": name, "passed": False, "error": str(error), "traceback": traceback.format_exc()})
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(name, report["checks"][-1]["passed"], flush=True)

    def activations():
        worst = 0.0
        cases = 0
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for dim in (2, 3):
                for ni, nb in ((0, 0), (0, 3), (1, 0), (17, 5), (129, 31)):
                    for channels in ((7, 64) if args.quick else (7, 32, 64, 128)):
                        layout = PackedLayout(ni, nb, dim)
                        z = torch.randn(layout.rows, channels, device="cuda", dtype=torch.float64)*0.3
                        seed = torch.randn_like(z)
                        expected = TensorActivation(layout)
                        observed = native.activation_factory(layout)
                        h, a = observed.forward(z)
                        hr, ar = expected.forward(z)
                        dz = observed.vjp(h, a, seed)
                        dr = expected.vjp(hr, ar, seed)
                        for actual, target in ((h, hr), (a, ar), (dz, dr)):
                            worst = max(worst, compare(actual, target))
                        # Device canaries around an exact-sized input view.
                        holder = torch.full((z.numel()+2,), 917.0, dtype=z.dtype, device=z.device)
                        guarded = holder[1:-1].view_as(z)
                        guarded.copy_(z)
                        gh, ga = observed.forward(guarded)
                        worst = max(worst, compare(gh, hr), compare(ga, ar))
                        assert holder[0].item() == 917.0 and holder[-1].item() == 917.0
                        cases += 1
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        return {"cases": cases, "max_error_over_tolerance": worst, "non_default_stream": True}

    check("packed_full_and_value_activation_vjp", activations)

    def scalar_tails():
        mp.mp.dps = 420
        probes = [0, 15, 20, 21, 100, 300, 355, 360, 370, 372, 373, 374, 380]
        z = torch.tensor([[sign*p] for p in probes for sign in (1, -1)], device="cuda", dtype=torch.float64)
        h, a = native.activation_factory(PackedLayout(0, len(z))).forward(z)
        tiny = mp.mpf(float(np.nextafter(0.0, 1.0)))
        rows = []
        for value, aux in zip(z.cpu().numpy().reshape(-1), a.cpu().numpy().reshape(-1)):
            r = mp.exp(-abs(mp.mpf(float(value))))
            exact = (2*r/(1+r*r))**2
            error = abs(mp.mpf(float(aux))-exact)
            normal = exact >= mp.mpf(np.finfo(np.float64).tiny)
            if normal:
                assert float(error/exact) < 5e-13
            else:
                assert error <= 2*tiny
            rows.append({"z": float(value), "aux": float(aux), "normal": bool(normal)})
        assert bool(torch.isfinite(h).all())
        return rows

    check("libdevice_scalar_aux_tail_contract", scalar_tails)

    def residual_seed():
        worst = 0.0
        cases = 0
        for ni, nb in ((0, 0), (0, 5), (1, 0), (17, 3), (129, 17)):
            j = torch.randn(ni, 10, 3, dtype=torch.float64, device="cuda")*0.4
            # Explicitly test a strided boundary view, as supported by the ABI.
            b = torch.randn(nb, 10, 3, dtype=j.dtype, device=j.device)[:, 0]
            w = torch.rand(ni, dtype=j.dtype, device=j.device)/max(ni, 1)
            bw = torch.rand(nb, 3, dtype=j.dtype, device=j.device)/max(nb, 1)
            target = torch.randn(nb, 3, dtype=j.dtype, device=j.device)*0.2
            if ni:
                w[0] = 0
            loss = NSLoss()
            total, di, db = native.seed(j, b, w, bw, target, loss)
            lp, expected = explicit_seed(j, w)
            expected_loss = lp + 5*(bw*(b-target).square()).sum()
            for actual, other in ((total, expected_loss), (di, expected), (db, 10*bw*(b-target))):
                worst = max(worst, compare(actual, other))
            cases += 1
        return {"cases": cases, "max_error_over_tolerance": worst}

    check("explicit_cuda_residual_seed", residual_seed)

    def graph_updates():
        problem = Problem(interior_count=17 if args.quick else 2048,
                          edge_count=3 if args.quick else 128, validation_count=64)
        initial = parameters(361901)
        rows = []
        for layout in ("dense", "split", "compact"):
            for seed in (("autograd", "cuda") if args.quick else ("autograd", "explicit", "compiled", "cuda")):
                engine = Engine(problem, initial, layout=layout, seed=seed, activation="cuda", native=native)
                optimizer = Adam(engine.flat, engine.gradient)
                reference_parameters = initial.numpy().copy()
                moment, variance = np.zeros_like(reference_parameters), np.zeros_like(reference_parameters)
                worst = 0.0
                for update in range(1, 4):
                    value, gradient = engine.replay()
                    reference_flat = torch.tensor(reference_parameters, dtype=torch.float64, device="cuda")
                    expected_value, expected_gradient = nested_step(problem, reference_flat)
                    worst = max(worst, compare(value, expected_value), compare(gradient, expected_gradient))
                    g = expected_gradient.cpu().numpy()
                    moment = 0.9*moment + 0.1*g
                    variance = 0.999*variance + 0.001*g*g
                    reference_parameters -= 0.001*(moment/(1-0.9**update))/(np.sqrt(variance/(1-0.999**update))+1e-8)
                    optimizer.step()
                    torch.cuda.synchronize()
                    worst = max(worst, compare(engine.flat, reference_parameters))
                rows.append({"layout": layout, "seed": seed, "adam_updates": 3,
                             "max_error_over_tolerance": worst, "metadata": engine.metadata})
                del optimizer, engine
        return rows

    check("all_layouts_actual_graph_adam_updates_vs_nested_ad_and_numpy", graph_updates)
    torch.cuda.synchronize()
    report["completed"] = True
    report["passed"] = all(row["passed"] for row in report["checks"])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"])}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
