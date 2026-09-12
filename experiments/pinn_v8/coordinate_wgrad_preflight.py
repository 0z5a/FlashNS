"""G1 coordinate parameter contraction GPU correctness; no timing.

Complete materialized activation VJP is retained. --kernel-only runs bounded
sanitizer checks and cannot authorize timing or full-model correctness claims.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from runtime import (Adam, NativeCUDA, Problem, compare, device_metadata, digest,
                     parameters, snapshot, source_hashes, views)
from backends import nested_step
from flashns.jet_packed import PackedLayout, TensorActivation
from flashns.pinn_packed import PackedPINN
from coordinate_sm120 import guarded, check_canary
from tail_sm120 import device_uuid, sha256


def dense_contraction(layout, x, derivative):
    return derivative.T @ layout.coordinates(x), derivative[layout.zero_rows(x.device)].sum(0)


def compare_pair(actual, expected):
    return max(compare(a, e) for a, e in zip(actual, expected))


def compare_patterns(actual, expected):
    worst = 0.
    for observed, target in zip(actual, expected):
        for predicate in (torch.isnan, torch.isposinf, torch.isneginf):
            assert torch.equal(predicate(observed), predicate(target)), "nonfinite propagation differs"
        finite = torch.isfinite(target)
        worst = max(worst, compare(observed[finite], target[finite]))
    return worst


def kernel_preflight(native, quick=False):
    rows = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for dimension in (2, 3):
            for ni, nb in (((0, 0), (0, 3), (17, 5)) if quick else
                           ((0, 0), (0, 3), (1, 0), (17, 5), (129, 31))):
                for channels in ((7, 64) if quick else (7, 32, 64, 128)):
                    layout = PackedLayout(ni, nb, dimension)
                    originals = (torch.randn(layout.points, dimension, dtype=torch.float64, device="cuda")*0.3,
                                 torch.randn(layout.rows, channels, dtype=torch.float64, device="cuda")*0.3)
                    (x, derivative), holders = zip(*(guarded(t) for t in originals))
                    expected = dense_contraction(layout, x, derivative)
                    actual = native.coordinate_wgrad(layout, x, derivative)
                    partials, ph = guarded(derivative.new_full(((layout.points+7)//8, channels, dimension+1), float("nan")))
                    dw, wh = guarded(torch.full_like(expected[0], float("nan")))
                    db, bh = guarded(torch.full_like(expected[1], float("nan")))
                    native.launch(native.coordinate_wgrad_launch,
                                  [dimension, derivative.data_ptr(), x.data_ptr(), partials.data_ptr(),
                                   dw.data_ptr(), db.data_ptr(), ni, nb, channels], (x, derivative, partials, dw, db))
                    worst = max(compare_pair(actual, expected), compare_pair((dw, db), expected))
                    assert bool(torch.isfinite(partials).all()), "partial scratch was not fully written"
                    for view, original, holder in zip((x, derivative), originals, holders):
                        assert torch.equal(view, original), "coordinate contraction modified its input"
                        check_canary(holder)
                    for holder in (ph, wh, bh): check_canary(holder)
                    rows.append({"q": layout.q, "full_points": ni, "value_points": nb, "channels": channels,
                                 "max_error_over_tolerance": worst, "partial_tiles": (layout.points+7)//8})
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return {"rows": rows, "non_default_stream": True, "offset_inputs_outputs_and_partials": True,
            "outputs_and_partials_prefilled_with_nan": True, "reference": "dense physical-coordinate jet GEMM + bias sum"}


def nonfinite_and_saturation_preflight(native, quick=False):
    rows = []
    for dimension in (2, 3):
        layout = PackedLayout(1, 1, dimension)
        x = torch.ones(layout.points, dimension, dtype=torch.float64, device="cuda")
        slots = sorted({0, 1, dimension, layout.q-1, layout.q}) if quick else range(layout.rows)
        for slot in slots:
            for value in (float("nan"), float("inf"), -float("inf")):
                derivative = torch.zeros(layout.rows, 3, dtype=x.dtype, device=x.device)
                derivative[slot, 1] = value
                worst = compare_patterns(native.coordinate_wgrad(layout, x, derivative), dense_contraction(layout, x, derivative))
                rows.append({"q": layout.q, "slot": slot, "value": str(value), "matching_nonfinite_masks": True,
                             "finite_max_error_over_tolerance": worst})
        # Complete stable VJP remains the producer. High-order seeds and rounded
        # tanh centers must still contribute to the low-order bar_Z consumers.
        layout = PackedLayout(3, 2, dimension)
        x = torch.randn(layout.points, dimension, dtype=x.dtype, device=x.device)*0.2
        z = torch.randn(layout.rows, 2, dtype=x.dtype, device=x.device)*0.1
        z[layout.zero_rows(x.device)] = torch.tensor([20., -20.], dtype=x.dtype, device=x.device)
        seed = torch.randn_like(z)
        ops, tensor_ops = native.activation_factory(layout), TensorActivation(layout)
        h, a1 = ops.forward(z)
        assert bool((a1 > 0).all()) and bool((h[layout.zero_rows(x.device)].abs() == 1).all())
        derivative = ops.vjp(h, a1, seed)
        reference_h, reference_aux = tensor_ops.forward(z)
        expected_derivative = tensor_ops.vjp(reference_h, reference_aux, seed)
        worst = max(compare(derivative, expected_derivative),
                    compare_pair(native.coordinate_wgrad(layout, x, derivative), dense_contraction(layout, x, expected_derivative)))
        rows.append({"q": layout.q, "saturated_centers": [20., -20.], "positive_stable_a1": True,
                     "complete_VJP_producer": True, "max_error_over_tolerance": worst})
    return rows


def kernel_graph_updates(native, quick=False):
    rows = []
    for dimension in (2, 3):
        for channels in ((7,) if quick else (7, 32, 64, 128)):
            layout = PackedLayout(17, 5, dimension)
            x = torch.randn(layout.points, dimension, dtype=torch.float64, device="cuda")*0.3
            derivative = torch.randn(layout.rows, channels, dtype=x.dtype, device=x.device)*0.3
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): native.coordinate_wgrad(layout, x, derivative)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                dw, db = native.coordinate_wgrad(layout, x, derivative)
            worst, previous = 0., None
            for _ in range(3):
                x.normal_(0., 0.3)
                derivative.normal_(0., 0.3)
                graph.replay()
                worst = max(worst, compare_pair((dw, db), dense_contraction(layout, x, derivative)))
                current = dw.clone()
                if previous is not None: assert not torch.equal(previous, current), "Graph used stale inputs"
                previous = current
            rows.append({"q": layout.q, "channels": channels, "coordinate_and_bar_Z_updates": 3,
                         "max_error_over_tolerance": worst})
    torch.cuda.synchronize()
    return rows


class CoordinateWgradModuleGraph:
    def __init__(self, problem, flat, native, mode, layout):
        self.flat = flat
        ni = problem.interior_count
        self.step = PackedPINN(
            problem.x, ni, problem.pde_weights[:ni], problem.boundary_weights[ni:], problem.target[ni:],
            layout=layout, seed="cuda", activation_factory=native.activation_factory, cuda_seed=native.seed,
            coordinate_wgrad=native.coordinate_wgrad if mode == "coordinate" else None)
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(4): self.eager()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss, self.gradient = self.eager()
        self.metadata = dict(self.step.metadata, wgrad_mode=mode, graph_pool=list(self.graph.pool()),
                             parameter_data_ptr=flat.data_ptr())

    def eager(self):
        loss, gradients = self.step(*views(self.flat))
        return loss, torch.cat([g.reshape(-1) for g in gradients])

    def replay(self):
        self.graph.replay()
        return self.loss, self.gradient


def module_preflight(native):
    problem = Problem(interior_count=2048, edge_count=128, validation_count=64)
    initial = parameters(431901)
    reference = initial.numpy().copy()
    moment, variance = np.zeros_like(reference), np.zeros_like(reference)
    trajectory = []
    for update in range(1, 4):
        loss, gradient = nested_step(problem, torch.tensor(reference, dtype=torch.float64, device="cuda"))
        loss, gradient = loss.cpu().numpy(), gradient.cpu().numpy()
        moment = 0.9*moment+0.1*gradient
        variance = 0.999*variance+0.001*gradient*gradient
        reference -= 0.001*(moment/(1-0.9**update))/(np.sqrt(variance/(1-0.999**update))+1e-8)
        trajectory.append((loss, gradient, reference.copy()))
    rows = []
    for layout in ("dense", "split", "compact"):
        for mode in ("gemm", "coordinate"):
            engine = CoordinateWgradModuleGraph(problem, initial.to("cuda").clone(), native, mode, layout)
            optimizer = Adam(engine.flat, engine.gradient)
            worst = 0.
            for expected_loss, expected_gradient, expected_parameters in trajectory:
                loss, gradient = engine.replay()
                worst = max(worst, compare(loss, expected_loss), compare(gradient, expected_gradient))
                optimizer.step()
                worst = max(worst, compare(engine.flat, expected_parameters))
            assert digest(engine.flat) != digest(initial), "Adam did not update parameter storage"
            rows.append({"layout": layout, "wgrad_mode": mode, "adam_updates": 3,
                         "max_error_over_tolerance": worst, "metadata": engine.metadata})
            del optimizer, engine
    return {"problem": problem.meta, "rows": rows, "reference": "independent nested AD + NumPy Adam"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-only", action="store_true", help="bounded Sanitizer checks; no performance authorization")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output: output.write('{"completed": false, "preflight_passed": false}\n')
    report = {"completed": False, "preflight_passed": False, "checks": [],
              "contract": {"candidate": "coordinate_wgrad_G1", "kernel_only": args.kernel_only,
                           "matched_control": "unchanged full VJP, library coordinate GEMM and bias sum",
                           "precision": "FP64 full Q10/Q20 bar_Z; structural zero products retained; fmad=false",
                           "point_tile": 8, "merge": "sequential deterministic; no atomics; not GEMM bitwise order",
                           "atol": 2e-12, "rtol": 2e-11, "tail_mode": "torch", "timing": "none",
                           "solver_convergence_validated": False}}

    def save(): args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("this preflight requires an allocated SM120 CUDA GPU")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(86111)
        native = NativeCUDA(args.build)
        library = args.build/native.metadata["library"]
        report.update(device=dict(device_metadata(), uuid=device_uuid()), source_hashes=snapshot(args.output),
                      build=native.metadata, binary_sha256=sha256(library), binary_path=str(library.resolve()),
                      build_metadata_sha256=sha256(args.build/"build.json"),
                      process={"pid": os.getpid(), "python": sys.executable, "argv": sys.argv})
        checks = [("Q10_Q20_full_value_empty_current_stream_and_canaries", lambda: kernel_preflight(native, args.kernel_only)),
                  ("axis_specific_nonfinite_and_saturated_complete_VJP", lambda: nonfinite_and_saturation_preflight(native, args.kernel_only)),
                  ("graph_coordinate_and_bar_Z_updates", lambda: kernel_graph_updates(native, args.kernel_only))]
        if not args.kernel_only:
            checks.append(("all_layouts_three_actual_adam_updates_vs_nested_AD", lambda: module_preflight(native)))
        for name, check in checks:
            report["active_check"] = name
            save()
            detail = check()
            report["checks"].append({"name": name, "passed": True, "detail": detail})
            save()
            print(name, "passed", flush=True)
        if (source_hashes() != report["source_hashes"] or sha256(library) != report["binary_sha256"]
                or sha256(args.build/"build.json") != report["build_metadata_sha256"]):
            raise RuntimeError("source or build changed during preflight")
        del report["active_check"]
        report.update(completed=True, preflight_passed=True)
        save()
    except Exception as error:
        report["failure"] = {"error": str(error), "traceback": traceback.format_exc()}
        save()
        raise


if __name__ == "__main__":
    main()
