"""G2 full stable activation VJP + coordinate parameter contraction preflight."""

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
from coordinate_wgrad_preflight import compare_pair, compare_patterns, dense_contraction
from tail_sm120 import device_uuid, sha256


def materialized(native, layout, x, hidden, aux, seed):
    derivative = native.activation_factory(layout).vjp(hidden, aux, seed)
    return native.coordinate_wgrad(layout, x, derivative)


def exact_matched(actual, expected):
    worst = compare_patterns(actual, expected)
    for observed, target in zip(actual, expected):
        finite = torch.isfinite(target)
        assert torch.equal(observed.view(torch.int64)[finite], target.view(torch.int64)[finite]), "G1/G2 finite bits differ"
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
                    z = torch.randn(layout.rows, channels, dtype=torch.float64, device="cuda")*0.3
                    h, aux = native.activation_factory(layout).forward(z)
                    originals = (torch.randn(layout.points, dimension, dtype=z.dtype, device=z.device)*0.3,
                                 h, aux, torch.randn_like(h)*0.3)
                    inputs, holders = zip(*(guarded(t) for t in originals))
                    x, h, aux, seed = inputs
                    expected = materialized(native, layout, *inputs)
                    actual = native.coordinate_activation_wgrad(layout, *inputs)
                    partials, ph = guarded(h.new_full(((layout.points+7)//8, channels, dimension+1), float("nan")))
                    dw, wh = guarded(torch.full_like(expected[0], float("nan")))
                    db, bh = guarded(torch.full_like(expected[1], float("nan")))
                    native.launch(native.coordinate_activation_wgrad_launch,
                                  [dimension, h.data_ptr(), aux.data_ptr(), seed.data_ptr(), x.data_ptr(),
                                   partials.data_ptr(), dw.data_ptr(), db.data_ptr(), ni, nb, channels],
                                  (*inputs, partials, dw, db))
                    worst = max(exact_matched(actual, expected), exact_matched((dw, db), expected))
                    if not quick:
                        tensor_ops = TensorActivation(layout)
                        tensor_h, tensor_aux = tensor_ops.forward(z)
                        independent = dense_contraction(layout, x, tensor_ops.vjp(tensor_h, tensor_aux, seed))
                        worst = max(worst, compare_pair(actual, independent))
                    assert bool(torch.isfinite(partials).all()), "G2 partials were not fully written"
                    for view, original, holder in zip(inputs, originals, holders):
                        assert torch.equal(view, original), "G2 modified its input"
                        check_canary(holder)
                    for holder in (ph, wh, bh): check_canary(holder)
                    rows.append({"q": layout.q, "full_points": ni, "value_points": nb, "channels": channels,
                                 "G1_G2_finite_bitwise_match": True, "max_error_over_tolerance": worst,
                                 "independent_tensor_VJP_and_dense_wgrad": not quick})
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return {"rows": rows, "non_default_stream": True, "offset_inputs_outputs_and_partials": True,
            "outputs_and_partials_prefilled_with_nan": True}


def nonfinite_and_saturation_preflight(native, quick=False):
    rows = []
    for dimension in (2, 3):
        layout = PackedLayout(1, 1, dimension)
        x = torch.ones(layout.points, dimension, dtype=torch.float64, device="cuda")
        base = torch.full((layout.rows, 2), 0.1, dtype=x.dtype, device=x.device)
        slots = sorted({0, 1, dimension, layout.q-1, layout.q}) if quick else range(layout.rows)
        for location in ("hidden", "seed"):
            for slot in slots:
                for value in (float("nan"), float("inf"), -float("inf")):
                    h, seed = base.clone(), base.clone()
                    aux = torch.ones(layout.points, 2, dtype=x.dtype, device=x.device)
                    (h if location == "hidden" else seed)[slot, 1] = value
                    worst = exact_matched(native.coordinate_activation_wgrad(layout, x, h, aux, seed),
                                          materialized(native, layout, x, h, aux, seed))
                    rows.append({"q": layout.q, "location": location, "slot": slot, "value": str(value),
                                 "matching_nonfinite_masks": True, "finite_max_error_over_tolerance": worst})
        for value in (float("nan"), float("inf"), -float("inf")):
            aux = torch.ones(layout.points, 2, dtype=x.dtype, device=x.device)
            aux[0, 1] = value
            exact_matched(native.coordinate_activation_wgrad(layout, x, base, aux, base),
                          materialized(native, layout, x, base, aux, base))
        h, seed = torch.ones_like(base), torch.full_like(base, 1e308)
        aux = torch.ones(layout.points, 2, dtype=x.dtype, device=x.device)
        expected = materialized(native, layout, x, h, aux, seed)
        exact_matched(native.coordinate_activation_wgrad(layout, x, h, aux, seed), expected)
        assert any(bool((~torch.isfinite(t)).any()) for t in expected), "overflow probe did not overflow"
        rows.append({"q": layout.q, "all_finite_inputs_with_internal_VJP_overflow": True,
                     "matching_nonfinite_masks_and_finite_bits": True, "aux_nonfinite_cases": 3})
        for center in ((20., -20.) if quick else (20., -20., 100., -100., 350., -350.)):
            z = torch.full_like(base, 0.1)
            z[layout.zero_rows(x.device)] = center
            h, aux = native.activation_factory(layout).forward(z)
            seed = torch.zeros_like(h)
            seed[layout.zero_rows(x.device)] = 1.
            actual = native.coordinate_activation_wgrad(layout, x, h, aux, seed)
            exact_matched(actual, materialized(native, layout, x, h, aux, seed))
            assert bool((aux>0).all()) and bool((actual[1]>0).all()), "saturated derivative tail vanished"
            rows.append({"q": layout.q, "saturated_center": center, "positive_a1_and_bias_gradient": True})
    return rows


def kernel_graph_updates(native, quick=False):
    rows = []
    for dimension in (2, 3):
        for channels in ((7,) if quick else (7, 32, 64, 128)):
            layout = PackedLayout(17, 5, dimension)
            x = torch.randn(layout.points, dimension, dtype=torch.float64, device="cuda")*0.3
            z = torch.randn(layout.rows, channels, dtype=x.dtype, device=x.device)*0.3
            ops = native.activation_factory(layout)
            h, aux = ops.forward(z)
            seed = torch.randn_like(h)*0.3
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): native.coordinate_activation_wgrad(layout, x, h, aux, seed)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = native.coordinate_activation_wgrad(layout, x, h, aux, seed)
            previous = None
            for _ in range(3):
                x.normal_(0., 0.3)
                z.normal_(0., 0.3)
                seed.normal_(0., 0.3)
                updated_h, updated_aux = ops.forward(z)
                h.copy_(updated_h)
                aux.copy_(updated_aux)
                graph.replay()
                exact_matched(actual, materialized(native, layout, x, h, aux, seed))
                current = actual[0].clone()
                if previous is not None: assert not torch.equal(previous, current), "Graph used stale inputs"
                previous = current
            rows.append({"q": layout.q, "channels": channels, "x_H_a1_bar_H_updates": 3,
                         "G1_G2_finite_bitwise_match": True})
    torch.cuda.synchronize()
    return rows


class FirstVjpWgradModuleGraph:
    def __init__(self, problem, flat, native, mode, layout):
        self.flat = flat
        ni = problem.interior_count
        self.step = PackedPINN(
            problem.x, ni, problem.pde_weights[:ni], problem.boundary_weights[ni:], problem.target[ni:],
            layout=layout, seed="cuda", activation_factory=native.activation_factory, cuda_seed=native.seed,
            coordinate_wgrad=native.coordinate_wgrad,
            first_activation_wgrad=native.coordinate_activation_wgrad if mode == "fused" else None)
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(4): self.eager()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss, self.gradient = self.eager()
        self.metadata = dict(self.step.metadata, first_vjp_mode=mode, graph_pool=list(self.graph.pool()),
                             parameter_data_ptr=flat.data_ptr())

    def eager(self):
        loss, gradients = self.step(*views(self.flat))
        return loss, torch.cat([g.reshape(-1) for g in gradients])

    def replay(self):
        self.graph.replay()
        return self.loss, self.gradient


def module_preflight(native):
    problem = Problem(interior_count=2048, edge_count=128, validation_count=64)
    initial = parameters(531901)
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
        for mode in ("coordinate", "fused"):
            engine = FirstVjpWgradModuleGraph(problem, initial.to("cuda").clone(), native, mode, layout)
            optimizer = Adam(engine.flat, engine.gradient)
            worst = 0.
            for expected_loss, expected_gradient, expected_parameters in trajectory:
                loss, gradient = engine.replay()
                worst = max(worst, compare(loss, expected_loss), compare(gradient, expected_gradient))
                optimizer.step()
                worst = max(worst, compare(engine.flat, expected_parameters))
            assert digest(engine.flat) != digest(initial), "Adam did not update parameter storage"
            rows.append({"layout": layout, "first_vjp_mode": mode, "adam_updates": 3,
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
              "contract": {"candidate": "coordinate_activation_wgrad_G2", "kernel_only": args.kernel_only,
                           "matched_control": "complete native VJP materialized then identical coordinate_wgrad G1 helper",
                           "precision": "complete FP64 stable Q10/Q20 VJP; same G1 reduction and zero products; fmad=false",
                           "point_tile": 8, "merge": "sequential deterministic; no atomics; not GEMM bitwise order",
                           "atol": 2e-12, "rtol": 2e-11, "tail_mode": "torch", "timing": "none",
                           "solver_convergence_validated": False}}

    def save(): args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("this preflight requires an allocated SM120 CUDA GPU")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(91171)
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
