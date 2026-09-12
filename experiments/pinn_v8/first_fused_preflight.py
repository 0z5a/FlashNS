"""First coordinate-affine/stable-activation GPU correctness only; no timing.

The matched control is coordinate_affine followed by the original activation.
--kernel-only is bounded sanitizer coverage and cannot authorize performance.
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
from coordinate_sm120 import dense_affine, guarded, check_canary
from tail_sm120 import device_uuid, sha256


def separate(native, layout, x, weight, bias):
    return native.activation_factory(layout).forward(native.coordinate_affine(layout, x, weight, bias))


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
                                 torch.randn(channels, dimension, dtype=torch.float64, device="cuda")*0.3,
                                 torch.randn(channels, dtype=torch.float64, device="cuda")*0.3)
                    inputs, holders = zip(*(guarded(t) for t in originals))
                    expected_h, expected_aux = separate(native, layout, *inputs)
                    actual_h, actual_aux = native.coordinate_affine_activation(layout, *inputs)
                    output_h, holder_h = guarded(torch.full_like(expected_h, float("nan")))
                    output_aux, holder_aux = guarded(torch.full_like(expected_aux, float("nan")))
                    native.launch(native.first_fused_launch,
                                  [dimension, *[t.data_ptr() for t in inputs], output_h.data_ptr(),
                                   output_aux.data_ptr(), ni, nb, channels], (*inputs, output_h, output_aux))
                    worst = max(compare(actual_h, expected_h), compare(actual_aux, expected_aux),
                                compare(output_h, expected_h), compare(output_aux, expected_aux))
                    seed = torch.randn_like(expected_h)
                    ops = native.activation_factory(layout)
                    actual_vjp = ops.vjp(actual_h, actual_aux, seed)
                    worst = max(worst, compare(actual_vjp, ops.vjp(expected_h, expected_aux, seed)))
                    if not quick:
                        # The matched native arms share their coefficient helper.
                        # Dense coordinate jets and tensor activation independently
                        # cover the helper/indexing, including the Q20 full basis.
                        tensor_ops = TensorActivation(layout)
                        independent_h, independent_aux = tensor_ops.forward(dense_affine(layout, *inputs))
                        worst = max(worst, compare(expected_h, independent_h), compare(expected_aux, independent_aux),
                                    compare(actual_h, independent_h), compare(actual_aux, independent_aux),
                                    compare(actual_vjp, tensor_ops.vjp(independent_h, independent_aux, seed)))
                    for view, original, holder in zip(inputs, originals, holders):
                        assert torch.equal(view, original), "first fusion modified its input"
                        check_canary(holder)
                    check_canary(holder_h)
                    check_canary(holder_aux)
                    rows.append({"q": layout.q, "full_points": ni, "value_points": nb,
                                 "channels": channels, "max_error_over_tolerance": worst,
                                 "independent_dense_coordinate_and_tensor_activation_oracle": not quick})
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return {"rows": rows, "non_default_stream": True, "offset_inputs_and_outputs": True,
            "both_outputs_prefilled_with_nan": True, "subsequent_VJP_checked": True}


def nonfinite_and_tail_preflight(native):
    rows = []
    for dimension in (2, 3):
        layout = PackedLayout(3, 2, dimension)
        for location in ("weight", "bias", "coordinate"):
            x = torch.ones(layout.points, dimension, dtype=torch.float64, device="cuda")
            x[0] = 0.
            weight = torch.ones(7, dimension, dtype=x.dtype, device=x.device)*0.3
            bias = torch.zeros(7, dtype=x.dtype, device=x.device)
            for i, value in enumerate((float("inf"), -float("inf"), float("nan"))):
                if location == "coordinate": x[i, 0] = value
                elif location == "weight": weight[i, 0] = value
                else: bias[i] = value
            actual, expected = native.coordinate_affine_activation(layout, x, weight, bias), separate(native, layout, x, weight, bias)
            worst = 0.
            for observed, target in zip(actual, expected):
                for predicate in (torch.isnan, torch.isposinf, torch.isneginf):
                    assert torch.equal(predicate(observed), predicate(target)), "nonfinite propagation differs"
                mask = torch.isfinite(target)
                worst = max(worst, compare(observed[mask], target[mask]))
            rows.append({"q": layout.q, "nonfinite_input": location, "matching_nonfinite_masks": True,
                         "finite_max_error_over_tolerance": worst})
        # Rounded H[0] cannot replace this auxiliary checkpoint in either segment.
        x = torch.zeros(layout.points, dimension, dtype=torch.float64, device="cuda")
        weight = torch.full((2, dimension), 0.3, dtype=x.dtype, device=x.device)
        bias = torch.tensor([20., -20.], dtype=x.dtype, device=x.device)
        hidden, aux = native.coordinate_affine_activation(layout, x, weight, bias)
        expected_h, expected_aux = separate(native, layout, x, weight, bias)
        assert torch.equal(hidden[layout.zero_rows(x.device)].abs(), torch.ones_like(aux))
        assert bool((aux > 0).all()), "saturated stable a1 was reconstructed from rounded H"
        rows.append({"q": layout.q, "saturated_centers": [20., -20.],
                     "max_error_over_tolerance": max(compare(hidden, expected_h), compare(aux, expected_aux)),
                     "positive_a1_with_rounded_H": True})
    return rows


def kernel_graph_updates(native, quick=False):
    rows = []
    for dimension in (2, 3):
        for channels in ((7,) if quick else (7, 32, 64, 128)):
            layout = PackedLayout(17, 5, dimension)
            x = torch.randn(layout.points, dimension, dtype=torch.float64, device="cuda")*0.3
            weight = torch.randn(channels, dimension, dtype=x.dtype, device=x.device)*0.3
            bias = torch.randn(channels, dtype=x.dtype, device=x.device)*0.3
            seed = torch.randn(layout.rows, channels, dtype=x.dtype, device=x.device)*0.3
            ops = native.activation_factory(layout)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    h, a = native.coordinate_affine_activation(layout, x, weight, bias)
                    ops.vjp(h, a, seed)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                hidden, aux = native.coordinate_affine_activation(layout, x, weight, bias)
                gradient = ops.vjp(hidden, aux, seed)
            worst, previous = 0., None
            for _ in range(3):
                for tensor in (x, weight, bias, seed):
                    tensor.normal_(0., 0.3)
                graph.replay()
                reference_h, reference_aux = separate(native, layout, x, weight, bias)
                reference_gradient = ops.vjp(reference_h, reference_aux, seed)
                worst = max(worst, compare(hidden, reference_h), compare(aux, reference_aux),
                            compare(gradient, reference_gradient))
                current = hidden.clone()
                if previous is not None:
                    assert not torch.equal(previous, current), "Graph used stale first-layer inputs"
                previous = current
            rows.append({"q": layout.q, "channels": channels, "input_and_parameter_updates": 3,
                         "max_error_over_tolerance": worst})
    torch.cuda.synchronize()
    return rows


class FirstModuleGraph:
    def __init__(self, problem, flat, native, mode, layout):
        self.flat = flat
        ni = problem.interior_count
        self.step = PackedPINN(
            problem.x, ni, problem.pde_weights[:ni], problem.boundary_weights[ni:], problem.target[ni:],
            layout=layout, seed="cuda", activation_factory=native.activation_factory, cuda_seed=native.seed,
            coordinate_affine=native.coordinate_affine,
            first_affine_activation=native.coordinate_affine_activation if mode == "fused" else None)
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(4):
                self.eager()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss, self.gradient = self.eager()
        self.metadata = dict(self.step.metadata, first_mode=mode, graph_pool=list(self.graph.pool()),
                             parameter_data_ptr=flat.data_ptr())

    def eager(self):
        loss, gradients = self.step(*views(self.flat))
        return loss, torch.cat([g.reshape(-1) for g in gradients])

    def replay(self):
        self.graph.replay()
        return self.loss, self.gradient


def module_preflight(native):
    problem = Problem(interior_count=2048, edge_count=128, validation_count=64)
    initial = parameters(361901)
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
        for mode in ("separate", "fused"):
            engine = FirstModuleGraph(problem, initial.to("cuda").clone(), native, mode, layout)
            optimizer = Adam(engine.flat, engine.gradient)
            worst = 0.
            for expected_loss, expected_gradient, expected_parameters in trajectory:
                loss, gradient = engine.replay()
                worst = max(worst, compare(loss, expected_loss), compare(gradient, expected_gradient))
                optimizer.step()
                worst = max(worst, compare(engine.flat, expected_parameters))
            assert digest(engine.flat) != digest(initial), "Adam did not change the parameter storage"
            rows.append({"layout": layout, "first_mode": mode, "adam_updates": 3,
                         "max_error_over_tolerance": worst, "metadata": engine.metadata})
            del optimizer, engine
    return {"problem": problem.meta, "rows": rows, "reference": "independent nested AD + NumPy Adam"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-only", action="store_true", help="bounded checks for Sanitizer; no performance authorization")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write('{"completed": false, "preflight_passed": false}\n')
    report = {"completed": False, "preflight_passed": False, "checks": [],
              "contract": {"candidate": "coordinate_affine_activation", "kernel_only": args.kernel_only,
                           "matched_control": "coordinate_affine + original native stable activation",
                           "precision": "FP64 full Q10/Q20, same coefficient helper, fmad=false, stable a1",
                           "atol": 2e-12, "rtol": 2e-11, "tail_mode": "torch",
                           "timing": "none", "solver_convergence_validated": False}}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("this preflight requires an allocated SM120 CUDA GPU")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(83111)
        native = NativeCUDA(args.build)
        library = args.build/native.metadata["library"]
        report.update(device=dict(device_metadata(), uuid=device_uuid()), source_hashes=snapshot(args.output),
                      build=native.metadata, binary_sha256=sha256(library), binary_path=str(library.resolve()),
                      build_metadata_sha256=sha256(args.build/"build.json"),
                      process={"pid": os.getpid(), "python": sys.executable, "argv": sys.argv})
        checks = [("Q10_Q20_full_value_empty_tail_current_stream_and_canaries", lambda: kernel_preflight(native, args.kernel_only)),
                  ("nonfinite_propagation_and_stable_saturated_aux", lambda: nonfinite_and_tail_preflight(native)),
                  ("graph_coordinate_weight_bias_seed_updates", lambda: kernel_graph_updates(native, args.kernel_only))]
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
