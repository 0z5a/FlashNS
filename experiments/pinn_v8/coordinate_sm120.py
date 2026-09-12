"""Physical-coordinate first-affine CUDA checks and paired SM120 screening.

Only raw Cartesian inputs are covered. Timing measures the complete compact
loss/parameter-gradient Graph with the original torch tail in both arms.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

from runtime import (NativeCUDA, Problem, ROOT, compare, device_metadata,
                     snapshot, source_hashes)
from flashns.jet_packed import PackedLayout
from tail_sm120 import (MODES, command_output, device_uuid, module_preflight,
                        screening, sha256, telemetry)


def dense_affine(layout, x, weight, bias):
    result = layout.coordinates(x) @ weight.T
    result[layout.zero_rows(x.device)] += bias
    return result


def guarded(tensor):
    holder = torch.full((tensor.numel()+2,), 917., dtype=tensor.dtype, device=tensor.device)
    view = holder[1:-1].view_as(tensor)
    view.copy_(tensor)
    return view, holder


def check_canary(holder):
    assert holder[0].item() == 917. and holder[-1].item() == 917., "modified allocation guard"


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
                    tensors = (torch.randn(layout.points, dimension, dtype=torch.float64, device="cuda"),
                               torch.randn(channels, dimension, dtype=torch.float64, device="cuda") * 0.3,
                               torch.randn(channels, dtype=torch.float64, device="cuda") * 0.3)
                    inputs, holders = zip(*(guarded(tensor) for tensor in tensors))
                    expected = dense_affine(layout, *inputs)
                    actual = native.coordinate_affine(layout, *inputs)
                    # Also exercise the C ABI with a caller-owned offset output.
                    # NaNs expose missed stores; canaries expose adjacent writes.
                    output, output_holder = guarded(torch.full_like(expected, float("nan")))
                    native.launch(native.coordinate_launch, [dimension, *[t.data_ptr() for t in inputs],
                                  output.data_ptr(), ni, nb, channels], (*inputs, output))
                    worst = max(compare(actual, expected), compare(output, expected))
                    for view, original, holder in zip(inputs, tensors, holders):
                        assert torch.equal(view, original), "coordinate affine modified an input"
                        check_canary(holder)
                    check_canary(output_holder)
                    rows.append({"q": layout.q, "full_points": ni, "value_points": nb,
                                 "channels": channels, "max_error_over_tolerance": worst})
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return {"cases": rows, "non_default_stream": True, "offset_input_and_output_views": True,
            "output_prefilled_with_nan": True, "reference": "original dense coordinate-jet GEMM + bias"}


def nonfinite_preflight(native):
    rows = []
    for dimension in (2, 3):
        layout = PackedLayout(3, 2, dimension)
        x = torch.ones(layout.points, dimension, dtype=torch.float64, device="cuda")
        x[0] = 0.
        weight = torch.ones(5, dimension, dtype=x.dtype, device=x.device)
        weight[0, 0], weight[1, 0], weight[2, 0] = float("inf"), -float("inf"), float("nan")
        bias = torch.zeros(5, dtype=x.dtype, device=x.device)
        bias[3] = float("nan")
        expected = dense_affine(layout, x, weight, bias)
        actual = native.coordinate_affine(layout, x, weight, bias)
        for predicate in (torch.isnan, torch.isposinf, torch.isneginf):
            assert torch.equal(predicate(actual), predicate(expected)), "nonfinite propagation differs"
        finite = torch.isfinite(expected)
        worst = compare(actual[finite], expected[finite])
        rows.append({"q": layout.q, "matching_nan_and_signed_inf_masks": True,
                     "finite_max_error_over_tolerance": worst})
    return rows


def kernel_graph_updates(native, quick=False):
    rows = []
    for dimension in (2, 3):
        for channels in ((7,) if quick else (7, 32, 64, 128)):
            layout = PackedLayout(17, 5, dimension)
            x = torch.randn(layout.points, dimension, dtype=torch.float64, device="cuda")
            weight = torch.randn(channels, dimension, dtype=x.dtype, device=x.device)
            bias = torch.randn(channels, dtype=x.dtype, device=x.device)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    native.coordinate_affine(layout, x, weight, bias)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = native.coordinate_affine(layout, x, weight, bias)
            worst, previous = 0., None
            for update in range(3):
                x.normal_()
                weight.normal_(0., 0.3)
                bias.normal_(0., 0.3)
                graph.replay()
                worst = max(worst, compare(output, dense_affine(layout, x, weight, bias)))
                current = output.clone()
                if previous is not None:
                    assert not torch.equal(previous, current), "Graph replay used stale inputs"
                previous = current
            rows.append({"q": layout.q, "channels": channels, "coordinate_weight_bias_updates": 3,
                         "max_error_over_tolerance": worst})
    torch.cuda.synchronize()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--kernel-only", action="store_true",
                        help="with --preflight-only: bounded native checks for compute-sanitizer")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=89242)
    args = parser.parse_args()
    if args.kernel_only and not args.preflight_only:
        parser.error("--kernel-only requires --preflight-only; it cannot authorize timing")
    if args.blocks < 6 or args.blocks % 2 or args.replays < 1 or args.warmup < 1:
        parser.error("even --blocks >= 6 and positive --replays/--warmup required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write('{"completed": false, "preflight_passed": false}\n')
    report = {"completed": False, "preflight_passed": False, "checks": [], "screening": None,
              "contract": {"atol": 2e-12, "rtol": 2e-11,
                           "input": "raw physical Cartesian coordinates; full Taylor basis revision 1",
                           "first_affine_modes": ["dense_affine", "coordinate_affine"],
                           "module_preflight_tail_modes": MODES,
                           "screening_tail_mode": "torch", "layout": "compact", "seed": "cuda",
                           "activation": "native stable FP64",
                           "workload": "2048 interior + 512 boundary; 2-64-64-3",
                           "weight_seed_preflight": 361901, "weight_seed_screening": 361902,
                           "order_seed": args.order_seed, "paired_blocks_per_comparison": args.blocks,
                           "warmup_replays_every_arm": args.warmup, "replays_per_arm": args.replays,
                           "minimum_gain": 0.01, "AA_ratio_deviation_limit": 0.02,
                           "baseline_return_deviation_limit": 0.05,
                           "drift_policy": "retain every raw block; any drift failure prevents a positive signal",
                           "sanitizers": "separate coordinate-specific compute-sanitizer invocation required",
                           "kernel_only": args.kernel_only,
                           "evidence_level": "screening only; no solver convergence or clean PR validation"}}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("this run requires an SM120 CUDA GPU")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(83110)
        native = NativeCUDA(args.build)
        library = args.build / native.metadata["library"]
        report.update(device=dict(device_metadata(), uuid=device_uuid()), source_hashes=snapshot(args.output),
                      build=native.metadata, binary_path=str(library.resolve()), binary_sha256=sha256(library),
                      build_metadata_sha256=sha256(args.build / "build.json"),
                      process={"pid": os.getpid(), "python": sys.executable, "argv": sys.argv,
                               "modules": {name: str(Path(sys.modules[name].__file__).resolve())
                                           for name in ("runtime", "native", "backends", "tail_sm120", "flashns.pinn_packed")},
                               "git_head": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                               "git_status": command_output(["git", "-C", str(ROOT), "status", "--short"]),
                               "environment": {name: os.environ[name] for name in
                                               ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS")
                                               if name in os.environ}}, telemetry_start=telemetry())

        def locked():
            if (source_hashes() != report["source_hashes"] or sha256(library) != report["binary_sha256"]
                    or sha256(args.build / "build.json") != report["build_metadata_sha256"]):
                raise RuntimeError("source/build changed during this process; timings invalid")

        checks = [("Q10_Q20_coordinate_boundaries_nondefault_stream",
                   lambda: kernel_preflight(native, quick=args.kernel_only)),
                  ("nonfinite_weight_and_bias_propagation", lambda: nonfinite_preflight(native)),
                  ("coordinate_graph_input_updates", lambda: kernel_graph_updates(native, quick=args.kernel_only))]
        problem = None
        if not args.kernel_only:
            problem = Problem(interior_count=2048, edge_count=128, validation_count=64)
            checks.append(("coordinate_all_layouts_tail_modes_three_adam_updates_vs_nested_AD",
                           lambda: module_preflight(problem, native, coordinate=True)))
        for name, check in checks:
            report["active_check"] = name
            save()
            detail = check()
            report["checks"].append({"name": name, "passed": True, "detail": detail})
            save()
            print(name, "passed", flush=True)
        locked()
        del report["active_check"]
        report["preflight_passed"] = True
        save()
        if not args.preflight_only:
            screening(problem, native, args, report, save, locked, coordinate=True)
        report["completed"] = True
        report["telemetry_end"] = telemetry()
        save()
    except Exception as error:
        if report.get("screening") is not None:
            report["screening"].update(status="invalid", invalid_reason=str(error))
        report["failure"] = {"error": str(error), "traceback": traceback.format_exc()}
        save()
        raise


if __name__ == "__main__":
    main()
