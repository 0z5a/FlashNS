"""Width/depth wrapper around bench_gpu for the 2D representation/execution ablation.

    python ablation_bench.py --hidden 64,64 <bench_gpu arguments>

The hidden widths select the network 2-h1-...-hk-3. For the original topology (2, 64, 64, 3)
no function is replaced and bench_gpu runs unchanged. For any other topology the module-level
WIDTHS of problem.py is set and the hard-coded three-matrix views() is replaced by a
depth-generic one in every module that imported it; parameters() already reads WIDTHS at call
time. Every report records the topology and whether a replacement was active.
"""
import argparse
import hashlib
import sys
from itertools import pairwise

import bench_gpu
import gpu_engine
import problem

ORIGINAL = (2, 64, 64, 3)


def generic_views(widths):
    """problem.views for an arbitrary number of layers."""
    layers = len(widths) - 1

    def views(flat):
        sizes = [i * o for i, o in pairwise(widths)] + list(widths[1:])
        parts = flat.split(sizes)
        weights = [part.view(o, i) for part, (i, o) in zip(parts[:layers], pairwise(widths))]
        return weights, list(parts[layers:])
    return views


def configure(hidden, seed):
    widths = (2, *hidden, 3)
    replaced = widths != ORIGINAL
    if replaced:
        views = generic_views(widths)
        problem.WIDTHS = widths
        problem.views = gpu_engine.views = bench_gpu.views = views
    initial = problem.parameters(seed)
    network = {"widths": list(widths), "parameter_count": sum(i * o + o for i, o in pairwise(widths)),
               "initial_seed": seed, "initial_parameters_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
               "generic_views_active": replaced, "wrapper": "experiments/contracted_jets/ablation_bench.py"}
    assert initial.numel() == network["parameter_count"]
    original_write = bench_gpu.write_report

    def write_report(path, report):
        report["network"] = network
        return original_write(path, report)
    bench_gpu.write_report = write_report
    return widths


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--hidden", default="64,64")
    pre.add_argument("--seed", type=int, default=75101)
    known, rest = pre.parse_known_args()
    rest = [*rest, "--seed", str(known.seed)]
    hidden = tuple(int(value) for value in known.hidden.split(","))
    if not hidden or any(value < 1 for value in hidden):
        raise SystemExit("--hidden needs positive widths")
    configure(hidden, known.seed)
    sys.argv = [sys.argv[0], *rest]
    bench_gpu.main()


if __name__ == "__main__":
    main()
