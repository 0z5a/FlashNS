"""Development record (not a result): executed once on GH200 against the development tree of 2026-10-22;
later revisions of box376.external_c36 changed the surrogate form and removed build_functional_func.

Third development probe of the external C36 backends on GH200 (development seed; not a result).

Variants: func_jacfwd traced with make_fx and run without torch.compile, then captured in a CUDA
Graph; func_jacfwd as one functional transform (torch.func.grad_and_value) compiled with explicit
arguments and captured; jet_collapsed traced at chunk 128 and captured. Each is compared with dense
parameter AD at the development initialization and, for every graph, after an in-place change.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.data import flatten, initialize, views
from box376.external_c36 import build_compiled_external, build_functional_func
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_run import capture, load_problem


def timed(function, repeats):
    torch.cuda.synchronize(); started = time.perf_counter()
    for _ in range(repeats):
        result = function()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) / repeats


def compare(actual, reference, protocol):
    worst = 0.0
    for a, b in zip(actual, reference):
        a, b = a.detach().double().cpu().numpy(), b.detach().double().cpu().numpy()
        worst = max(worst, float(np.max(np.abs(a - b) / (protocol['atol'] + protocol['rtol'] * np.abs(b)))))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=361901)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads(args.protocol.read_text())
    data, calibration = load_problem(args, protocol)
    topology = tuple(protocol['topology'])
    initial = flatten(initialize(topology, args.seed)).cuda()
    changed = initial + 1e-3
    chunk = protocol['explicit_chunk']
    dense = PreparedPotentialADObjective(data, calibration)
    reference = tuple(v.clone() for v in dense(views(initial, topology), chunk))
    reference_changed = tuple(v.clone() for v in dense(views(changed, topology), chunk))
    out = {'state': 'RUNNING', 'kind': 'development probe, not a result', 'seed': args.seed, 'variants': {}}

    def save():
        args.output.write_text(json.dumps(out, indent=1) + '\n')

    def run_variant(name, build, call_of, sizes):
        row = out['variants'][name] = {}
        for size in sizes:
            try:
                theta = initial.clone().requires_grad_(name != 'func_functional_compiled')
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                built = build(theta, size)
                call = call_of(built, theta)
                for _ in range(2):
                    call()
                torch.cuda.synchronize()
                row.update(chunk=size, build_and_warmup_seconds=time.perf_counter() - started)
                value, row['seconds'] = timed(call, 3)
                row['error'] = compare(value, reference, protocol)
                save()
                graph, row['capture_seconds'] = capture(call)
                value, row['graph_seconds'] = timed(graph, 10)
                row['graph_error'] = compare(value, reference, protocol)
                with torch.no_grad():
                    theta.copy_(changed)
                value = graph()
                torch.cuda.synchronize()
                row['graph_changed_error'] = compare(value, reference_changed, protocol)
                row['peak_bytes'] = torch.cuda.max_memory_allocated()
                break
            except torch.cuda.OutOfMemoryError:
                row.setdefault('oom_chunks', []).append(size)
                torch.cuda.empty_cache()
            except BaseException:
                row['failure'] = traceback.format_exc()[-3000:]
                break
        save()

    run_variant('func_traced', lambda theta, size: build_compiled_external(data, calibration, 'func_jacfwd', theta, topology, size, compile=False),
                lambda built, theta: built, (4096, 1024))
    run_variant('func_functional_compiled', lambda theta, size: build_functional_func(data, calibration, topology, size),
                lambda built, theta: (lambda: (lambda r: (r[1][1][0], r[0], r[1][1][1]))(built(theta))), (1024, 512))
    run_variant('jet_traced', lambda theta, size: build_compiled_external(data, calibration, 'jet_collapsed', theta, topology, size, compile=False),
                lambda built, theta: built, (128,))
    out['state'] = 'COMPLETE'
    save()
    print(json.dumps(out['variants'], indent=1))


if __name__ == '__main__':
    main()
