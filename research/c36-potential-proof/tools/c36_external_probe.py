"""Development record (not a result): executed once on GH200 against the development tree of 2026-10-22;
later revisions of box376.external_c36 changed the surrogate form and removed build_functional_func.

Development probe of the external C36 backends on GH200 (development seed; not a result).

    PYTHONPATH=src FLASHNS_TORCHJET=SNAPSHOT python tools/c36_external_probe.py --inputs DATA --protocol P --output OUT.json

For the stored development initialization: dense parameter AD on the complete data is the
reference. The analytic engine is timed eager, compiled and as a CUDA Graph replay. Each external
backend is evaluated eager on the complete data at the largest chunk that fits, compared with the
reference within the original budget, and timed; a backend whose first interior chunk projects to
more than --max-seconds for a complete evaluation is timed on that chunk only and recorded as such.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.data import flatten, initialize, views
from box376.external_c36 import RAW, PreparedPotentialExternalObjective
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.potential_run import capture, load_problem


def sync():
    torch.cuda.synchronize()


def timed(function, repeats):
    sync(); started = time.perf_counter()
    for _ in range(repeats):
        result = function()
    sync()
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
    parser.add_argument('--max-seconds', type=float, default=900.)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads(args.protocol.read_text())
    data, calibration = load_problem(args, protocol)
    topology = tuple(protocol['topology'])
    theta = flatten(initialize(topology, args.seed)).cuda()
    params = views(theta, topology)
    chunk = protocol['explicit_chunk']
    out = {'state': 'RUNNING', 'kind': 'development probe, not a result', 'seed': args.seed,
           'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'interior_points': len(data['families']['interior']['points']),
           'methods': {}}

    def save():
        args.output.write_text(json.dumps(out, indent=1) + '\n')

    dense = PreparedPotentialADObjective(data, calibration)
    reference, out['dense_ad_seconds'] = timed(lambda: tuple(v.clone() for v in dense(params, chunk)), 1)
    save()
    # Analytic engine, as in the trainers.
    from torch_c36_packed import activate_packed
    from torch_c36_packed_vjp import activate_packed_vjp
    eager = PreparedPotentialNormalizedObjective(data, calibration, activation=activate_packed, activation_vjp=activate_packed_vjp)
    row = {}
    value, row['eager_seconds'] = timed(lambda: eager(params, chunk), 3)
    row['eager_error'] = compare(value, reference, protocol)
    forward = torch.compile(activate_packed, fullgraph=True, dynamic=False)
    transpose = torch.compile(activate_packed_vjp, fullgraph=True, dynamic=False)
    compiled = PreparedPotentialNormalizedObjective(data, calibration, activation=forward, activation_vjp=transpose)
    for _ in range(2):
        compiled(params, chunk)
    value, row['compiled_seconds'] = timed(lambda: compiled(params, chunk), 5)
    replay, _ = capture(lambda: compiled(params, chunk))
    value, row['graph_seconds'] = timed(replay, 20)
    row['graph_error'] = compare(value, reference, protocol)
    out['methods']['analytic'] = row
    save()
    for backend in RAW:
        row = out['methods'][backend] = {}
        try:
            objective = PreparedPotentialExternalObjective(data, calibration, backend)
            # Projection from the first interior chunk at each candidate chunk size.
            for size in (512, 128, 32):
                try:
                    torch.cuda.reset_peak_memory_stats()
                    inputs, gate = objective._base._geometry['interior']
                    leaves = params
                    alpha, origin, _ = objective._base._model_constants
                    weight, bias = leaves[0]
                    def first_chunk():
                        w = weight.detach().clone().requires_grad_(True)
                        physical = [(w * alpha, bias - (w * alpha) @ origin)] + list(leaves[1:])
                        raw = RAW[backend](physical, inputs[:size])
                        return torch.autograd.grad(raw.square().sum(), w)[0]
                    _, seconds = timed(first_chunk, 1)
                    row['chunk'] = size
                    row['first_chunk_seconds'] = seconds
                    row['first_chunk_peak_bytes'] = torch.cuda.max_memory_allocated()
                    row['projected_interior_seconds'] = seconds * row['interior_points'] / size if 'interior_points' in row else seconds * out['interior_points'] / size
                    break
                except torch.cuda.OutOfMemoryError:
                    row.setdefault('oom_chunks', []).append(size)
                    torch.cuda.empty_cache()
            save()
            if row.get('projected_interior_seconds', 1e30) <= args.max_seconds:
                value, row['eager_seconds'] = timed(lambda: objective(params, row['chunk']), 1)
                row['eager_error'] = compare(value, reference, protocol)
                row['peak_bytes'] = torch.cuda.max_memory_allocated()
                row['complete_evaluation'] = True
            else:
                row['complete_evaluation'] = False
        except BaseException:
            row['failure'] = traceback.format_exc()[-3000:]
        save()
    out['state'] = 'COMPLETE'
    save()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'failure'} for k, v in out['methods'].items()}, indent=1))


if __name__ == '__main__':
    main()
