"""Development record (not a result): executed once on GH200 against the development tree of 2026-10-22;
later revisions of box376.external_c36 changed the surrogate form and removed build_functional_func.

Second development probe of the external C36 backends on GH200 (development seed; not a result).

For func_jacfwd and jet_collapsed: eager, compiled (make_fx + torch.compile, the Ethier pattern) and
the compiled callable captured in a CUDA Graph. Every variant is compared with dense parameter AD
at the stored development initialization and, for the graph, again after an in-place parameter
change. The analytic engine's graph replay is timed in the same process for reference.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.data import flatten, initialize, views
from box376.external_c36 import PreparedPotentialExternalObjective, build_compiled_external
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
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
    chunk = protocol['explicit_chunk']
    out = {'state': 'RUNNING', 'kind': 'development probe, not a result', 'seed': args.seed, 'methods': {}}

    def save():
        args.output.write_text(json.dumps(out, indent=1) + '\n')

    dense = PreparedPotentialADObjective(data, calibration)
    params = views(initial, topology)
    reference = tuple(v.clone() for v in dense(params, chunk))
    changed = initial + 1e-3
    reference_changed = tuple(v.clone() for v in dense(views(changed, topology), chunk))
    from torch_c36_packed import activate_packed
    from torch_c36_packed_vjp import activate_packed_vjp
    forward = torch.compile(activate_packed, fullgraph=True, dynamic=False)
    transpose = torch.compile(activate_packed_vjp, fullgraph=True, dynamic=False)
    analytic = PreparedPotentialNormalizedObjective(data, calibration, activation=forward, activation_vjp=transpose)
    for _ in range(2):
        analytic(params, chunk)
    replay, _ = capture(lambda: analytic(params, chunk))
    _, seconds = timed(replay, 20)
    out['methods']['analytic'] = {'graph_seconds': seconds}
    save()
    for backend in ('func_jacfwd', 'jet_collapsed'):
        row = out['methods'][backend] = {}
        try:
            objective = PreparedPotentialExternalObjective(data, calibration, backend)
            value, row['eager_seconds'] = timed(lambda: objective(params, 512), 1)
            row['eager_error'] = compare(value, reference, protocol)
            save()
            for size in (4096, 1024, 512):
                try:
                    theta = initial.clone().requires_grad_(True)
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    compiled = build_compiled_external(data, calibration, backend, theta, topology, size)
                    for _ in range(2):
                        compiled()
                    torch.cuda.synchronize()
                    row.update(compiled_chunk=size, compile_and_warmup_seconds=time.perf_counter() - started)
                    value, row['compiled_seconds'] = timed(compiled, 3)
                    row['compiled_error'] = compare(value, reference, protocol)
                    graph, row['capture_seconds'] = capture(compiled)
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
                    for name in ('compiled', 'graph'):
                        locals().pop(name, None)
                    torch.cuda.empty_cache()
        except BaseException:
            row['failure'] = traceback.format_exc()[-3000:]
        save()
    out['state'] = 'COMPLETE'
    save()
    print(json.dumps(out['methods'], indent=1))


if __name__ == '__main__':
    main()
