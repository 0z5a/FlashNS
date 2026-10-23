"""Development diagnostic (not a result): where the torch.func arm's device memory goes.

For development seed 361901, step by step: allocated and reserved memory after the dense reference,
after the eager torch.func evaluation, after building the chunked traced evaluation, after its warm-up,
after CUDA Graph capture and after empty_cache. Then the alternative without make_fx: the eager
torch.func evaluation captured directly in a CUDA Graph, at two chunk sizes, with the same records,
a changed-parameter check and the replay time.
"""
import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.data import flatten, initialize, views
from box376.external_c36 import PreparedPotentialExternalObjective, build_chunked_traced_external
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_run import capture, load_problem


def memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(), 'free': free,
            'peak': torch.cuda.max_memory_allocated()}


def ratio(actual, reference, protocol):
    worst = 0.0
    for a, b in zip(actual, reference):
        a, b = a.detach().double().cpu().numpy(), b.detach().double().cpu().numpy()
        worst = max(worst, float(np.max(np.abs(a - b) / (protocol['atol'] + protocol['rtol'] * np.abs(b)))))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    protocol = json.loads(args.protocol.read_text())
    data, calibration = load_problem(args, protocol)
    topology = tuple(protocol['topology'])
    initial = flatten(initialize(topology, 361901)).cuda()
    changed = initial + 1e-3
    dense = PreparedPotentialADObjective(data, calibration)
    reference = tuple(v.clone() for v in dense(views(initial, topology), protocol['explicit_chunk']))
    reference_changed = tuple(v.clone() for v in dense(views(changed, topology), protocol['explicit_chunk']))
    out = {'kind': 'development diagnostic, not a result', 'steps': {}, 'eager_graph': {}}

    def save():
        args.output.write_text(json.dumps(out, indent=1) + '\n')

    def clear():
        gc.collect(); torch.cuda.empty_cache()

    out['steps']['after_dense_reference'] = memory()
    try:
        theta = initial.clone().requires_grad_(True)
        eager = PreparedPotentialExternalObjective(data, calibration, 'func_jacfwd')
        eager(views(theta, topology), 1024)
        out['steps']['after_eager_1024'] = memory()
        evaluation = build_chunked_traced_external(data, calibration, 'func_jacfwd', theta, topology, 1024)
        out['steps']['after_chunked_trace_1024'] = memory()
        for _ in range(2):
            evaluation()
        out['steps']['after_warmup'] = memory()
        replay, _ = capture(evaluation)
        out['steps']['after_capture'] = memory()
        clear()
        out['steps']['after_empty_cache'] = memory()
        del replay, evaluation, eager
        clear()
        out['steps']['after_release'] = memory()
    except BaseException:
        out['steps']['failure'] = traceback.format_exc()[-2000:]
    save()
    for chunk in (1024, 512):
        row = out['eager_graph'][str(chunk)] = {}
        try:
            clear(); torch.cuda.reset_peak_memory_stats()
            theta = initial.clone().requires_grad_(True)
            objective = PreparedPotentialExternalObjective(data, calibration, 'func_jacfwd')
            params = views(theta, topology)
            call = lambda: objective(params, chunk)
            for _ in range(2):
                call()
            row['after_warmup'] = memory()
            replay, row['capture_seconds'] = capture(call)
            row['after_capture'] = memory()
            row['error'] = ratio(replay(), reference, protocol)
            torch.cuda.synchronize(); started = time.perf_counter()
            for _ in range(10):
                replay()
            torch.cuda.synchronize()
            row['replay_seconds'] = (time.perf_counter() - started) / 10
            with torch.no_grad():
                theta.copy_(changed)
            value = tuple(v.clone() for v in replay())
            _ = dense(views(changed, topology), protocol['explicit_chunk'])  # an evaluation between replays
            row['changed_error'] = ratio(value, reference_changed, protocol)
            row['changed_error_after_other_work'] = ratio(replay(), reference_changed, protocol)
            clear()
            row['after_empty_cache'] = memory()
            del replay, call, objective, params
            clear()
        except BaseException:
            row['failure'] = traceback.format_exc()[-2000:]
        save()
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
