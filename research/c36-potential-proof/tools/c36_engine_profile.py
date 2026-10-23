"""Development profile (not a result): where a C36 analytic evaluation spends its time on GH200.

For development seed 361901: the analytic engine (compiled packed activation and transpose, CUDA Graph
replay) at several chunk sizes, with replay time, peak memory and a check against dense parameter AD;
then one profiled replay at the protocol chunk and at the fastest chunk, with kernel time grouped by
kind (GEMM, Triton/Inductor, elementwise/index, reduction, copy, other) and the kernel count. The
torch.func eager-graph evaluation is timed at the same chunk sizes for reference.
"""
import argparse
import gc
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from box376.data import flatten, initialize, views
from box376.external_c36 import PreparedPotentialExternalObjective
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.potential_run import capture, load_problem


def ratio(actual, reference, protocol):
    worst = 0.0
    for a, b in zip(actual, reference):
        a, b = a.detach().double().cpu().numpy(), b.detach().double().cpu().numpy()
        worst = max(worst, float(np.max(np.abs(a - b) / (protocol['atol'] + protocol['rtol'] * np.abs(b)))))
    return worst


def kind(name):
    n = name.lower()
    if 'gemm' in n or 'cutlass' in n or 'sm90' in n or 'xmma' in n or 'dgemm' in n:
        return 'gemm'
    if n.startswith('triton') or 'triton' in n:
        return 'triton'
    if 'reduce' in n or 'reduction' in n:
        return 'reduction'
    if 'copy' in n or 'memcpy' in n or 'memset' in n:
        return 'copy'
    if 'elementwise' in n or 'index' in n or 'gather' in n or 'scatter' in n or 'cat' in n:
        return 'elementwise_index'
    return 'other'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads(args.protocol.read_text())
    data, calibration = load_problem(args, protocol)
    topology = tuple(protocol['topology'])
    initial = flatten(initialize(topology, 361901)).cuda()
    params = views(initial, topology)
    dense = PreparedPotentialADObjective(data, calibration)
    reference = tuple(v.clone() for v in dense(params, protocol['explicit_chunk']))
    from torch_c36_packed import activate_packed
    from torch_c36_packed_vjp import activate_packed_vjp
    forward = torch.compile(activate_packed, fullgraph=True, dynamic=False)
    transpose = torch.compile(activate_packed_vjp, fullgraph=True, dynamic=False)
    out = {'kind': 'development profile, not a result', 'analytic': {}, 'func_eager_graph': {}, 'profiles': {}}

    def save():
        args.output.write_text(json.dumps(out, indent=1) + '\n')

    keep = {}
    for chunk in (128, 256, 512, 1024, 2048, 4096):
        row = out['analytic'][str(chunk)] = {}
        try:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            engine = PreparedPotentialNormalizedObjective(data, calibration, activation=forward, activation_vjp=transpose)
            call = lambda engine=engine, chunk=chunk: engine(params, chunk)
            for _ in range(2):
                call()
            replay, _ = capture(call)
            row['error'] = ratio(replay(), reference, protocol)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(20):
                replay()
            torch.cuda.synchronize()
            row['replay_seconds'] = (time.perf_counter() - t) / 20
            row['peak_bytes'] = torch.cuda.max_memory_allocated()
            keep[chunk] = (replay, engine, call)
        except BaseException:
            row['failure'] = traceback.format_exc()[-1500:]
        save()
    timed = {c: r['replay_seconds'] for c, r in out['analytic'].items() if 'replay_seconds' in r}
    best = min(timed, key=timed.get) if timed else None
    out['analytic_best_chunk'] = best
    for chunk in sorted({str(protocol['explicit_chunk']), best} - {None}, key=int):
        replay = keep[int(chunk)][0]
        try:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                replay(); torch.cuda.synchronize()
            groups, count = defaultdict(float), 0
            for event in prof.key_averages():
                if event.device_type.name == 'CUDA' or getattr(event, 'self_device_time_total', 0):
                    us = getattr(event, 'self_device_time_total', None) or getattr(event, 'self_cuda_time_total', 0)
                    if us:
                        groups[kind(event.key)] += us
                        count += event.count
            top = sorted(((e.key, getattr(e, 'self_device_time_total', 0) or getattr(e, 'self_cuda_time_total', 0), e.count) for e in prof.key_averages()), key=lambda r: -r[1])[:15]
            out['profiles'][chunk] = {'device_microseconds_by_kind': dict(groups), 'kernel_launches': count,
                                      'top': [{'kernel': k[:120], 'microseconds': us, 'count': n} for k, us, n in top]}
        except BaseException:
            out['profiles'][chunk] = {'failure': traceback.format_exc()[-1500:]}
        save()
    keep.clear(); gc.collect(); torch.cuda.empty_cache()
    for chunk in (512, 1024, 2048):
        row = out['func_eager_graph'][str(chunk)] = {}
        try:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            theta = initial.clone()
            objective = PreparedPotentialExternalObjective(data, calibration, 'func_jacfwd')
            p = views(theta, topology)
            call = lambda objective=objective, p=p, chunk=chunk: objective(p, chunk)
            for _ in range(2):
                call()
            replay, _ = capture(call)
            row['error'] = ratio(replay(), reference, protocol)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(5):
                replay()
            torch.cuda.synchronize()
            row['replay_seconds'] = (time.perf_counter() - t) / 5
            row['peak_bytes'] = torch.cuda.max_memory_allocated()
            del replay, call, objective
        except BaseException:
            row['failure'] = traceback.format_exc()[-1500:]
        save()
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
