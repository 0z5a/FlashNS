"""Registered throughput comparison of the C36 analytic engine with external derivative backends.

    PYTHONPATH=src FLASHNS_TORCHJET=SNAPSHOT python tools/c36_external_throughput.py \
        --inputs DATA --protocol protocol-external-throughput-20261022.json --output OUT.json

Phase 1 (selection, development initialization): every candidate variant of every method is built
at the largest candidate chunk that runs, checked against dense parameter AD at the initialization
and, for traced and graph variants, after an in-place parameter change, and timed. A candidate that
fails or runs out of memory is recorded and dropped. Each method's selected variant is its fastest
passing candidate. Phase 2 (timing, registered initializations): for each initialization, each
method's eager variant and selected variant are built, checked against dense parameter AD and timed
in windows. The record holds every window; ratios are formed by box376-free code below.
"""
import argparse
import gc
import json
import math
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


def budget_ratio(actual, reference, protocol):
    worst = 0.0
    for a, b in zip(actual, reference):
        a, b = a.detach().double().cpu().numpy(), b.detach().double().cpu().numpy()
        worst = max(worst, float(np.max(np.abs(a - b) / (protocol['atol'] + protocol['rtol'] * np.abs(b)))))
    return worst


def clear():
    gc.collect(); torch.cuda.empty_cache()


class Builder:
    """Builds one variant of one method as a zero-argument callable over a persistent parameter leaf."""

    def __init__(self, data, calibration, topology, protocol):
        self.data, self.calibration, self.topology, self.protocol = data, calibration, topology, protocol
        self._analytic_compiled = None

    def build(self, method, variant, theta, chunk):
        data, calibration, topology = self.data, self.calibration, self.topology
        if method == 'analytic':
            from torch_c36_packed import activate_packed
            from torch_c36_packed_vjp import activate_packed_vjp
            params = views(theta, topology)
            if variant == 'eager':
                engine = PreparedPotentialNormalizedObjective(data, calibration, activation=activate_packed, activation_vjp=activate_packed_vjp)
                return lambda: engine(params, chunk)
            forward = torch.compile(activate_packed, fullgraph=True, dynamic=False)
            transpose = torch.compile(activate_packed_vjp, fullgraph=True, dynamic=False)
            engine = PreparedPotentialNormalizedObjective(data, calibration, activation=forward, activation_vjp=transpose)
            call = lambda: engine(params, chunk)
            for _ in range(2):
                call()
            if variant == 'compiled':
                return call
            return capture(call)[0]
        if variant == 'eager':
            objective = PreparedPotentialExternalObjective(data, calibration, method)
            params = views(theta, topology)
            return lambda: objective(params, chunk)
        traced = build_compiled_external(data, calibration, method, theta, topology, chunk, compile=False)
        if variant == 'traced':
            return traced
        for _ in range(2):
            traced()
        return capture(traced)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads(args.protocol.read_text())
    spec = protocol['throughput']
    data, calibration = load_problem(args, protocol)
    topology = tuple(protocol['topology'])
    chunk_reference = protocol['explicit_chunk']
    dense = PreparedPotentialADObjective(data, calibration)
    builder = Builder(data, calibration, topology, protocol)
    out = {'state': 'RUNNING', 'protocol_version': protocol['version'], 'gpu': torch.cuda.get_device_name(0),
           'torch': torch.__version__, 'selection': {}, 'selected': {}, 'timing': {}}

    def save():
        args.output.write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')

    def reference_at(theta):
        return tuple(v.clone() for v in dense(views(theta, topology), chunk_reference))

    def try_variant(method, variant, initial, chunks, check_change):
        """Build at the first chunk that runs; return (callable, theta, row) or (None, None, row)."""
        row = {'oom_chunks': []}
        reference = reference_at(initial)
        for chunk in chunks:
            theta = initial.clone().requires_grad_(method != 'analytic')
            call = None
            try:
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                call = builder.build(method, variant, theta, chunk)
                value = call(); torch.cuda.synchronize()
                row.update(chunk=chunk, build_and_first_call_seconds=time.perf_counter() - started)
                row['error_over_budget'] = budget_ratio(value, reference, protocol)
                if check_change and variant in ('traced', 'graph', 'compiled'):
                    changed = initial + spec['parameter_change']
                    with torch.no_grad():
                        theta.copy_(changed)
                    row['changed_error_over_budget'] = budget_ratio(call(), reference_at(changed), protocol)
                    with torch.no_grad():
                        theta.copy_(initial)
                row['peak_bytes'] = torch.cuda.max_memory_allocated()
                passed = row['error_over_budget'] <= 1 and row.get('changed_error_over_budget', 0) <= 1
                row['state'] = 'PASS' if passed else 'CHECK_FAILED'
                return (call, theta, row) if passed else (None, None, row)
            except torch.cuda.OutOfMemoryError:
                row['oom_chunks'].append(chunk)
                del call, theta
                clear()
            except BaseException:
                row.update(state='FAILED', failure=traceback.format_exc()[-2500:])
                del call, theta
                clear()
                return None, None, row
        row['state'] = 'OUT_OF_MEMORY'
        return None, None, row

    # Phase 1: selection on the development initialization.
    selection_init = flatten(initialize(topology, spec['selection_seed'])).cuda()
    for method, variants in spec['candidates'].items():
        out['selection'][method] = {}
        for variant in variants:
            call, theta, row = try_variant(method, variant, selection_init, spec['chunks'][method], True)
            if call is not None:
                repeats = 1 if row['build_and_first_call_seconds'] > 60 else spec['selection_repeats']
                _, row['seconds_per_evaluation'] = timed(call, repeats)
                row['repeats'] = repeats
            out['selection'][method][variant] = row
            del call, theta
            clear()
            save()
        passing = {v: r for v, r in out['selection'][method].items() if r.get('state') == 'PASS'}
        out['selected'][method] = min(passing, key=lambda v: passing[v]['seconds_per_evaluation']) if passing else None
        save()

    # Phase 2: timing on the registered initializations.
    for seed in spec['timing_seeds']:
        initial = flatten(initialize(topology, seed)).cuda()
        rows = out['timing'][str(seed)] = {}
        for method in spec['candidates']:
            for variant in sorted({'eager', out['selected'][method]} - {None}):
                key = f'{method}:{variant}'
                chunk = out['selection'][method][variant]['chunk']
                call, theta, row = try_variant(method, variant, initial, [chunk], False)
                if call is not None:
                    call(); torch.cuda.synchronize()
                    estimate = out['selection'][method][variant]['seconds_per_evaluation']
                    per_window = max(1, math.ceil(spec['window_seconds'] / estimate))
                    windows = spec['windows_fast'] if estimate * per_window < spec['slow_window_seconds'] else spec['windows_slow']
                    row['evaluations_per_window'] = per_window
                    row['windows'] = [timed(call, per_window)[1] for _ in range(windows)]
                    row['median_seconds_per_evaluation'] = float(np.median(row['windows']))
                rows[key] = row
                del call, theta
                clear()
                save()

    # Ratios: per seed, method variant over analytic variant; geometric mean over seeds.
    ratios = {}
    analytic_selected = out['selected']['analytic']
    for method in spec['candidates']:
        if method == 'analytic':
            continue
        for label, mine, base in (('matched_eager', 'eager', 'eager'), ('best_vs_best', out['selected'][method], analytic_selected)):
            per_seed = {}
            for seed in spec['timing_seeds']:
                rows = out['timing'][str(seed)]
                a, b = rows.get(f'{method}:{mine}', {}), rows.get(f'analytic:{base}', {})
                if 'median_seconds_per_evaluation' in a and 'median_seconds_per_evaluation' in b:
                    per_seed[str(seed)] = a['median_seconds_per_evaluation'] / b['median_seconds_per_evaluation']
            ratios[f'{method}:{label}'] = {'variant': mine, 'analytic_variant': base, 'per_seed': per_seed,
                                           'geometric_mean': float(np.exp(np.mean(np.log(list(per_seed.values()))))) if per_seed else None}
    out['ratios'] = ratios
    # Feasibility rule: projected solve seconds with the selected variant.
    feasibility = {}
    for method in spec['candidates']:
        variant = out['selected'][method]
        seconds = [out['timing'][str(s)].get(f'{method}:{variant}', {}).get('median_seconds_per_evaluation') for s in spec['timing_seeds']]
        seconds = [s for s in seconds if s is not None]
        if not seconds:
            feasibility[method] = {'state': 'NO_TIMED_VARIANT'}
            continue
        projected = spec['solve_evaluations'] * max(seconds) + spec['observation_and_setup_seconds']
        feasibility[method] = {'variant': variant, 'projected_solve_seconds': projected,
                               'fits_one_job': projected <= spec['job_seconds_available']}
    out['feasibility'] = feasibility
    out['state'] = 'COMPLETE'
    save()
    print(json.dumps({'selected': out['selected'], 'ratios': {k: v['geometric_mean'] for k, v in ratios.items()}, 'feasibility': feasibility}, indent=1))


if __name__ == '__main__':
    main()
