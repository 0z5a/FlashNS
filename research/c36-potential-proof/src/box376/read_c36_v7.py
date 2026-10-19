"""Independent NumPy reader for the C36 v7 optimizer batch.

The read_c36_v2 reader adapted to the v7 event log: it rechecks every saved complete-gradient
comparison, every checkpoint and the iteration-log hash, counts Adam and SSBroyden events against
the report, recomputes the first training-gate evaluation from the raw events, summarizes the loss
curve (first evaluation at or below each protocol threshold, counting every evaluation from the
first Adam update as v5 does), the optimizer record (iterations, line-search evaluations, resets,
skipped updates, seconds per iteration, the carried inverse-Hessian product checks) and every
physical observation. It does not rerun training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(directory, protocol):
    train = directory / 'train'
    report = json.loads((train / 'report.json').read_text())
    out = {'seed': report['seed'], 'arm': report.get('arm'), 'training_state': report['state'],
           'training_seconds': report.get('training_seconds'), 'Adam_updates': report['Adam_updates'],
           'SSBroyden_iterations': report['SSBroyden_iterations'], 'SSBroyden_evaluations': report['SSBroyden_evaluations'],
           'resets': report['resets'], 'skipped_updates': report['skipped_updates'], 'stop_reason': report.get('stop_reason'),
           'budget_truncated': report['budget_truncated'], 'continuity': report.get('continuity'),
           'inverse_hessian_memory': report.get('inverse_hessian_memory'),
           'final_loss': report.get('final_loss'), 'final_gradient_infinity': report.get('final_gradient_infinity')}
    if report['state'] != 'COMPLETE':
        out['failure_tail'] = report.get('failure', 'no traceback: process ended before its final report')[-2000:]
        out['last_phase'] = report.get('phase')
    worst, checks = 0.0, 0
    for row in report['checks']:
        path = train / row['filename']
        assert sha(path) == row['sha256'], path
        with np.load(path) as arrays:
            for comparison in row['comparisons']:
                a = arrays[comparison['name'] + '_actual']
                b = arrays[comparison['name'] + '_reference']
                error = float(np.max(np.abs(a - b) / (protocol['atol'] + protocol['rtol'] * np.abs(b))))
                assert error <= 1 and abs(error - comparison['maximum_error_over_original_budget']) <= 1e-9 * max(1.0, error)
                worst, checks = max(worst, error), checks + 1
    out['gradient_checks'] = {'saved_comparisons': checks, 'check_points': len(report['checks']),
                              'maximum_error_over_original_budget': worst}
    product = [c['relative_difference'] for c in report.get('inverse_hessian_product_checks', [])]
    out['inverse_hessian_product_checks'] = {'count': len(product), 'maximum_relative_difference': max(product) if product else None}
    for checkpoint in report['checkpoints']:
        assert sha(train / checkpoint['filename']) == checkpoint['sha256'], checkpoint
    out['checkpoints'] = [{k: c[k] for k in ('tag', 'training_seconds')} for c in report['checkpoints']]
    events = [json.loads(line) for line in (train / 'events.jsonl').read_text().splitlines() if line.strip()]
    adam = [e for e in events if e['phase'] == 'Adam']
    quasi = [e for e in events if e['phase'] == 'SSBroyden']
    assert len(adam) + len(quasi) == len(events)
    out['events_sealed'] = 'events_sha256' in report
    if out['events_sealed']:
        assert sha(train / 'events.jsonl') == report['events_sha256']
        assert len(adam) == report['Adam_updates'] and len(quasi) == report['SSBroyden_evaluations']
    iterations_path = train / 'iterations.jsonl'
    iterations = [json.loads(line) for line in iterations_path.read_text().splitlines() if line.strip()] if iterations_path.exists() else []
    if 'iterations_sha256' in report:
        assert sha(iterations_path) == report['iterations_sha256'] and len(iterations) == report['SSBroyden_iterations']
    if iterations:
        updated = [r for r in iterations if 'tau' in r]
        out['optimizer'] = {
            'iterations': len(iterations),
            'line_search_evaluations_mean': float(np.mean([r['line_search_evaluations'] for r in iterations])),
            'seconds_per_iteration_mean': float(np.mean([r['seconds'] for r in iterations])),
            'tau_range': [min(r['tau'] for r in updated), max(r['tau'] for r in updated)] if updated else None,
            'phi_range': [min(r['phi'] for r in updated), max(r['phi'] for r in updated)] if updated else None,
            'resets': sum('reset' in r for r in iterations), 'skipped_updates': sum('update_skipped' in r for r in iterations),
            'step_length_median': float(np.median([r['step_length'] for r in iterations]))}
    if not events:
        out['loss_curve'], out['training_gate_first'], out['physical'] = None, None, {}
        return out
    spec = protocol['training']
    gate = next((e for e in events if e['loss'] <= spec['loss_convergence_limit']
                 and e['gradient_infinity'] <= spec['gradient_infinity_convergence_limit']), None)
    reported = report['training_gate_first']
    assert (gate is None) == (reported is None) or not out['events_sealed']
    if gate is not None and reported is not None:
        assert gate['loss'] == reported['loss'] and gate['phase'] == reported['phase']
    out['training_gate_first'] = gate
    losses = np.array([e['loss'] for e in events])
    seconds = np.array([e['training_seconds'] for e in events])
    first = {}
    for threshold in spec['loss_thresholds']:
        hit = np.flatnonzero(losses <= threshold)
        first[f'{threshold:g}'] = None if not len(hit) else {'evaluation': int(hit[0]) + 1, 'training_seconds': float(seconds[hit[0]])}
    out['loss_curve'] = {'evaluations': len(events), 'initial': float(losses[0]),
                         'after_Adam': float(adam[-1]['loss']) if adam else None,
                         'first_SSBroyden': float(quasi[0]['loss']) if quasi else None,
                         'minimum': float(losses.min()), 'final_reported': report.get('final_loss'),
                         'first_evaluation_at_or_below': first,
                         'mean_seconds_per_evaluation': float(seconds[-1] / len(events))}
    out['physical'] = {}
    for path in sorted(directory.glob('physical-*/report.json')):
        physical = json.loads(path.read_text())
        tag = path.parent.name[len('physical-'):]
        row = {'state': physical['state'], 'physical_acceptance_pass': physical.get('physical_acceptance_pass')}
        if physical['state'] == 'COMPLETE':
            assert sha(path.parent / 'raw-physical.npz') == physical['raw_sha256']
            row['datasets'] = {name: {'field_gate_pass': d['field_gate_pass'], 'global': d['global']}
                               for name, d in physical['datasets'].items()}
            row['initial_boundary_normalized_velocity_rms'] = [
                physical['initial_normalized_velocity_rms'], physical['boundary_normalized_velocity_rms']]
            row['trajectory'] = physical['trajectories'][-1]
            row['observation_seconds'] = physical['finished_unix'] - physical['started_unix']
        out['physical'][tag] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    runs = [read_run(d, protocol) for d in sorted(args.runs.glob('seed*')) if (d / 'train' / 'report.json').exists()]
    result = {'state': 'C36_V7_DEVELOPMENT_READ', 'protocol_version': protocol['version'],
              'protocol_sha256': sha(args.protocol), 'runs': runs,
              'any_training_gate': any(r['training_gate_first'] is not None for r in runs),
              'any_physical_acceptance': any(any(p.get('physical_acceptance_pass') for p in r['physical'].values()) for r in runs),
              'formal_claim': False}
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('state', 'any_training_gate', 'any_physical_acceptance')}))


if __name__ == '__main__':
    main()
