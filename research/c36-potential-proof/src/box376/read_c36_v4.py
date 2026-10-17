"""Independent NumPy reader for the C36 v4 resumed-L-BFGS (combination) batch.

As the v3 reader, with the v4 loss thresholds and checkpoint calls taken from the protocol;
evaluation counts and seconds are measured from the resume point. It also records the
start checkpoint, the continuity check and any truncation. Physical observation runs in
a separate batch and the registered decision is made by box376.decide_c36_v4.

Rechecks every saved complete-gradient comparison, file hash and event count,
recomputes the first training-gate closure from the raw event log, and summarizes
the loss curve and every physical observation. It does not rerun training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_seed(directory, protocol):
    train = directory / 'train'
    report = json.loads((train / 'report.json').read_text())
    out = {'seed': report['seed'], 'start': report.get('start'), 'schedule': report.get('schedule'),
           'engine': report.get('engine'), 'start_checkpoint': report.get('start_checkpoint'),
           'continuity': report.get('continuity'), 'training_state': report['state'],
           'training_seconds': report.get('training_seconds'),
           'Adam_updates': report['Adam_updates'], 'LBFGS_calls': report['LBFGS_calls'],
           'LBFGS_closures': report['LBFGS_closures'], 'budget_truncated': report['budget_truncated'],
           'initialize_matches_stored_bytes': report.get('initialize_matches_stored_bytes')}
    if report['state'] != 'COMPLETE':
        # A failed child records its traceback; a killed child stops at its last progress write.
        out['failure_tail'] = report.get('failure', 'no traceback: process ended before its final report')[-2000:]
        out['last_phase'] = report.get('phase')
    # Saved complete-gradient comparisons.
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
    for checkpoint in report['checkpoints']:
        assert sha(train / checkpoint['filename']) == checkpoint['sha256'], checkpoint
    out['checkpoints'] = [{k: c.get(k) for k in ('tag', 'LBFGS_calls', 'LBFGS_closures', 'training_seconds')}
                          for c in report['checkpoints']]
    # Raw event log. Its hash and counts are sealed only when the child reached its final report.
    events = [json.loads(line) for line in (train / 'events.jsonl').read_text().splitlines() if line.strip()]
    adam = [e for e in events if e['phase'] == 'Adam']
    lbfgs = [e for e in events if e['phase'] == 'LBFGS']
    out['events_sealed'] = 'events_sha256' in report
    if out['events_sealed']:
        assert sha(train / 'events.jsonl') == report['events_sha256']
        assert len(adam) == report['Adam_updates'] and len(lbfgs) == report['LBFGS_closures']
    if not events:
        out['loss_curve'] = None
        out['training_gate_first'] = None
        out['physical'] = {}
        return out
    spec = protocol['training']
    original = [i for i, e in enumerate(events) if e.get('width', .5) == .5]
    gate = next((events[i] for i in original if events[i]['loss'] <= spec['loss_convergence_limit']
                 and events[i]['gradient_infinity'] <= spec['gradient_infinity_convergence_limit']), None)
    reported = report['training_gate_first']
    assert (gate is None) == (reported is None) or not out['events_sealed']
    if gate is not None and reported is not None:
        assert gate['loss'] == reported['loss'] and gate['phase'] == reported['phase']
    out['training_gate_first'] = gate
    losses = np.array([e['loss'] for e in events])
    seconds = np.array([e['training_seconds'] for e in events])
    first = {}
    for threshold in protocol['training']['loss_thresholds']:
        hit = [i for i in original if losses[i] <= threshold]
        first[f'{threshold:g}'] = None if not len(hit) else {
            'evaluation': int(hit[0]) + 1, 'training_seconds': float(seconds[hit[0]])}
    out['loss_curve'] = {
        'evaluations': len(events), 'first_original_problem_evaluation': (original[0] + 1) if original else None,
        'initial': float(losses[0]),
        'after_Adam': float(adam[-1]['loss']) if adam else None,
        'minimum': float(losses.min()), 'final_reported': report.get('final_loss'),
        'final_gradient_infinity': report.get('final_gradient_infinity'),
        'first_evaluation_at_or_below': first,
        'mean_seconds_per_evaluation': float(seconds[-1] / len(events)),
        'per_LBFGS_call_end': {str(c): float(min(e['loss'] for e in lbfgs if e['call'] == c))
                               for c in sorted({e['call'] for e in lbfgs}) if c == 1 or c in protocol['training']['checkpoint_LBFGS_calls']}}
    # Physical observations.
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
    runs = [read_seed(d, protocol) for d in sorted(args.runs.glob('seed*')) if (d / 'train' / 'report.json').exists()]
    result = {'state': 'C36_V4_COMBINATION_TRAINING_READ', 'protocol_version': protocol['version'],
              'protocol_sha256': sha(args.protocol), 'runs': runs,
              'all_complete': bool(runs) and all(r['training_state'] == 'COMPLETE' for r in runs),
              'any_budget_truncated': any(r['budget_truncated'] for r in runs),
              'decision': 'not made here: the registered decision needs the separate physical observations '
                          '(box376.decide_c36_v4)',
              'formal_claim': False}
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('state', 'all_complete', 'any_budget_truncated')}))


if __name__ == '__main__':
    main()
