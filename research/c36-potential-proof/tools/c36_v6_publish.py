"""Publish the C36 v6 training and observation record from a finished batch.

    PYTHONPATH=src python3 tools/c36_v6_publish.py --runs RUNS --protocol PROTOCOL --output OUT.json

Reads each lane's train/report.json and physical-*/report.json and writes, per run, the training
summary (gate-first event, final loss, timing, every saved check's worst error over the original
budget) and, per observation, acceptance, every failing test with its value over limit
(box376.decide_c36_v6.over_limit), the global, regional and time-slice metrics, the initial and
boundary velocity and the trajectory refinements. Standard library only; no paths are written.
"""
import argparse
import json
from pathlib import Path

from box376.decide_c36_v6 import over_limit

REGIONS = ('plateau', 'face', 'edge', 'corner', 'outside')
KEEP = ('velocity_relative_l2', 'normalized_corrected_pressure_rms', 'normalized_momentum_rms',
        'normalized_momentum_gradient_rms', 'normalized_divergence_rms')


def observation(report, protocol):
    row = {'state': report['state'], 'physical_acceptance_pass': report.get('physical_acceptance_pass'),
           'seconds': report.get('finished_unix', 0) - report['started_unix'] if report.get('finished_unix') else None,
           'observer_chunk': report.get('observer_chunk'), 'holdout_eta': report['validation_metadata']['eta']}
    if report['state'] != 'COMPLETE':
        return row
    reader_row = {'datasets': {n: {'global': d['global']} for n, d in report['datasets'].items()},
                  'initial_boundary_normalized_velocity_rms': [report['initial_normalized_velocity_rms'], report['boundary_normalized_velocity_rms']],
                  'trajectory': report['trajectories'][-1]}
    factor, worst, failing = over_limit(reader_row, protocol)
    row.update(worst_over_limit=factor, worst_test=worst, failing_tests=failing,
               datasets={name: {'field_gate_pass': d['field_gate_pass'], 'global': {k: d['global'].get(k) for k in KEEP},
                                'regions': {REGIONS[int(r)]: {k: m.get(k) for k in KEEP} for r, m in d.get('regions', {}).items()}}
                         for name, d in report['datasets'].items()},
               initial_boundary_normalized_velocity_rms=reader_row['initial_boundary_normalized_velocity_rms'],
               trajectories=report['trajectories'])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    record = {'protocol_version': protocol['version'], 'eta': protocol['instance']['eta'], 'runs': {}}
    for run in sorted(p for p in args.runs.iterdir() if p.is_dir()):
        train = json.loads((run / 'train' / 'report.json').read_text())
        gate = train.get('training_gate_first')
        record['runs'][run.name] = {
            'engine': train['engine'], 'seed': train['seed'], 'training_state': train['state'],
            'Adam_updates': train['Adam_updates'], 'LBFGS_calls': train['LBFGS_calls'], 'LBFGS_closures': train['LBFGS_closures'],
            'training_seconds': train.get('training_seconds'), 'budget_truncated': train['budget_truncated'],
            'final_loss': train.get('final_loss'), 'final_gradient_infinity': train.get('final_gradient_infinity'),
            'training_gate_first': None if gate is None else {k: gate.get(k) for k in ('phase', 'call', 'closure', 'loss', 'gradient_infinity', 'training_seconds')},
            'setup_seconds': train['setup_seconds'],
            'checks': {c['tag']: max(x['maximum_error_over_original_budget'] for x in c['comparisons']) for c in train['checks']},
            'observations': {p.name[len('physical-'):]: observation(json.loads((p / 'report.json').read_text()), protocol)
                             for p in sorted(run.glob('physical-*')) if (p / 'report.json').is_file()}}
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + '\n')
    for name, run in record['runs'].items():
        final = run['observations'].get('final', {})
        print(name, run['training_gate_first'] and run['training_gate_first']['call'], round(run['final_loss'], 9),
              final.get('worst_test'), round(final.get('worst_over_limit', 0), 3))


if __name__ == '__main__':
    main()
