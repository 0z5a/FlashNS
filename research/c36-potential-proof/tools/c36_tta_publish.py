"""Publish the C36 formal TTA cohort record from its sealed per-seed archives.

    PYTHONPATH=src python3 tools/c36_tta_publish.py --root ROOT --output OUT.json

ROOT holds packet/PROTOCOL (default protocol-formal-tta-20261021.json), archives/seed{seed}-job*.tar.gz with their
receipts, and audit/ with the frozen auditor's outputs. Every archive member read is checked
against its receipt. For each registered seed the record gives the training summary, every
physical observation (box376.decide_c36_v6.over_limit) and the frozen audit's state; the
registered cohort classification is box376.tta_control.cohort_result over the audit entries
exactly as recorded. A seed without an audit pass stays in the denominator. Standard library and
the frozen modules only; no paths or scheduler identifiers are written.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from box376.decide_c36_v6 import over_limit
from box376.tta_control import cohort_result

KEEP = ('velocity_relative_l2', 'normalized_corrected_pressure_rms', 'normalized_momentum_rms',
        'normalized_momentum_gradient_rms', 'normalized_divergence_rms')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation(report, protocol):
    row = {'state': report['state'], 'physical_acceptance_pass': report.get('physical_acceptance_pass'),
           'seconds': report['finished_unix'] - report['started_unix'] if report.get('finished_unix') else None,
           'phase': report.get('phase'), 'observer_chunk': report.get('observer_chunk')}
    if report['state'] != 'COMPLETE':
        return row
    reader_row = {'datasets': {n: {'global': d['global']} for n, d in report['datasets'].items()},
                  'initial_boundary_normalized_velocity_rms': [report['initial_normalized_velocity_rms'], report['boundary_normalized_velocity_rms']],
                  'trajectory': report['trajectories'][-1]}
    factor, worst, failing = over_limit(reader_row, protocol)
    row.update(worst_over_limit=factor, worst_test=worst, failing_tests=failing,
               datasets={name: {'field_gate_pass': d['field_gate_pass'], 'global': {k: d['global'].get(k) for k in KEEP}}
                         for name, d in report['datasets'].items()},
               initial_boundary_normalized_velocity_rms=reader_row['initial_boundary_normalized_velocity_rms'],
               trajectories=report['trajectories'])
    return row


def audit_state(audit, stem):
    passed, failed = audit / (stem + '-independent-readback.json'), audit / (stem + '-independent-readback-failure.json')
    if passed.is_file():
        record = json.loads(passed.read_text())
        return {'state': record['state'], 'output_sha256': sha(passed)}, record['cohort_entry']
    if failed.is_file():
        record = json.loads(failed.read_text())
        last = record['traceback'].strip().splitlines()[-1]
        return {'state': record['state'], 'exit': record['exit'], 'failure': last,
                'frozen_audit_script_sha256': record['frozen_audit_script_sha256']}, record['cohort_entry']
    return {'state': 'NOT_RUN'}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol-name', default='protocol-formal-tta-20261021.json')
    args = parser.parse_args()
    protocol_path = args.root / 'packet' / args.protocol_name
    protocol = json.loads(protocol_path.read_text())
    record = {'protocol_version': protocol['version'], 'protocol_sha256': sha(protocol_path),
              'eta': protocol['instance']['eta'], 'holdout_seed': protocol['physical']['holdout_seed'], 'seeds': {}}
    entries = []
    for seed in protocol['seeds']:
        [archive] = sorted((args.root / 'archives').glob(f'seed{seed}-job*.tar.gz'))
        receipt = json.loads(archive.with_name(archive.name[:-len('.tar.gz')] + '.json').read_text())
        assert sha(archive) == receipt['sha256']
        with tarfile.open(archive, 'r:gz') as t:
            def read(name):
                data = t.extractfile(name).read()
                assert hashlib.sha256(data).hexdigest() == receipt['member_sha256'][name], name
                return json.loads(data)
            train = read('run/train/report.json')
            observations = {name.split('/')[1][len('physical-'):]: observation(read(name), protocol)
                            for name in sorted(receipt['member_sha256']) if name.startswith('run/physical-') and name.endswith('/report.json')}
        gate = train.get('training_gate_first')
        audit, entry = audit_state(args.root / 'audit', archive.name[:-len('.tar.gz')])
        if entry is not None:
            entries.append(entry)
        record['seeds'][str(seed)] = {
            'archive_sha256': receipt['sha256'], 'archive_members': len(receipt['member_sha256']),
            'training_state': train['state'], 'stop_reason': train.get('stop_reason'),
            'setup_seconds': train['setup_seconds'], 'Adam_updates': train['Adam_updates'],
            'SSBroyden_iterations': train['SSBroyden_iterations'], 'SSBroyden_evaluations': train['SSBroyden_evaluations'],
            'resets': train['resets'], 'skipped_updates': train['skipped_updates'],
            'training_gate_first': None if gate is None else {k: gate.get(k) for k in ('phase', 'iteration', 'evaluation', 'loss', 'gradient_infinity', 'training_seconds')},
            'training_seconds': train.get('training_seconds'), 'final_loss': train.get('final_loss'),
            'final_gradient_infinity': train.get('final_gradient_infinity'), 'wall_seconds': train.get('wall_seconds'),
            'observer_seconds_excluded': train.get('observer_seconds_excluded'),
            'inspections': [{k: o.get(k) for k in ('tag', 'active_training_seconds')} for o in train.get('observations', [])],
            'candidate_tta': train.get('tta'),
            'checks': {c['tag']: max(x['maximum_error_over_original_budget'] for x in c['comparisons']) for c in train['checks']},
            'observations': observations, 'frozen_audit': audit}
    record['registered_cohort_result'] = cohort_result(protocol['seeds'], entries)
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + '\n')
    for seed, run in record['seeds'].items():
        tta = run['candidate_tta'] or {}
        print(seed, run['training_state'], run['SSBroyden_iterations'], tta.get('state'), tta.get('time_to_verified_accuracy_s'),
              {t: (o['state'], o.get('worst_test'), o.get('worst_over_limit')) for t, o in run['observations'].items()}, run['frozen_audit']['state'])
    print(record['registered_cohort_result']['state'], record['registered_cohort_result']['passes'], '/', record['registered_cohort_result']['denominator'])


if __name__ == '__main__':
    main()
