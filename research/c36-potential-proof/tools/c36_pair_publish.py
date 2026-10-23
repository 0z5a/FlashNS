"""Publish the paired C36 engine cohort record from its sealed per-seed archives.

    PYTHONPATH=src python3 tools/c36_pair_publish.py --root ROOT --output OUT.json

ROOT holds packet/protocol-formal-pair-20261021.json, archives/seed{seed}-job*.tar.gz with their
receipts, and audit/ with the auditor's outputs. Every archive member read is checked against its
receipt. Per seed and arm the record gives the threshold measurement, per-evaluation seconds in each
phase, setup, the physical observation (box376.decide_c36_v6.over_limit) and the audit state. The
registered decision is box376.decide_c36_pair over the audit entries exactly as recorded. Standard
library and the frozen modules only; no paths or scheduler identifiers are written.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from box376.decide_c36_pair import decide, pair_entry
from box376.decide_c36_v6 import over_limit

PROTOCOL = 'protocol-formal-pair-20261021.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def per_evaluation(events, phase):
    rows = [e for e in events if e['phase'] == phase]
    if len(rows) < 2:
        return None
    return (rows[-1]['training_seconds'] - rows[0]['training_seconds']) / (len(rows) - 1)


def observation(report, protocol):
    row = {'state': report['state'], 'physical_acceptance_pass': report.get('physical_acceptance_pass'),
           'seconds': report['finished_unix'] - report['started_unix'] if report.get('finished_unix') else None}
    if report['state'] == 'COMPLETE':
        reader_row = {'datasets': {n: {'global': d['global']} for n, d in report['datasets'].items()},
                      'initial_boundary_normalized_velocity_rms': [report['initial_normalized_velocity_rms'], report['boundary_normalized_velocity_rms']],
                      'trajectory': report['trajectories'][-1]}
        factor, worst, failing = over_limit(reader_row, protocol)
        g = report['datasets']['stratified']['global']
        row.update(worst_over_limit=factor, worst_test=worst, failing_tests=failing,
                   stratified_velocity_relative_l2=g['velocity_relative_l2'],
                   stratified_corrected_pressure_over_limit=g['normalized_corrected_pressure_rms'] / protocol['physical']['normalized_corrected_pressure_rms_limit'],
                   trajectory_endpoint_over_limit=report['trajectories'][-1]['maximum_endpoint_error'] / protocol['physical']['maximum_trajectory_endpoint_error'])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.root / 'packet' / PROTOCOL
    protocol = json.loads(protocol_path.read_text())
    record = {'protocol_version': protocol['version'], 'protocol_sha256': sha(protocol_path),
              'threshold_loss': protocol['pair']['threshold_loss'], 'prediction': protocol['pair']['prediction'], 'seeds': {}}
    entries = []
    for seed in protocol['seeds']:
        [archive] = sorted((args.root / 'archives').glob(f'seed{seed}-job*.tar.gz'))
        stem = archive.name[:-len('.tar.gz')]
        receipt = json.loads((args.root / 'archives' / (stem + '.json')).read_text())
        assert sha(archive) == receipt['sha256']
        row = {'archive_sha256': receipt['sha256'], 'archive_members': len(receipt['member_sha256']),
               'arm_order': protocol['pair']['arm_order'][str(seed)], 'arms': {}}
        with tarfile.open(archive, 'r:gz') as t:
            def read(name, raw=False):
                data = t.extractfile(name).read()
                assert hashlib.sha256(data).hexdigest() == receipt['member_sha256'][name], name
                return data if raw else json.loads(data)
            for arm in row['arm_order']:
                prefix = f'run/{arm}/'
                if prefix + 'train/report.json' not in receipt['member_sha256']:
                    row['arms'][arm] = {'training_state': 'MISSING'}
                    continue
                train = read(prefix + 'train/report.json')
                events = [json.loads(l) for l in read(prefix + 'train/events.jsonl', raw=True).decode().splitlines() if l.strip()]
                physical = prefix + 'physical-threshold/report.json'
                row['arms'][arm] = {
                    'training_state': train['state'], 'stop_reason': train.get('stop_reason'), 'pair': train.get('pair'),
                    'setup_seconds': train['setup_seconds'], 'Adam_updates': train['Adam_updates'],
                    'SSBroyden_iterations': train['SSBroyden_iterations'], 'SSBroyden_evaluations': train['SSBroyden_evaluations'],
                    'resets': train['resets'], 'skipped_updates': train['skipped_updates'],
                    'Adam_seconds_per_evaluation': per_evaluation(events, 'Adam'),
                    'SSBroyden_seconds_per_evaluation': per_evaluation(events, 'SSBroyden'),
                    'checks': {c['tag']: max(x['maximum_error_over_original_budget'] for x in c['comparisons']) for c in train['checks']},
                    'peak_allocated_bytes': train.get('peak_allocated_bytes'),
                    'observation': observation(read(physical), protocol) if physical in receipt['member_sha256'] else None}
        audit = args.root / 'audit' / (stem + '-independent-readback.json')
        if audit.is_file():
            result = json.loads(audit.read_text())
            row['audit'] = {'state': result['state'], 'output_sha256': sha(audit)}
            entries.append(pair_entry(seed, result['cohort_entry']['arms']))
        else:
            row['audit'] = {'state': 'NOT_PASSED_OR_NOT_RUN'}
        record['seeds'][str(seed)] = row
    record['registered_decision'] = decide(protocol['seeds'], entries)
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + '\n')
    for seed, row in record['seeds'].items():
        print(seed, row['audit']['state'], {a: ((r.get('pair') or {}).get('state'), (r.get('pair') or {}).get('active_training_seconds'),
                                                (r.get('observation') or {}).get('physical_acceptance_pass')) for a, r in row['arms'].items()})
    d = record['registered_decision']
    print(d['outcome'], d['jointly_accepted_pairs'], d.get('geometric_mean_ratio'), d.get('range'))


if __name__ == '__main__':
    main()
