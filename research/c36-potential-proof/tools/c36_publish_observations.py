"""Publishable summary of every C36 all-planes observation (development and smoothness runs).

    python3 c36_publish_observations.py OUT.json WIDTHS.json LABEL=BATCH_ROOT[,CHECKPOINT_ROOT] ...

BATCH_ROOT is an observation packet root (with observe-plan, control/driver-state.json,
observations/ and observation-summary.json). Every observation report and reader audit is
read directly; paths are reduced to batch labels and packet-relative names. Pure standard
library.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

FIELD_LIMITS = {'velocity_relative_l2': .01, 'normalized_divergence_rms': .001, 'normalized_momentum_rms': .01,
                'normalized_momentum_gradient_rms': .02, 'normalized_divergence_gradient_rms': .02,
                'normalized_corrected_pressure_rms': .001}
REGION_KEYS = ('velocity_relative_l2', 'normalized_corrected_pressure_rms', 'normalized_momentum_rms')
REGIONS = ('plateau', 'face', 'edge', 'corner', 'outside')
# Physical acceptance gates the stratified and uniform holdouts, initial and boundary values and
# the finest trajectory refinement; the seven time slices are diagnostic planes with the same limits.
GATED = ('stratified', 'uniform')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slope(points):
    """Least-squares slope of log10(metric) against log10(loss)."""
    if len(points) < 2:
        return None
    xs = [math.log10(a) for a, _ in points]
    ys = [math.log10(b) for _, b in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def main():
    out_path, widths = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    batches, observations = [], {}
    for spec in sys.argv[3:]:
        label, root = spec.split('=', 1)
        root = Path(root)
        plan_path = root / 'observe-plan-20261015.json'
        plan = json.loads(plan_path.read_text())
        state = json.loads((root / 'control' / 'driver-state.json').read_text())
        summary = json.loads((root / 'observation-summary.json').read_text())
        calibration = state.get('calibration', {}).get('chosen', {})
        archive = state.get('archive') or {}
        batches.append({
            'label': label, 'job': state.get('slurm_job_id'), 'driver_state': state['state'],
            'plan_version': plan['version'], 'plan_sha256': sha(plan_path),
            'manifest_sha256': sha(root / 'manifest.json'), 'protocol_sha256': plan['protocol_sha256'],
            'observer_chunk': summary['observer_chunk'],
            'calibration': {k: calibration.get(k) for k in ('chunk', 'seconds_per_point', 'peak_allocated_bytes',
                                                             'max_abs_difference_vs_first', 'max_relative_difference_vs_first')},
            'tasks': len(plan['tasks']), 'counts': summary['counts'], 'audits_passed': summary['audits_passed'],
            'archive': {k: archive.get(k) for k in ('sha256', 'bytes', 'members', 'members_match_sources', 'whole_file_sha_reread')},
            'locks_released': all(lane.get('released_unix') is not None for lane in state.get('lanes', []))})
        rows = {row['id']: row for row in summary['rows']}
        for task in plan['tasks']:
            row = rows[task['id']]
            if row.get('observation_state') != 'COMPLETE':
                continue
            assert task['id'] not in observations, ('observed twice', task['id'])
            report = json.loads((root / 'observations' / task['id'] / 'report.json').read_text())
            audit = json.loads((root / 'observations' / (task['id'] + '-reader') / 'audit.json').read_text())
            planes = {}
            for name, dataset in report['datasets'].items():
                g = dataset['global']
                planes[name] = {'gated': name in GATED, 'pass': dataset['field_gate_pass'], 'global': g,
                                'over_limit': {k: g[k] / v for k, v in FIELD_LIMITS.items()
                                               if isinstance(g.get(k), (int, float)) and not isinstance(g.get(k), bool) and g[k] > v}}
                if name == 'stratified':
                    planes[name]['regions'] = {REGIONS[int(r)]: {k: v.get(k) for k in REGION_KEYS}
                                               for r, v in dataset['regions'].items()}
            initial, boundary = report['initial_normalized_velocity_rms'], report['boundary_normalized_velocity_rms']
            planes['initial'] = {'gated': True, 'pass': initial <= 1e-3, 'normalized_velocity_rms': initial}
            planes['boundary'] = {'gated': True, 'pass': boundary <= 1e-3, 'normalized_velocity_rms': boundary}
            finest = report['trajectories'][-1]
            planes['trajectory'] = {'gated': True, 'pass': finest['maximum_endpoint_error'] <= 1e-3 and finest['maximum_refinement_change'] <= 1e-7,
                                    'refinements': report['trajectories']}
            gated = {k: p for k, p in planes.items() if p['gated']}
            assert all(p['pass'] for p in gated.values()) == report['physical_acceptance_pass'], task['id']
            observations[task['id']] = {
                'id': task['id'], 'batch': label, 'run': task['run'], 'tags': row['tags'],
                'width': widths.get(task['id'], .5), 'checkpoint_sha256': task['checkpoint_sha256'],
                'theta_sha256': task['theta_sha256'], 'Adam_updates': row['Adam_updates'],
                'LBFGS_calls': row['LBFGS_calls'], 'LBFGS_closures': row['LBFGS_closures'],
                'loss_at_checkpoint': row.get('loss_at_checkpoint'),
                'gradient_infinity_at_checkpoint': row.get('gradient_infinity_at_checkpoint'),
                'observer_chunk': summary['observer_chunk'], 'seconds': row.get('seconds'),
                'physical_acceptance_pass': report['physical_acceptance_pass'],
                'acceptance_tests_passed': sum(p['pass'] for p in planes.values() if p['gated']),
                'acceptance_tests': sum(p['gated'] for p in planes.values()),
                'diagnostic_planes_passed': sum(p['pass'] for p in planes.values() if not p['gated']),
                'diagnostic_planes': sum(not p['gated'] for p in planes.values()),
                'failed_planes': sorted(k for k, p in planes.items() if not p['pass']),
                'reader': {'status': audit.get('status'), 'comparisons': audit.get('comparisons')},
                'reproduction': row.get('reproduction'), 'planes': planes}
    rows = sorted(observations.values(), key=lambda r: (r['run'], r['Adam_updates'], r['LBFGS_closures']))
    trends = {}
    for run in sorted({r['run'] for r in rows}):
        mine = [r for r in rows if r['run'] == run and r['width'] == .5 and r['loss_at_checkpoint'] is not None]
        trend = {'checkpoints': [r['tags'][0].split(':')[1] for r in mine]}
        for key in ('velocity_relative_l2', 'normalized_corrected_pressure_rms'):
            series = [(r['loss_at_checkpoint'], r['planes']['stratified']['global'][key]) for r in mine]
            lbfgs = [(a, b) for (a, b), r in zip(series, mine) if r['LBFGS_closures'] > 0]
            rising = [(mine[i - 1]['tags'][0].split(':')[1], mine[i]['tags'][0].split(':')[1])
                      for i in range(1, len(series)) if series[i][1] > series[i - 1][1]]
            trend[key] = {'first': series[0][1], 'last': series[-1][1], 'minimum': min(b for _, b in series),
                          'last_over_previous': series[-1][1] / series[-2][1] if len(series) > 1 else None,
                          'log_log_slope_LBFGS': slope(lbfgs), 'increases_between_consecutive_checkpoints': rising}
        trends[run] = trend
    result = {
        'scope': 'Development observations of C36 checkpoints with the unchanged observer on all data planes; '
                 'physical acceptance gates the stratified and uniform holdouts, initial and boundary values and the '
                 'finest trajectory refinement, while the seven time slices are diagnostic planes. Loss is the last logged training evaluation at the '
                 'checkpoint (the objective of its stage width), not a recomputation at the saved parameters.',
        'batches': batches,
        'totals': {'observations': len(rows), 'unique_parameter_vectors': len({r['theta_sha256'] for r in rows}),
                   'acceptance_tests': sum(r['acceptance_tests'] for r in rows),
                   'acceptance_tests_passed': sum(r['acceptance_tests_passed'] for r in rows),
                   'diagnostic_planes': sum(r['diagnostic_planes'] for r in rows),
                   'diagnostic_planes_passed': sum(r['diagnostic_planes_passed'] for r in rows),
                   'reader_pass': sum(r['reader']['status'] == 'PASS' for r in rows),
                   'reader_comparisons': sum(r['reader']['comparisons'] or 0 for r in rows),
                   'physical_acceptance_pass': sum(r['physical_acceptance_pass'] for r in rows),
                   'failed_plane_sets': sorted({tuple(r['failed_planes']) for r in rows})},
        'trends_original_problem_checkpoints': trends, 'observations': rows}
    out_path.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    t = result['totals']
    print(f"{t['observations']} observations, {t['unique_parameter_vectors']} parameter vectors, acceptance tests "
          f"{t['acceptance_tests_passed']}/{t['acceptance_tests']}, diagnostic planes {t['diagnostic_planes_passed']}/{t['diagnostic_planes']}, reader PASS {t['reader_pass']} ({t['reader_comparisons']} comparisons), "
          f"accepted {t['physical_acceptance_pass']}")
    for fails in t['failed_plane_sets']:
        print(' failed set', fails, sum(tuple(r['failed_planes']) == fails for r in rows))


if __name__ == '__main__':
    main()
