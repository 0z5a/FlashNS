"""Observe every unique C36 v2 checkpoint on all physical holdout planes, once.

Runs in its own allocation: admission, observer chunk calibration, a work queue of
box376.potential_physical observations over the GPU lanes (deadline-guarded with measured
durations), the independent NumPy reader (box376.audit_potential_run) on every completed
observation, a reproduction check of previously observed checkpoints, a summary, and a
verified archive. The observer and the reader are used unchanged; only --observer-chunk
is chosen by the frozen calibration rule.
"""
import argparse
import fcntl
import hashlib
import json
import os
import queue
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ratio(actual, reference):
    a, b = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    return float(np.max(np.abs(a - b) / (1e-11 + 1e-9 * np.abs(b))))


def choose_chunk(calibration, rule):
    """Fastest calibrated chunk within the memory limit and numerical agreement bound."""
    bound = rule['max_abs_difference_over_reference_max'] * calibration['reference_max_abs']
    accepted = [row for row in calibration['rows'] if 'failed' not in row
                and (row['peak_allocated_bytes'] or 0) <= rule['max_peak_allocated_bytes']
                and (row['chunk'] == calibration['reference_chunk'] or row['max_abs_difference_vs_first'] <= bound)]
    return min(accepted, key=lambda row: row['seconds_per_point'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='new run root (packet copy with src/, data/, plan)')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--checkpoints', type=Path, required=True, help='v2 packet root with runs/')
    parser.add_argument('--job', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--lane', action='append', required=True, help='GPU_UUID:LOCK_PATH:LOCK_INODE')
    parser.add_argument('--self-test', action='store_true', help='CPU flow test; observations are copied, not run')
    args = parser.parse_args()
    root = args.root.resolve()
    control, out = root / 'control', root / 'observations'
    control.mkdir(exist_ok=True)
    assert not out.exists(), 'observations exist: this batch runs once'
    out.mkdir()
    plan = json.loads((root / args.plan).read_text())
    protocol_path = args.checkpoints / plan['protocol']
    assert sha256(protocol_path) == plan['protocol_sha256']
    state = {'state': 'STARTING', 'driver_pid': os.getpid(), 'started_unix': time.time(),
             'host': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
             'self_test': args.self_test, 'checks': {}, 'tasks': {}}
    lock = threading.Lock()

    def save():
        with lock:
            tmp = control / 'driver-state.json.tmp'
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(control / 'driver-state.json')

    def finish(final, code):
        state.update(state=final, finished_unix=time.time())
        save()
        return code

    # Admission.
    state['checks']['job_and_node'] = args.self_test or (state['slurm_job_id'] == args.job and state['host'] == args.node)
    manifest = json.loads((root / 'manifest.json').read_text())
    bad = [n for n, v in manifest['files'].items() if sha256(root / n) != v['sha256']]
    state['checks']['source_files'] = {'verified': len(manifest['files']), 'mismatches': bad}
    tasks = plan['tasks']
    missing = [t['id'] for t in tasks if sha256(args.checkpoints / t['checkpoint']) != t['checkpoint_sha256']]
    state['checks']['checkpoints'] = {'verified': len(tasks), 'mismatches': missing}
    if not state['checks']['job_and_node'] or bad or missing:
        return finish('ADMISSION_REFUSED', 2)
    lanes, held = [], []
    for spec in args.lane:
        uuid, path, inode = spec.split(':')
        fd = os.open(path, os.O_RDONLY)
        lane = {'uuid': uuid, 'lock_path': path, 'lock_inode': int(inode), 'observed_inode': os.fstat(fd).st_ino}
        lanes.append(lane)
        try:
            assert lane['observed_inode'] == lane['lock_inode']
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (AssertionError, BlockingIOError):
            os.close(fd)
            for f in held:
                os.close(f)
            state['lanes'] = lanes
            return finish('ADMISSION_REFUSED_LOCK', 2)
        lane['acquired_unix'] = time.time()
        held.append(fd)
    state['lanes'] = lanes
    save()

    py = sys.executable
    base_env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='4', PYTHONPATH=os.pathsep.join([str(root / 'src'), os.environ.get('PYTHONPATH', '')]))

    def lane_env(lane):
        return dict(base_env, CUDA_VISIBLE_DEVICES='' if args.self_test else lane['uuid'])

    def run(name, argv, env, cwd=root):
        with open(control / f'{name}.log', 'w') as log:
            return subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode

    # Calibration on the first lane.
    state['state'] = 'CALIBRATING'
    save()
    calibration_dir = root / 'calibration'
    calibration_code = run('calibration', [py, '-m', 'box376.observer_calibration', '--protocol', str(protocol_path),
                           '--checkpoint', str(args.checkpoints / plan['calibration_checkpoint']),
                           '--output', str(calibration_dir)]
                           + (['--points', '8', '--chunks', '4,8', '--device', 'cpu'] if args.self_test
                              else ['--points', str(plan['calibration']['points']), '--chunks', plan['calibration']['chunks']]),
                           lane_env(lanes[0]))
    calibration = json.loads((calibration_dir / 'calibration.json').read_text()) if calibration_code == 0 else None
    if calibration is None:
        chosen = {'chunk': plan['calibration']['fallback_chunk'], 'seconds_per_point': plan['calibration']['fallback_seconds_per_point']}
    else:
        chosen = choose_chunk(calibration, plan['calibration']['rule'])
    state['calibration'] = {'exit': calibration_code, 'chosen': chosen}
    chunk = chosen['chunk']
    estimate = [chosen['seconds_per_point'] * plan['observed_points_per_observation'] * plan['calibration']['estimate_margin']
                + plan['calibration']['estimate_fixed_seconds']]
    if args.self_test:
        estimate = [0.0]  # observations are copied, so the CPU calibration does not predict their cost
    save()

    # Observation work queue.
    work = queue.Queue()
    for task in tasks:
        work.put(task)
    audits = []

    def audit(task, directory):
        record = state['tasks'][task['id']]
        reader = directory.parent / (directory.name + '-reader')
        (reader / 'training').mkdir(parents=True)
        code = subprocess.run([py, '-c', 'import sys,torch,numpy as np; c=torch.load(sys.argv[1],map_location="cpu",weights_only=True); '
                               'np.savez(sys.argv[2], theta=c["theta"].numpy())',
                               str(args.checkpoints / task['checkpoint']),
                               str(reader / 'training' / f"seed{task['seed']}-final-gradient.npz")], env=base_env).returncode
        if code == 0:
            with open(reader / 'audit.log', 'w') as log:
                code = subprocess.run([py, '-m', 'box376.audit_potential_run', '--proof', str(directory),
                                       '--inputs', str(args.checkpoints / 'data'), '--protocol', str(protocol_path),
                                       '--training', str(reader / 'training'), '--output', str(reader / 'audit.json')],
                                      env=dict(base_env, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT).returncode
        record['audit_exit'] = code
        save()

    def worker(index):
        lane = lanes[index]
        while True:
            try:
                task = work.get_nowait()
            except queue.Empty:
                return
            record = state['tasks'][task['id']] = {'lane': index, 'checkpoint': task['checkpoint']}
            if time.time() + max(estimate) > args.deadline_unix:
                record['state'] = 'NOT_RUN_DEADLINE'
                save()
                continue
            directory = out / task['id']
            record['started_unix'] = time.time()
            save()
            if args.self_test:
                source = args.checkpoints / task.get('self_test_copy', '')
                if not task.get('self_test_copy') or not source.exists():
                    record['state'] = 'NOT_RUN_SELF_TEST'
                    save()
                    continue
                subprocess.run(['cp', '-r', str(source), str(directory)], check=True)
                code = 0
            else:
                code = run('observe-' + task['id'], [py, '-m', 'box376.potential_physical', '--inputs', str(args.checkpoints / 'data'),
                           '--protocol', str(protocol_path), '--checkpoint', str(args.checkpoints / task['checkpoint']),
                           '--output', str(directory), '--observer-chunk', str(chunk)], lane_env(lane))
            record.update(exit=code, finished_unix=time.time(), seconds=time.time() - record['started_unix'])
            if code == 0 and not args.self_test:
                estimate.append(record['seconds'] * plan['calibration']['estimate_margin'])
            record['state'] = 'OBSERVED' if code == 0 else 'OBSERVER_FAILED'
            save()
            if code == 0:
                thread = threading.Thread(target=audit, args=(task, directory))
                thread.start()
                audits.append(thread)

    state['state'] = 'OBSERVING'
    save()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(lanes))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for fd, lane in zip(held, lanes):
        os.close(fd)
        lane['released_unix'] = time.time()
    state['state'] = 'LOCKS_RELEASED_AUDITING'
    save()
    for thread in audits:
        thread.join()

    # Summary: metrics, audits, reproduction and checkpoint losses.
    import torch
    summary = {'protocol_sha256': plan['protocol_sha256'], 'observer_chunk': chunk, 'calibration': calibration, 'rows': []}
    for task in tasks:
        record = state['tasks'].get(task['id'], {})
        row = {k: task[k] for k in ('id', 'run', 'seed', 'engine', 'tags', 'checkpoint_sha256', 'theta_sha256')}
        row.update(state=record.get('state'), seconds=record.get('seconds'), audit_exit=record.get('audit_exit'))
        checkpoint = torch.load(args.checkpoints / task['checkpoint'], map_location='cpu', weights_only=True)
        position = checkpoint['Adam_updates'] + checkpoint['LBFGS_closures']
        events = (args.checkpoints / 'runs' / task['run'] / 'train' / 'events.jsonl').read_text().splitlines()
        if position > 0:
            event = json.loads(events[position - 1])
            row['loss_at_checkpoint'] = event['loss']
            row['gradient_infinity_at_checkpoint'] = event['gradient_infinity']
        row.update(Adam_updates=checkpoint['Adam_updates'], LBFGS_calls=checkpoint['LBFGS_calls'],
                   LBFGS_closures=checkpoint['LBFGS_closures'])
        report_path = out / task['id'] / 'report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            row['observation_state'] = report['state']
            if report['state'] == 'COMPLETE':
                row['physical_acceptance_pass'] = report['physical_acceptance_pass']
                row['datasets'] = {k: {'field_gate_pass': v['field_gate_pass'], 'global': v['global'], 'regions': v['regions']}
                                   for k, v in report['datasets'].items()}
                row['initial_boundary_normalized_velocity_rms'] = [report['initial_normalized_velocity_rms'],
                                                                   report['boundary_normalized_velocity_rms']]
                row['trajectory'] = report['trajectories'][-1]
                audit_path = out / (task['id'] + '-reader') / 'audit.json'
                if audit_path.exists():
                    row['audit'] = json.loads(audit_path.read_text())
                if task.get('reproduce_report'):
                    previous = json.loads((args.checkpoints / task['reproduce_report']).read_text())
                    worst = 0.0
                    for name, d in report['datasets'].items():
                        for key, value in d['global'].items():
                            other = previous['datasets'][name]['global'][key]
                            if isinstance(value, bool) or value is None or other is None:
                                assert value == other, (name, key)
                            elif isinstance(value, list):
                                worst = max(worst, ratio([v if v is not None else 0.0 for v in value],
                                                         [v if v is not None else 0.0 for v in other]))
                            else:
                                worst = max(worst, ratio(value, other))
                    for key in ('maximum_endpoint_error', 'maximum_refinement_change'):
                        worst = max(worst, ratio(report['trajectories'][-1][key], previous['trajectories'][-1][key]))
                    row['reproduction'] = {'reference': task['reproduce_report'], 'max_ratio_over_original_budget': worst,
                                           'within_original_budget': worst <= 1,
                                           'acceptance_identical': report['physical_acceptance_pass'] == previous['physical_acceptance_pass']}
        summary['rows'].append(row)
    counts = {}
    for row in summary['rows']:
        counts[row['state'] or 'NONE'] = counts.get(row['state'] or 'NONE', 0) + 1
    summary['counts'] = counts
    summary['audits_passed'] = sum(1 for r in summary['rows'] if r.get('audit', {}).get('status') == 'PASS')
    (root / 'observation-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    state['summary_counts'] = counts
    state['audits_passed'] = summary['audits_passed']
    save()

    # Archive.
    durable = root / 'durable'
    durable.mkdir(exist_ok=True)
    archive = durable / 'c36-observe-complete.tar.gz'
    state['state'] = 'ARCHIVING'
    save()
    members = sorted(str(p.relative_to(root)) for p in root.rglob('*')
                     if p.is_file() and durable not in p.parents and not p.name.endswith('.tmp') and not p.name.startswith('._'))
    with tarfile.open(archive, 'w:gz') as tar:
        for name in members:
            tar.add(root / name, arcname=name)
    digest = sha256(archive)
    with tarfile.open(archive, 'r:gz') as tar:
        names = tar.getnames()
        member_ok = all(hashlib.sha256(tar.extractfile(n).read()).hexdigest() == sha256(root / n)
                        for n in names if n != 'control/driver-state.json')
    state['archive'] = {'path': str(archive), 'sha256': digest, 'bytes': archive.stat().st_size, 'members': len(names),
                        'members_match_sources': member_ok, 'whole_file_sha_reread': sha256(archive) == digest}
    (durable / (archive.name + '.sha256')).write_text(f'{digest}  {archive.name}\n')
    ok = member_ok and state['archive']['whole_file_sha_reread']
    return finish('COMPLETE' if ok else 'COMPLETE_WITH_ARCHIVE_FAILURE', 0 if ok else 1)


if __name__ == '__main__':
    sys.exit(main())
