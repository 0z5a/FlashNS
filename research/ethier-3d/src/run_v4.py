"""Cluster-side driver for the frozen Ethier v4 batch on an existing allocation.

Runs once: fresh admission checks, original GPU lane locks, 18 fresh solve
processes (6 seeds x 3 arms) on two lanes, lock release, independent NumPy
audit, archive and archive verification. No retries; every outcome is kept.
"""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(argv, **kw):
    cp = subprocess.run(argv, capture_output=True, text=True, **kw)
    return {'argv': argv, 'exit': cp.returncode, 'stdout': cp.stdout, 'stderr': cp.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--lane', action='append', required=True,
                        help='GPU_UUID:LOCK_PATH:LOCK_INODE (exactly two)')
    parser.add_argument('--admission-budget-minutes', type=float, default=165.0)
    parser.add_argument('--self-test', action='store_true',
                        help='CPU smoke of the whole driver flow; skips job/node/NVML checks')
    args = parser.parse_args()
    root = args.root.resolve()
    control = root / 'control'
    evidence = root / 'evidence'
    control.mkdir(exist_ok=True)
    assert not evidence.exists(), 'evidence directory exists: this batch runs once'
    evidence.mkdir()
    started = time.time()
    state = {'state': 'STARTING', 'driver_pid': os.getpid(), 'started_unix': started,
             'host': socket.gethostname(),
             'boot_id': (Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                         if Path('/proc/sys/kernel/random/boot_id').exists() else None),
             'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
             'slurm_step_id': os.environ.get('SLURM_STEP_ID'), 'checks': {}}
    lock = threading.Lock()

    def save(name='driver-state.json'):
        with lock:
            tmp = control / (name + '.tmp')
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(control / name)

    def finish(final_state, code):
        state['state'] = final_state
        state['finished_unix'] = time.time()
        save()
        return code

    # 1. Fresh admission: job, node, sources, locks, idle GPUs.
    state['self_test'] = args.self_test
    state['checks']['job_and_node'] = args.self_test or (state['slurm_job_id'] == args.job
                                                         and state['host'] == args.node)
    if not state['checks']['job_and_node']:
        return finish('ADMISSION_REFUSED_JOB_OR_NODE', 2)
    source = json.loads((root / 'manifest.json').read_text())
    bad = [n for n, v in source['files'].items() if sha256(root / n) != v['sha256']]
    state['checks']['source_files'] = {'verified': len(source['files']), 'mismatches': bad}
    if bad:
        return finish('ADMISSION_REFUSED_SOURCE_MISMATCH', 2)
    lanes = []
    for spec in args.lane:
        uuid, path, inode = spec.split(':')
        lanes.append({'uuid': uuid, 'lock_path': path, 'lock_inode': int(inode)})
    assert len(lanes) == 2
    held = []
    for lane in lanes:
        fd = os.open(lane['lock_path'], os.O_RDONLY)
        lane['observed_inode'] = os.fstat(fd).st_ino
        if lane['observed_inode'] != lane['lock_inode']:
            os.close(fd)
            lane['acquired'] = False
            break
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            lane['acquired'] = False
            break
        lane['acquired'] = True
        lane['acquired_unix'] = time.time()
        held.append(fd)
    state['lanes'] = lanes
    if len(held) != 2:
        for fd in held:
            os.close(fd)
        return finish('ADMISSION_REFUSED_LOCK_HELD_OR_INODE_CHANGED', 2)
    smi = {'exit': 0, 'stdout': '\n'.join(l['uuid'] + ', 0, 0' for l in lanes)} if args.self_test else run(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
               '--format=csv,noheader,nounits'])
    apps = {'exit': 0, 'stdout': ''} if args.self_test else run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
    state['checks']['nvml_before'] = {'gpus': smi, 'apps': apps}
    busy = []
    for lane in lanes:
        row = [r for r in smi['stdout'].splitlines() if r.startswith(lane['uuid'])]
        if (len(row) != 1 or int(row[0].split(',')[1]) > 100
                or lane['uuid'] in apps['stdout']):
            busy.append(lane['uuid'])
    state['checks']['lanes_idle'] = not busy
    if smi['exit'] or apps['exit'] or busy:
        for fd in held:
            os.close(fd)
        return finish('ADMISSION_REFUSED_GPU_NOT_IDLE', 2)
    save()

    # 2. Frozen balanced plan.
    protocol = json.loads((root / 'protocol-v4.json').read_text())
    seeds = protocol['initialization_seeds']
    orders = list(itertools.permutations(protocol['arms']))
    plan = [[] for _ in lanes]
    for i, seed in enumerate(seeds):
        plan[i % 2].extend({'id': f'{arm}-{seed}', 'arm': arm, 'seed': seed}
                           for arm in orders[i])
    state['plan'] = plan
    controller = {'state': 'RUNNING', 'driver_pid': os.getpid(), 'started_unix': started,
                  'children': []}
    deadline = started + 60 * args.admission_budget_minutes

    def save_controller():
        with lock:
            tmp = evidence / 'controller.json.tmp'
            tmp.write_text(json.dumps(controller, indent=2))
            tmp.replace(evidence / 'controller.json')

    def lane_worker(index):
        lane = lanes[index]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if args.self_test else lane['uuid'],
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=str(root) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        for case in plan[index]:
            if time.time() > deadline:
                with lock:
                    controller.setdefault('not_started_after_admission_budget', []).append(case)
                continue
            out = evidence / case['id']
            command = [sys.executable, str(root / 'solve.py'), '--arm', case['arm'],
                       '--seed', str(case['seed']), '--output', str(out)]
            command += (['--device', 'cpu', '--cpu-smoke'] if args.self_test else
                        ['--device', 'cuda:0', '--expected-gpu-uuid', lane['uuid']])
            with open(evidence / (case['id'] + '.log'), 'w') as log:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
                record = {**case, 'lane': index, 'gpu_uuid': lane['uuid'], 'pid': child.pid,
                          'command': command, 'started_unix': time.time()}
                with lock:
                    controller['children'].append(record)
                save_controller()
                code = child.wait()
            record.update(actual_exit_code=code, finished_unix=time.time(),
                          absent_after_wait=not Path(f'/proc/{child.pid}').exists())
            save_controller()

    threads = [threading.Thread(target=lane_worker, args=(i,)) for i in range(2)]
    state['state'] = 'RUNNING_GPU_CASES'
    save()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    controller['state'] = 'CHILDREN_JOINED'
    save_controller()

    # 3. Release the original lane locks before CPU-only audit and archive.
    state['checks']['nvml_after'] = None if args.self_test else {
        'gpus': run(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
                     '--format=csv,noheader,nounits']),
        'apps': run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                     '--format=csv,noheader,nounits'])}
    for fd, lane in zip(held, lanes):
        os.close(fd)
        lane['released_unix'] = time.time()
    controller['state'] = 'LOCKS_RELEASED'
    save_controller()
    state['state'] = 'LOCKS_RELEASED_AUDITING'
    save()

    # 4. Raw-file manifest and independent NumPy audit (CPU only).
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(evidence.rglob('*'))
             if p.is_file() and not p.name.endswith('.tmp')}
    (root / 'offbox-manifest.json').write_text(json.dumps({'files': files}, indent=2))
    audit_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                     PYTHONPATH=str(root) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    with open(control / 'audit.log', 'w') as log:
        audit_code = subprocess.run([sys.executable, str(root / 'audit.py'), '--root', str(root),
                                     '--output', str(root / 'audit-v4.json')],
                                    cwd=root, env=audit_env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
    state['audit_exit_code'] = audit_code
    save()

    # 5. Complete archive with member and whole-file verification.
    durable = root / 'durable'
    durable.mkdir(exist_ok=True)
    archive = durable / 'ethier-v4-complete.tar.gz'
    state['state'] = 'ARCHIVING'
    save()
    members = sorted(str(p.relative_to(root)) for p in root.rglob('*')
                     if p.is_file() and durable not in p.parents and not p.name.endswith('.tmp'))
    with tarfile.open(archive, 'w:gz') as tar:
        for name in members:
            tar.add(root / name, arcname=name)
    digest = sha256(archive)
    with tarfile.open(archive, 'r:gz') as tar:
        names = tar.getnames()
        member_ok = all(hashlib.sha256(tar.extractfile(n).read()).hexdigest() == sha256(root / n)
                        for n in names if n != 'control/driver-state.json')
    state['archive'] = {'path': str(archive), 'sha256': digest, 'bytes': archive.stat().st_size,
                        'members': len(names), 'members_match_sources': member_ok,
                        'whole_file_sha_reread': sha256(archive) == digest}
    (durable / 'ethier-v4-complete.tar.gz.sha256').write_text(f'{digest}  {archive.name}\n')
    codes = [c.get('actual_exit_code') for c in controller['children']]
    state['children'] = {'started': len(codes), 'exit_zero': codes.count(0),
                         'not_started': len(controller.get('not_started_after_admission_budget', []))}
    ok = audit_code == 0 and member_ok and state['archive']['whole_file_sha_reread']
    return finish('COMPLETE' if ok else 'COMPLETE_WITH_AUDIT_OR_ARCHIVE_FAILURE', 0 if ok else 1)


if __name__ == '__main__':
    sys.exit(main())
