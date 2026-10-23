"""Cluster-side driver for the C36 v3 smoothness-schedule batch in its own allocation.

Runs once: admission, lane locks, per lane a compiler smoke and one training run
(seed and continuation schedule, box376.potential_train_v3), then lock release, the
independent NumPy reader (box376.read_c36_v3) and a verified archive. Physical
observation of the checkpoints runs in a separate all-planes batch. No retries.
"""
import argparse
import fcntl
import hashlib
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
    parser.add_argument('--root', type=Path, required=True, help='packet root with src/, data/, protocol')
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--compiler-runtime', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--lane', action='append', required=True,
                        help='GPU_UUID:LOCK_PATH:LOCK_INODE:SEED:SCHEDULE')
    parser.add_argument('--deadline-unix', type=float, default=None,
                        help='allocation end minus the reserve needed for reader and archive')
    parser.add_argument('--self-test', action='store_true',
                        help='CPU control-flow test: skips job/node/NVML checks; children run --self-test')
    args = parser.parse_args()
    root = args.root.resolve()
    control, runs = root / 'control', root / 'runs'
    control.mkdir(exist_ok=True)
    assert not runs.exists(), 'runs directory exists: this batch runs once'
    runs.mkdir()
    protocol_path = root / args.protocol
    protocol = json.loads(protocol_path.read_text())
    started = time.time()
    state = {'state': 'STARTING', 'driver_pid': os.getpid(), 'started_unix': started,
             'host': socket.gethostname(), 'self_test': args.self_test,
             'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'checks': {}, 'lanes': []}
    lock = threading.Lock()

    def save():
        with lock:
            tmp = control / 'driver-state.json.tmp'
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(control / 'driver-state.json')

    def finish(final_state, code):
        state.update(state=final_state, finished_unix=time.time())
        save()
        return code

    state['checks']['job_and_node'] = args.self_test or (
        state['slurm_job_id'] == args.job and state['host'] == args.node)
    if not state['checks']['job_and_node']:
        return finish('ADMISSION_REFUSED_JOB_OR_NODE', 2)
    manifest = json.loads((root / 'manifest.json').read_text())
    bad = [n for n, v in manifest['files'].items() if sha256(root / n) != v['sha256']]
    state['checks']['source_files'] = {'verified': len(manifest['files']), 'mismatches': bad}
    if bad:
        return finish('ADMISSION_REFUSED_SOURCE_MISMATCH', 2)
    lanes, held = [], []
    for spec in args.lane:
        uuid, path, inode, seed, schedule = spec.split(':')
        lane = {'uuid': uuid, 'lock_path': path, 'lock_inode': int(inode), 'seed': int(seed), 'schedule': schedule}
        lanes.append(lane)
        fd = os.open(path, os.O_RDONLY)
        lane['observed_inode'] = os.fstat(fd).st_ino
        try:
            assert lane['observed_inode'] == lane['lock_inode']
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (AssertionError, BlockingIOError):
            os.close(fd)
            lane['acquired'] = False
            break
        lane.update(acquired=True, acquired_unix=time.time())
        held.append(fd)
    state['lanes'] = lanes
    if len(held) != len(args.lane) or len(lanes) != len(args.lane):
        for fd in held:
            os.close(fd)
        return finish('ADMISSION_REFUSED_LOCK_HELD_OR_INODE_CHANGED', 2)
    if not args.self_test:
        smi = run(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
        apps = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
        state['checks']['nvml_before'] = {'gpus': smi, 'apps': apps}
        busy = [l['uuid'] for l in lanes
                if l['uuid'] in apps['stdout'] or not any(
                    r.startswith(l['uuid']) and int(r.split(',')[1]) <= 100 for r in smi['stdout'].splitlines())]
        state['checks']['lanes_idle'] = not busy
        if smi['exit'] or apps['exit'] or busy:
            for fd in held:
                os.close(fd)
            return finish('ADMISSION_REFUSED_GPU_NOT_IDLE', 2)
    save()

    reference_ok = args.self_test or sha256(args.reference) == '172174ea49a82a21e0616571ae303560d275317d8b5319c22ea49dbd628d8b66'
    state['checks']['reference_sha256'] = reference_ok
    if not reference_ok:
        for fd in held:
            os.close(fd)
        return finish('ADMISSION_REFUSED_REFERENCE_MISMATCH', 2)

    def lane_worker(index):
        lane = lanes[index]
        out = runs / f"seed{lane['seed']}-{lane['schedule']}"
        out.mkdir()
        cache = root / 'cache' / f'lane{index}'
        cache.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if args.self_test else lane['uuid'],
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
                   TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
                   TORCHINDUCTOR_COMPILE_THREADS='1', XDG_CACHE_HOME=str(cache / 'xdg'),
                   TMPDIR=str(cache), OMP_NUM_THREADS='4',
                   PYTHONPATH=os.pathsep.join([str(root / 'src'), args.compiler_runtime,
                                               os.environ.get('PYTHONPATH', '')]))
        steps = lane['steps'] = []

        def step(name, argv):
            record = {'name': name, 'argv': argv, 'started_unix': time.time()}
            steps.append(record)
            save()
            with open(out / f'{name}.log', 'w') as log:
                child = subprocess.Popen(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                record['pid'] = child.pid
                save()
                record['exit'] = child.wait()
            record['finished_unix'] = time.time()
            save()
            return record['exit']

        py = sys.executable
        compat = [] if args.self_test else ['--legacy-triton-target']
        if not args.self_test:
            if step('compiler-smoke', [py, '-m', 'box376.packed_analytic_gpu_smoke',
                                       '--output', str(out / 'compiler-smoke')] + compat) != 0:
                lane['outcome'] = 'COMPILER_SMOKE_FAILED'
                return
        train = [py, '-m', 'box376.potential_train_v3', '--inputs', str(root / 'data'),
                 '--protocol', str(protocol_path), '--seed', str(lane['seed']), '--schedule', lane['schedule'],
                 '--output', str(out / 'train')] + compat
        train += ['--self-test'] if args.self_test else ['--reference', str(args.reference)]
        lane['training_exit'] = step('train', train)
        lane['outcome'] = 'LANE_COMPLETE'

    threads = [threading.Thread(target=lane_worker, args=(i,)) for i in range(len(lanes))]
    state['state'] = 'RUNNING_LANES'
    save()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not args.self_test:
        state['checks']['nvml_after'] = {
            'gpus': run(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits']),
            'apps': run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])}
    for fd, lane in zip(held, lanes):
        os.close(fd)
        lane['released_unix'] = time.time()
    state['state'] = 'LOCKS_RELEASED_READING'
    save()

    reader_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                      PYTHONPATH=os.pathsep.join([str(root / 'src'), os.environ.get('PYTHONPATH', '')]))
    with open(control / 'reader.log', 'w') as log:
        state['reader_exit'] = subprocess.run(
            [sys.executable, '-m', 'box376.read_c36_v3', '--runs', str(runs), '--protocol', str(protocol_path),
             '--output', str(root / 'c36-v3-smoothness-result.json')],
            cwd=root, env=reader_env, stdout=log, stderr=subprocess.STDOUT).returncode
    save()

    durable = root / 'durable'
    durable.mkdir(exist_ok=True)
    archive = durable / 'c36-v3-smoothness-complete.tar.gz'
    state['state'] = 'ARCHIVING'
    save()
    skip = (durable, root / 'cache')
    members = sorted(str(p.relative_to(root)) for p in root.rglob('*')
                     if p.is_file() and not any(s == p or s in p.parents for s in skip)
                     and not p.name.endswith('.tmp') and not p.name.startswith('._'))
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
    (durable / (archive.name + '.sha256')).write_text(f'{digest}  {archive.name}\n')
    ok = state['reader_exit'] == 0 and member_ok and state['archive']['whole_file_sha_reread']
    return finish('COMPLETE' if ok else 'COMPLETE_WITH_READER_OR_ARCHIVE_FAILURE', 0 if ok else 1)


if __name__ == '__main__':
    sys.exit(main())
