"""Cluster-side driver of the 2D representation/execution ablation, run once in its own allocation.

Admission (job/node, source and tool manifests, plan hash, GPU lock inodes, idle GPUs, formal
initialization hashes, a git checkout of the pinned commit), then four lanes pull fresh-process
tasks (ablation_bench.py) from one pre-randomized queue with a deadline guard and a fail-fast
guard, then lock release, the reader (read_ablation.py) and a verified archive. Each task's
lane, timing, exit code and log are recorded. No retries.
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


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='work dir with source/, ablation/, native-cache/')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--lane', action='append', required=True, help='GPU_UUID:LOCK_PATH:LOCK_INODE')
    parser.add_argument('--python', required=True)
    parser.add_argument('--self-test', action='store_true', help='CPU control flow: tasks run a stub instead of bench')
    args = parser.parse_args()
    root = args.root.resolve()
    ablation, source = root / 'ablation', root / 'source'
    control, results = ablation / 'control', ablation / 'results'
    control.mkdir(exist_ok=True)
    assert not results.exists(), 'results exist: this batch runs once'
    results.mkdir()
    plan_path = ablation / args.plan
    plan = json.loads(plan_path.read_text())
    protocol = json.loads((ablation / plan['protocol']).read_text())
    state = {'state': 'STARTING', 'driver_pid': os.getpid(), 'started_unix': time.time(), 'host': socket.gethostname(),
             'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'self_test': args.self_test, 'checks': {}, 'tasks': {}}
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

    state['checks']['job_and_node'] = args.self_test or (state['slurm_job_id'] == args.job and state['host'] == args.node)
    manifest = json.loads((ablation / 'manifest.json').read_text())
    bad = [n for n, v in manifest['files'].items() if sha256(root / n) != v['sha256']]
    state['checks']['manifest'] = {'verified': len(manifest['files']), 'mismatches': bad}
    state['checks']['plan_protocol'] = plan['protocol_sha256'] == sha256(ablation / plan['protocol'])
    # bench_gpu records `git rev-parse HEAD` of the source tree; it must be the pinned commit.
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True, text=True)
    state['checks']['source_commit'] = head.stdout.strip()
    pinned = head.returncode == 0 and head.stdout.strip() == protocol['source']['commit']
    if not state['checks']['job_and_node'] or bad or not state['checks']['plan_protocol'] or not pinned:
        return finish('ADMISSION_REFUSED', 2)
    env_base = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                    PYTHONPATH=os.pathsep.join([str(source / 'src'), os.environ.get('PYTHONPATH', '')]),
                    FLASHNS_CUDA_CACHE=str(root / 'native-cache'))
    # Formal initializations and data must match the frozen H20 hashes (CPU, before any GPU work).
    check = subprocess.run([args.python, '-c', (
        'import json,hashlib,sys; sys.path.insert(0,"experiments/pinn_solver"); import problem; '
        'p=problem.Problem(interior_count=2048,device="cpu"); '
        'print(json.dumps({"theta":{s:hashlib.sha256(problem.parameters(s).numpy().tobytes()).hexdigest() for s in %r},'
        '"meta":{k:v for k,v in p.meta.items() if k.endswith("sha256")}}))' % (protocol['solve_seeds'],))],
        cwd=source, env=env_base, capture_output=True, text=True)
    observed = json.loads(check.stdout.strip().splitlines()[-1]) if check.returncode == 0 else {}
    expected = protocol['frozen_inputs']
    state['checks']['frozen_inputs'] = {
        'exit': check.returncode, 'stderr_tail': check.stderr[-500:],
        'theta_match': {str(s): observed.get('theta', {}).get(str(s)) == expected['theta_sha256'][str(s)] for s in protocol['solve_seeds']},
        'data_match': {k: observed.get('meta', {}).get(k) == v for k, v in expected['data_sha256'].items()}}
    if check.returncode or not all(state['checks']['frozen_inputs']['theta_match'].values()) \
            or not all(state['checks']['frozen_inputs']['data_match'].values()):
        return finish('ADMISSION_REFUSED_FROZEN_INPUTS', 2)
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
    if not args.self_test:
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                              capture_output=True, text=True)
        state['checks']['gpus_idle'] = apps.returncode == 0 and not any(l['uuid'] in apps.stdout for l in lanes)
        if not state['checks']['gpus_idle']:
            for f in held:
                os.close(f)
            return finish('ADMISSION_REFUSED_GPU_NOT_IDLE', 2)
    save()

    work = queue.Queue()
    for t in plan['tasks']:
        work.put(t)
    durations = {}
    # Fail fast: if the first FAIL_FAST finished tasks all failed, stop starting new ones (a systematic
    # start-up error would otherwise run through the whole plan); skipped tasks are recorded.
    FAIL_FAST = 8
    outcomes = []
    defaults = protocol['default_task_estimate_seconds']
    bench = source / 'experiments' / 'contracted_jets'

    def estimate(t):
        key = (t['kind'], t['cell'], t['interior'], tuple(t['hidden']))
        return 1.25 * max(durations.get(key, [defaults[t['kind']]]))

    def worker(index):
        lane = lanes[index]
        cache = root / 'cache' / f'lane{index}'
        cache.mkdir(parents=True, exist_ok=True)
        env = dict(env_base, CUDA_VISIBLE_DEVICES='' if args.self_test else lane['uuid'],
                   TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
                   XDG_CACHE_HOME=str(cache / 'xdg'), TMPDIR=str(cache), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        while True:
            try:
                t = work.get_nowait()
            except queue.Empty:
                return
            record = state['tasks'][t['id']] = {'lane': index, 'tier': t['tier']}
            with lock:
                stop = len(outcomes) >= FAIL_FAST and not any(outcomes)
            if stop:
                record['state'] = 'NOT_RUN_FAIL_FAST'
                save()
                continue
            if time.time() + estimate(t) > args.deadline_unix:
                record['state'] = 'NOT_RUN_DEADLINE'
                save()
                continue
            out = results / f"{t['id']}.json"
            argv = [args.python, str(bench / 'ablation_bench.py'), '--hidden', ','.join(map(str, t['hidden'])),
                    '--backend', t['backend'], '--interior-count', str(t['interior']), '--seed', str(t['seed']),
                    '--output', str(out)]
            if t['eager']:
                argv.append('--eager')
            if t['kind'] == 'solve':
                argv.append('--solve')
            else:
                argv += ['--samples', '7', '--replays', '100', '--warmup-seconds', '2']
                if t['profile']:
                    argv.append('--profile')
            if args.self_test:
                argv = [args.python, '-c', 'import json,sys; json.dump({"status":"passed","self_test":True,"argv":sys.argv[1:]},'
                        'open(sys.argv[1],"w"))', str(out), *argv[2:]]
            record.update(argv=argv, started_unix=time.time())
            save()
            with open(control / f"{t['id']}.log", 'w') as log:
                try:
                    code = subprocess.run(argv, cwd=bench, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=protocol['task_timeout_seconds']).returncode
                except subprocess.TimeoutExpired:
                    code = 'TIMEOUT'
            record.update(exit=code, finished_unix=time.time(), seconds=time.time() - record['started_unix'],
                          state='COMPLETE' if code == 0 else 'FAILED')
            key = (t['kind'], t['cell'], t['interior'], tuple(t['hidden']))
            durations.setdefault(key, []).append(record['seconds'])
            with lock:
                outcomes.append(code == 0)
            save()

    state['state'] = 'RUNNING'
    save()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(lanes))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if not args.self_test:
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                              capture_output=True, text=True)
        state['checks']['gpus_idle_after'] = not any(l['uuid'] in apps.stdout for l in lanes)
    for fd, lane in zip(held, lanes):
        os.close(fd)
        lane['released_unix'] = time.time()
    state['state'] = 'LOCKS_RELEASED_READING'
    counts = {}
    for r in state['tasks'].values():
        counts[r.get('state')] = counts.get(r.get('state'), 0) + 1
    state['counts'] = counts
    save()
    with open(control / 'reader.log', 'w') as log:
        state['reader_exit'] = subprocess.run(
            [args.python, str(ablation / 'read_ablation.py'), '--plan', str(plan_path), '--protocol', str(ablation / plan['protocol']),
             '--results', str(results), '--state', str(control / 'driver-state.json'), '--output', str(ablation / 'ablation-result.json')]
            + (['--self-test'] if args.self_test else []),
            cwd=ablation, env=env_base, stdout=log, stderr=subprocess.STDOUT).returncode
    save()
    durable = ablation / 'durable'
    durable.mkdir(exist_ok=True)
    archive = durable / '2d-ablation-complete.tar.gz'
    state['state'] = 'ARCHIVING'
    save()
    members = sorted(str(p.relative_to(root)) for p in ablation.rglob('*')
                     if p.is_file() and durable not in p.parents and not p.name.endswith('.tmp'))
    members += [str((root / 'native-build.json').relative_to(root))] if (root / 'native-build.json').exists() else []
    with tarfile.open(archive, 'w:gz') as tar:
        for name in members:
            tar.add(root / name, arcname=name)
    digest = sha256(archive)
    with tarfile.open(archive, 'r:gz') as tar:
        names = tar.getnames()
        member_ok = all(hashlib.sha256(tar.extractfile(n).read()).hexdigest() == sha256(root / n)
                        for n in names if not n.endswith('control/driver-state.json'))
    state['archive'] = {'path': str(archive), 'sha256': digest, 'bytes': archive.stat().st_size, 'members': len(names),
                        'members_match_sources': member_ok, 'whole_file_sha_reread': sha256(archive) == digest}
    (durable / (archive.name + '.sha256')).write_text(f'{digest}  {archive.name}\n')
    ok = state['reader_exit'] == 0 and member_ok and state['archive']['whole_file_sha_reread']
    return finish('COMPLETE' if ok else 'COMPLETE_WITH_READER_OR_ARCHIVE_FAILURE', 0 if ok else 1)


if __name__ == '__main__':
    sys.exit(main())
