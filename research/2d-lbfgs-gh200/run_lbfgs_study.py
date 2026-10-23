"""Cluster-side driver of the 2D L-BFGS GH200 study, run once in its own allocation.

Derived from the H100 precision-study driver. Admission (job/node, manifest, plan hash, pinned
source commit with exactly the two new files, runtime versions and the torch.optim.lbfgs module
hash, formal initialization and data hashes, idle GPUs, one exclusive lock file per GPU), a
prebuild on lane 0 (native extension, then one unscored light solve so the Triton kernels are
compiled), then four lanes pull fresh-process solves (light_solve.py) from the pre-randomized plan
with a deadline guard and a fail-fast guard, then lock release, the reader (read_lbfgs_study.py)
and a verified archive. Each task's lane, timing, exit code and log are recorded. No retries.
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

NEW_FILES = ('experiments/contracted_jets/light_solve.py', 'experiments/contracted_jets/lbfgs_light.py')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='work dir with source/, lbfgs/, native-cache/')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--self-test', action='store_true', help='CPU control flow: four fake lanes, tasks run a stub')
    args = parser.parse_args()
    root = args.root.resolve()
    study, source = root / 'lbfgs', root / 'source'
    control, results = study / 'control', study / 'results'
    control.mkdir(exist_ok=True)
    assert not results.exists(), 'results exist: this batch runs once'
    results.mkdir()
    plan_path = study / args.plan
    plan = json.loads(plan_path.read_text())
    protocol = json.loads((study / plan['protocol']).read_text())
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
    manifest = json.loads((study / 'manifest.json').read_text())
    bad = [n for n, v in manifest['files'].items() if sha256(root / n) != v['sha256']]
    state['checks']['manifest'] = {'verified': len(manifest['files']), 'mismatches': bad}
    state['checks']['plan_protocol'] = plan['protocol_sha256'] == sha256(study / plan['protocol'])
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--porcelain', '-uall'], cwd=source, capture_output=True, text=True)
    state['checks']['source_commit'] = head.stdout.strip()
    state['checks']['source_status'] = status.stdout
    pinned = head.returncode == 0 and head.stdout.strip() == protocol['source']['commit']
    only_new = status.returncode == 0 and sorted(status.stdout.split()) == sorted(['??'] * len(NEW_FILES) + list(NEW_FILES))
    state['checks']['source_only_new_files'] = only_new
    if not state['checks']['job_and_node'] or bad or not state['checks']['plan_protocol'] or not pinned or not only_new:
        return finish('ADMISSION_REFUSED', 2)
    env_base = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                    PYTHONPATH=os.pathsep.join([str(source / 'src'), os.environ.get('PYTHONPATH', '')]),
                    FLASHNS_CUDA_CACHE=str(root / 'native-cache'), TRITON_CACHE_DIR=str(root / 'cache' / 'triton'),
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
    bench = source / 'experiments' / 'contracted_jets'
    if not args.self_test:
        versions = subprocess.run([args.python, '-c', (
            'import json,hashlib,inspect,torch,triton,torch.optim.lbfgs as l; '
            'print(json.dumps({"torch":torch.__version__,"triton":triton.__version__,'
            '"lbfgs":hashlib.sha256(inspect.getsource(l).encode()).hexdigest()}))')], cwd=bench, env=env_base, capture_output=True, text=True)
        observed = json.loads(versions.stdout.strip().splitlines()[-1]) if versions.returncode == 0 else {}
        state['checks']['runtime'] = {'observed': observed, 'stderr_tail': versions.stderr[-300:],
                                      'match': observed.get('torch') == protocol['runtime']['torch'] and observed.get('triton') == protocol['runtime']['triton']
                                      and observed.get('lbfgs') == protocol['runtime']['torch_optim_lbfgs_module_sha256']}
        if not state['checks']['runtime']['match']:
            return finish('ADMISSION_REFUSED_RUNTIME', 2)
        # Formal initializations and data must match the frozen hashes (CPU, before any GPU work).
        check = subprocess.run([args.python, '-c', (
            'import json,hashlib,sys; sys.path.insert(0,"experiments/pinn_solver"); import problem; '
            'print(json.dumps({"theta":{s:hashlib.sha256(problem.parameters(s).numpy().tobytes()).hexdigest() for s in %r},'
            '"meta":{n:{k:v for k,v in problem.Problem(interior_count=n,device="cpu").meta.items() if k.endswith("sha256")} for n in %r}}))'
            % (protocol['seeds'], protocol['interior_counts']))], cwd=source, env=env_base, capture_output=True, text=True)
        observed = json.loads(check.stdout.strip().splitlines()[-1]) if check.returncode == 0 else {}
        expected = protocol['frozen_inputs']
        state['checks']['frozen_inputs'] = {
            'exit': check.returncode, 'stderr_tail': check.stderr[-500:],
            'theta_match': {str(s): observed.get('theta', {}).get(str(s)) == expected['theta_sha256'][str(s)] for s in protocol['seeds']},
            'data_match': {f'{n}:{k}': observed.get('meta', {}).get(str(n), {}).get(k) == v
                           for n, hashes in expected['data_sha256'].items() for k, v in hashes.items()}}
        if check.returncode or not all(state['checks']['frozen_inputs']['theta_match'].values()) \
                or not all(state['checks']['frozen_inputs']['data_match'].values()):
            return finish('ADMISSION_REFUSED_FROZEN_INPUTS', 2)
        smi = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], capture_output=True, text=True)
        rows = [r.split(',') for r in smi.stdout.strip().splitlines()]
        uuids = [r[0].strip() for r in rows]
        state['checks']['gpus'] = {'rows': smi.stdout, 'apps': apps.stdout,
                                   'idle': smi.returncode == apps.returncode == 0 and len(rows) == 4 and all('GH200' in r[1] for r in rows)
                                   and all(int(r[2]) <= 100 for r in rows) and not any(u in apps.stdout for u in uuids)}
        if not state['checks']['gpus']['idle']:
            return finish('ADMISSION_REFUSED_GPU_NOT_IDLE', 2)
    else:
        uuids = [f'FAKE-{i}' for i in range(4)]
    locks_dir = root / 'locks'
    locks_dir.mkdir(exist_ok=True)
    lanes, held = [], []
    for uuid in uuids:
        path = locks_dir / (uuid + '.lock')
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (FileExistsError, BlockingIOError):
            for f in held:
                os.close(f)
            state['lanes'] = lanes
            return finish('ADMISSION_REFUSED_LOCK', 2)
        lanes.append({'uuid': uuid, 'lock_path': str(path), 'lock_inode': os.fstat(fd).st_ino, 'acquired_unix': time.time()})
        held.append(fd)
    state['lanes'] = lanes
    save()

    def lane_env(index):
        cache = root / 'cache' / f'lane{index}'
        cache.mkdir(parents=True, exist_ok=True)
        return dict(env_base, CUDA_VISIBLE_DEVICES='' if args.self_test else lanes[index]['uuid'],
                    TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), XDG_CACHE_HOME=str(cache / 'xdg'), TMPDIR=str(cache),
                    CUDA_CACHE_PATH=str(cache / 'cuda'))

    if not args.self_test:
        # Compile the native extension and the light optimizer's Triton kernels once, outside the timed queue.
        state['checks']['prebuild'] = {}
        for label, argv in (('build', [args.python, str(bench / 'light_solve.py'), '--build-only', '--output', str(control / 'prebuild-build.json')]),
                            ('light_solve', [args.python, str(bench / 'light_solve.py'), '--optimizer', 'light', '--backend', protocol['backend'],
                                             '--interior-count', '2048', '--seed', str(protocol['seeds'][0]), '--output', str(control / 'prebuild-light.json')])):
            with open(control / f'prebuild-{label}.log', 'w') as log:
                try:
                    code = subprocess.run(argv, cwd=bench, stdout=log, stderr=subprocess.STDOUT, timeout=1800, env=lane_env(0)).returncode
                except subprocess.TimeoutExpired:
                    code = 'TIMEOUT'
            state['checks']['prebuild'][label] = code
            save()
            if code != 0:
                for f in held:
                    os.close(f)
                return finish('PREBUILD_FAILED', 3)

    work = queue.Queue()
    for t in plan['tasks']:
        work.put(t)
    durations = {}
    FAIL_FAST = 8
    outcomes = []

    def estimate(t):
        return 1.25 * max(durations.get((t['cell'], t['interior']), [protocol['task_timeout_seconds'] / 3]))

    def worker(index):
        env = lane_env(index)
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
            argv = [args.python, str(bench / t['argv'][0]), *t['argv'][1:], '--output', str(out)]
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
            durations.setdefault((t['cell'], t['interior']), []).append(record['seconds'])
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
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], capture_output=True, text=True)
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
            [args.python, str(study / 'read_lbfgs_study.py'), '--root', str(root), '--plan', args.plan,
             '--output', str(study / 'lbfgs-result.json')], cwd=study, env=env_base, stdout=log, stderr=subprocess.STDOUT).returncode
    save()
    durable = study / 'durable'
    durable.mkdir(exist_ok=True)
    archive = durable / '2d-lbfgs-gh200-complete.tar.gz'
    state['state'] = 'ARCHIVING'
    save()
    members = sorted(str(p.relative_to(root)) for p in study.rglob('*')
                     if p.is_file() and durable not in p.parents and not p.name.endswith('.tmp'))
    members += [str((source / n).relative_to(root)) for n in NEW_FILES]
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
