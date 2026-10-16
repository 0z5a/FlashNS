"""Cluster-side driver for the C36 zero-gate folding GPU batch in its own allocation.

Runs once: admission (job, manifests, runtime hashes, GPU lock inode, idle GPU, node-local space),
the compiler smoke and CPU folding regressions in the actual runtime, the folded timing run and its
NumPy audit, then the unfolded control run and its audit if enough time remains, the reader
(box376.read_zero_gate_gpu), lock release, cache removal and a verified archive. Derived from the
2026-10-13 node driver; no retries.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--lane', required=True, help='GPU_UUID:LOCK_PATH:LOCK_INODE')
    parser.add_argument('--self-test', action='store_true',
                        help='CPU control-flow test: skips job/GPU checks; children are stubs')
    args = parser.parse_args()
    signal.pthread_sigmask(signal.SIG_SETMASK, [])
    root = args.root.resolve()
    protocol = json.loads((root / args.protocol).read_text())
    limits = protocol['timeouts_seconds']
    control = root / 'control'
    control.mkdir(exist_ok=False)
    children, locks = [], []
    cache = (Path(os.environ['NODE_LOCAL_SCRATCH']) / (os.environ['USER'] + '-' + root.name)
             if not args.self_test else root / 'selftest-cache')
    cache_created = False
    state = {'state': 'STARTING', 'started': time.time(), 'job': args.job, 'node': args.node,
             'self_test': args.self_test, 'deadline_unix': args.deadline_unix, 'stages': [],
             'runs': {}, 'resource_locks_released': False}

    def save(name, value):
        temporary = control / (name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(control / name)

    def remaining():
        return args.deadline_unix - time.time()

    def run(name, command, env, timeout):
        with (control / (name + '.stdout')).open('x') as out, (control / (name + '.stderr')).open('x') as err:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     env=env, start_new_session=True)
            children.append(child)
            save(name + '-started.json', {'pid': child.pid, 'command': command, 'timeout': timeout, 'started': time.time()})
            timed_out = False
            try:
                rc = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                rc = child.returncode
        row = {'name': name, 'pid': child.pid, 'exit': rc, 'timed_out': timed_out, 'finished': time.time()}
        state['stages'].append(row)
        save('progress.json', state)
        return rc == 0 and not timed_out

    def terminate(signum, frame):
        raise RuntimeError('Finite batch signalled for closure: ' + str(signum))

    signal.signal(signal.SIGUSR1, terminate)
    signal.signal(signal.SIGTERM, terminate)
    try:
        uuid, lock_path, lock_inode = args.lane.split(':')
        if not args.self_test:
            assert os.environ['SLURM_JOB_ID'] == args.job
            assert os.uname().nodename.split('.')[0] == args.node
        manifest = json.loads((root / 'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            assert digest(root / name) == expected, name
        runtime = json.loads((root / 'runtime-manifest.json').read_text())
        if not args.self_test:
            for path, expected in runtime['files'].items():
                assert digest(Path(path)) == expected, path
        state['checks'] = {'manifest_files': len(manifest['files']),
                           'runtime_files': 0 if args.self_test else len(runtime['files'])}
        lock = open(lock_path, 'r+b')
        assert os.fstat(lock.fileno()).st_ino == int(lock_inode)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
        if not args.self_test:
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
            gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            row = next(line.split(',') for line in gpus.splitlines() if line.split(',')[1].strip() == uuid)
            assert uuid not in apps and int(row[2]) >= 60000 and int(row[3]) == 0
            state['gpu_before'] = {'apps': apps, 'gpus': gpus}
        assert shutil.disk_usage(cache.parent).free > 4 * 1024**3
        assert shutil.disk_usage(root).free > 1024**3
        cache.mkdir(exist_ok=False)
        cache_created = True
        state['lane'] = {'uuid': uuid, 'lock_path': lock_path, 'lock_inode': int(lock_inode), 'acquired': time.time()}
        save('progress.json', state)
        env = os.environ.copy()
        env.pop('PYTHONHOME', None)
        env.pop('LD_PRELOAD', None)
        env.update(CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   PYTHONDONTWRITEBYTECODE='1', TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                   TRITON_CACHE_DIR=str(cache / 'triton'), TORCHINDUCTOR_COMPILE_THREADS='1',
                   XDG_CACHE_HOME=str(cache / 'xdg'), TMPDIR=str(cache), OMP_NUM_THREADS='4')
        activation = 'source ' + runtime['activation_script'] + ' >/dev/null 2>&1\n'
        activation += 'export PYTHONPATH=' + str(root / 'src') + ':' + runtime['compiler_runtime'] + ':${PYTHONPATH:-}\nexec "$@"'
        base = ['bash', '-c', activation, 'flashns']
        timing = root / 'protocol-graph-timing-20261012.json'
        reference = root / 'frozen-full-native-reference.npz'

        def stub(path, payload):
            return [sys.executable, '-c', 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); '
                    'p.write_text(json.dumps(json.loads(sys.argv[2])))', str(path), json.dumps(payload)]

        def science(name, flags):
            out = root / 'science' / name
            if args.self_test:
                command = stub(out / 'report.json', {'state': 'COMPLETE', 'self_test': True, 'zero_gate_folding': bool(flags),
                                                      'paired_geometric_mean_ratio': 1.3, 'checks': []})
            else:
                command = base + ['python', '-m', 'box376.packed_analytic_gpu_run', '--inputs', str(root / 'data'),
                                  '--protocol', str(timing), '--reference', str(reference), '--activation-mode', 'compiled',
                                  '--output', str(out), '--legacy-triton-target'] + flags
            ok = run('science-' + name, command, env, max(60, min(limits['science_max'], remaining())))
            audit = root / 'science' / (name + '-audit.json')
            if args.self_test:
                payload = {'state': 'PASS', 'self_test': True, 'ratio': 1.3, 'interval': [1.29, 1.31],
                           'candidate_mean_seconds': 0.1 if flags else 0.16, 'baseline_mean_seconds': 0.13 if flags else 0.21,
                           'maximum_error_over_original_budget': 0.0, 'paired_windows_checked': 20,
                           'gradient_and_objective_comparisons': 78}
                if flags:
                    payload['zero_gate_folding'] = {'state': 'PASS'}
                command = stub(audit, payload)
            else:
                command = base + ['python', '-m', 'box376.audit_packed_analytic_gpu', '--inputs', str(root / 'data'),
                                  '--proof', str(out), '--reference', str(reference), '--protocol', str(timing),
                                  '--output', str(audit)]
            audited = ok and run('audit-' + name, command, dict(env, CUDA_VISIBLE_DEVICES=''), limits['audit'])
            state['runs'][name] = {'science_ok': ok, 'audit_ok': audited}

        cpu_env = dict(env, CUDA_VISIBLE_DEVICES='')
        smoke = run('compiler-smoke', stub(root / 'compiler-smoke' / 'stub.json', {'self_test': True}) if args.self_test else
                    base + ['python', '-m', 'box376.packed_analytic_gpu_smoke', '--output', str(root / 'compiler-smoke'),
                            '--legacy-triton-target'], env, limits['compiler_smoke'])
        regressions = run('cpu-fold-regressions', stub(control / 'regression-stub.json', {'self_test': True}) if args.self_test else
                          base + ['python', '-m', 'unittest', 'box376.test_fold_zero_gate', '-v'], cpu_env, limits['cpu_regressions'])
        state['prerequisites_passed'] = smoke and regressions
        if state['prerequisites_passed']:
            science('folded', ['--fold-zero-gates'])
            if remaining() >= limits['control_min_remaining_seconds']:
                science('unfolded', [])
            else:
                state['runs']['unfolded'] = {'skipped': True, 'remaining_seconds': remaining()}
        reader = [sys.executable, '-I', str(root / 'src' / 'box376' / 'read_zero_gate_gpu.py'),
                  '--protocol', str(root / args.protocol), '--science', str(root / 'science'),
                  '--state', str(control / 'progress.json'), '--output', str(root / 'zero-gate-gpu-result.json')]
        save('progress.json', state)
        state['reader_ok'] = run('reader', reader, os.environ.copy(), 300)
        state['state'] = 'COMPLETE' if state['prerequisites_passed'] and state['reader_ok'] else 'FAILED'
    except BaseException:
        state.update(state='FAILED', failure=traceback.format_exc())
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        state['owned_children_joined'] = True
        try:
            state['gpu_apps_after'] = 'self-test' if args.self_test else subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
        except BaseException:
            state['gpu_apps_after'] = 'unavailable'
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        state['resource_locks_released'] = True
        if cache_created and cache.exists():
            shutil.rmtree(cache)  # Only this batch's newly created cache.
        state['finished'] = time.time()
        save('result-before-archive.json', state)
        try:
            durable = root / 'durable'
            durable.mkdir()
            files = {str(p.relative_to(root)): p for p in sorted(root.rglob('*'))
                     if p.is_file() and not p.is_symlink() and durable not in p.parents
                     and p.name != 'frozen-full-native-reference.npz' and not p.name.endswith('.tmp')}
            archive_manifest = {name: digest(path) for name, path in files.items()}
            (control / 'archive-manifest.json').write_text(json.dumps(archive_manifest, indent=1) + '\n')
            archive = durable / 'zero-gate-gpu-complete.tar.gz'
            with tarfile.open(archive, 'w:gz') as tf:
                for name, path in files.items():
                    tf.add(path, arcname=name, recursive=False)
                tf.add(control / 'archive-manifest.json', arcname='control/archive-manifest.json', recursive=False)
            with tarfile.open(archive) as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                assert len(members) == len(archive_manifest) + 1
                for member in members:
                    if member.name == 'control/archive-manifest.json':
                        continue
                    with tf.extractfile(member) as handle:
                        h = hashlib.sha256()
                        for block in iter(lambda: handle.read(1 << 20), b''):
                            h.update(block)
                    assert h.hexdigest() == archive_manifest[member.name], member.name
            state['archive'] = {'state': 'PASS', 'path': str(archive), 'sha256': digest(archive),
                                'bytes': archive.stat().st_size, 'members_verified': len(members),
                                'excluded': 'frozen-full-native-reference.npz (unchanged input, sha256 in manifest.json)'}
            (durable / (archive.name + '.sha256')).write_text(f"{state['archive']['sha256']}  {archive.name}\n")
        except BaseException:
            state.update(state='FAILED', archive_failure=traceback.format_exc())
        save('result.json', state)
    sys.exit(0 if state['state'] == 'COMPLETE' else 1)


if __name__ == '__main__':
    main()
