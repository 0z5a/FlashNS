"""One seed, one allocated GH200, one attempt; archive all outcomes and release immediately."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tarfile
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def command(argv):
    p = subprocess.run(argv,capture_output=True,text=True,timeout=30)
    return {'argv':argv,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}


def seal_archive(packet, run, archive):
    if archive.exists():
        raise RuntimeError('archive exists; no overwrite')
    files = {}
    for prefix, root in (('packet', packet), ('run', run)):
        for path in sorted(root.rglob('*')):
            if path.is_file() and not any(x in ('cache','__pycache__') for x in path.relative_to(root).parts) and not path.name.endswith('.tmp'):
                files[prefix+'/'+str(path.relative_to(root))] = path
    hashes = {name:sha(path) for name,path in files.items()}
    with tarfile.open(archive,'w:gz') as t:
        for name,path in files.items():t.add(path,arcname=name,recursive=False)
    with tarfile.open(archive,'r:gz') as t:
        assert set(t.getnames()) == set(hashes)
        for m in t:
            assert hashlib.sha256(t.extractfile(m).read()).hexdigest() == hashes[m.name],m.name
    result = {'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,
              'members':len(files),'member_sha256':hashes,'all_members_reread':True}
    assert sha(archive) == result['sha256']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    args = p.parse_args()
    packet,root = args.packet.resolve(),args.root.resolve()
    protocol_path = packet/'protocol-formal-tta-20261021.json'
    protocol = json.loads(protocol_path.read_text())
    assert args.seed in protocol['seeds']
    root.mkdir(parents=True,exist_ok=True)
    once = os.open(root/'once.lock',os.O_CREAT|os.O_RDWR,0o600)
    fcntl.flock(once,fcntl.LOCK_EX|fcntl.LOCK_NB)
    control = root/'control';control.mkdir(exist_ok=True)
    state_path = control/'driver-state.json'
    assert not state_path.exists(), 'seed already attempted; no retry'
    state = {'state':'ADMITTING','seed':args.seed,'job':os.environ['SLURM_JOB_ID'],
             'host':socket.gethostname(),'driver_pid':os.getpid(),'started_unix':time.time(),
             'protocol_sha256':sha(protocol_path),'checks':{}}
    child = None; gpu_lock = None; code = 1

    def save():
        tmp = state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)

    def terminate(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    save()
    try:
        manifest = json.loads((packet/'manifest.json').read_text())
        assert all(sha(packet/name)==row['sha256'] for name,row in manifest['files'].items())
        state['checks']['source_manifest_sha256'] = sha(packet/'manifest.json')
        state['checks']['frozen_files_verified'] = len(manifest['files'])
        import torch
        import triton
        import numpy as np
        import scipy
        assert torch.__version__ == '2.14.1+cu130' and triton.__version__ == '3.8.0'
        assert np.__version__ == '2.5.4' and scipy.__version__ == '1.18.1'
        smi = command(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
        apps = command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])
        rows = smi['stdout'].strip().splitlines()
        assert smi['exit']==apps['exit']==0 and len(rows)==1
        assert 'GH200' in rows[0]
        uuid = rows[0].split(',')[0].strip()
        assert int(rows[0].split(',')[3]) <= 100 and uuid not in apps['stdout']
        locks = root/'locks';locks.mkdir()
        lock_path = locks/(uuid+'.lock')
        gpu_lock = os.open(lock_path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state['gpu_lock'] = {'uuid':uuid,'path':str(lock_path),'inode':os.fstat(gpu_lock).st_ino,'acquired_unix':time.time()}
        state['checks']['nvml_before'] = {'gpus':smi,'apps':apps}
        assert args.deadline_unix-time.time() > 13100, 'insufficient full frozen worst-case window'
        cache = root/'cache';cache.mkdir(exist_ok=True)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',
                   TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),
                   CUDA_CACHE_PATH=str(cache/'cuda'),TMPDIR=str(cache),XDG_CACHE_HOME=str(cache/'xdg'),
                   TORCHINDUCTOR_COMPILE_THREADS='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        argv = [sys.executable,'-m','box376.potential_train_tta','--inputs',str(packet/'data'),
                '--protocol',str(protocol_path),'--seed',str(args.seed),'--arm','fresh','--output',str(root/'train')]
        state.update(state='RUNNING',argv=argv,training_process_started_unix=time.time());save()
        with (root/'train.log').open('x') as log:
            child = subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=child.pid;save()
            code = child.wait(timeout=max(1,args.deadline_unix-time.time()))
        state['child_exit']=code
        state['child_finished_unix']=time.time()
        if code == 0:
            result = json.loads((root/'train/report.json').read_text())
            assert result['state']=='COMPLETE' and not result['self_test']
            state['tta']=result['tta']
            state['scientific_outcome_pending_audit'] = result['tta']['state']
    except BaseException as error:
        state['failure']=traceback.format_exc()
        code = int(error.code) if isinstance(error,SystemExit) and isinstance(error.code,int) else 1
    finally:
        # Join only the session created for this seed, including any active observer child.
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        state['all_owned_children_joined_unix']=time.time()
        try:
            state['checks']['nvml_after']=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])
        except BaseException:state['checks']['nvml_after_failure']=traceback.format_exc()
        if gpu_lock is not None:
            os.close(gpu_lock);state['gpu_lock']['released_unix']=time.time()
        state.update(state='WORKLOAD_FINISHED_ARCHIVING',exit=code,finished_workload_unix=time.time());save()
        archives = root.parent/'archives';archives.mkdir(exist_ok=True)
        archive = archives/f"seed{args.seed}-job{state['job']}.tar.gz"
        try:
            receipt = seal_archive(packet,root,archive)
            receipt.update(seed=args.seed,job=state['job'],workload_exit=code,protocol_sha256=state['protocol_sha256'],finished_unix=time.time())
            archive.with_suffix('').with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
            state['archive']={k:v for k,v in receipt.items() if k!='member_sha256'}
            state['state']='COMPLETE' if code==0 else 'FAILED_ARCHIVED'
        except BaseException:
            state.update(state='ARCHIVE_FAILED',archive_failure=traceback.format_exc());code=90
        state['driver_finished_unix']=time.time();save()
        os.close(once)
    return code


if __name__ == '__main__':
    sys.exit(main())
