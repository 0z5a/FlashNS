"""Stdlib-only outer finalizer for one paired-engine seed, including failed admission.

box376.finalize_c36_tta (unchanged) bound to the paired trainer's command line and arm directories.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time
from box376.run_c36_tta import seal_archive, sha


def main():
    root, packet = map(Path, sys.argv[1:3]); job=sys.argv[3]; code=int(sys.argv[4])
    state_path=root/'control/driver-state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    # Normal driver already joins its child session. For an abruptly lost driver,
    # signal only the recorded session whose current command still binds this run.
    pid=state.get('child_pid')
    if pid and state.get('job')==job:
        try:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if b'box376.potential_train_pair' in cmd and any(a.startswith(str(root).encode()+b'/') and a.endswith(b'/train') for a in cmd) and os.getpgid(pid)==pid:
                os.killpg(pid,signal.SIGTERM)
                end=time.monotonic()+20
                while Path(f'/proc/{pid}').exists() and time.monotonic()<end:time.sleep(.25)
                if Path(f'/proc/{pid}').exists():os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError:pass
        except FileNotFoundError:pass
    closure={'job':job,'batch_exit':code,'finished_unix':time.time(),'driver_state':state.get('state','NOT_STARTED')}
    (root/'control/batch-exit.json').write_text(json.dumps(closure,indent=2)+'\n')
    archives=root.parent/'archives';archives.mkdir(exist_ok=True)
    stem=f'{root.name}-job{job}'; archive=archives/(stem+'.tar.gz'); receipt=archives/(stem+'.json')
    if archive.exists() and receipt.exists():
        r=json.loads(receipt.read_text());assert sha(archive)==r['sha256']
        assert state.get('state') in ('COMPLETE','FAILED_ARCHIVED')
        print(json.dumps({'state':'EXISTING_ARCHIVE_VERIFIED',**closure}));return
    if archive.exists():
        # Retain the damaged/partial original archive without overwriting it.
        archive=archives/(stem+'-fallback.tar.gz');receipt=archives/(stem+'-fallback.json')
    r=seal_archive(packet,root,archive)
    r.update(closure,seed=int(root.name.removeprefix('seed')),workload_exit=code,outer_failure_archive=True)
    receipt.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'state':'FAILURE_ARCHIVED','sha256':r['sha256'],**closure}))


if __name__=='__main__':main()
