"""Clock, physical-observer boundary and complete-denominator TTA classification."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


class ActiveClock:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.started = clock()
        self.excluded = 0.0
        self.paused_at = None

    def elapsed(self):
        return (self.paused_at if self.paused_at is not None else self.clock())-self.started-self.excluded

    @contextmanager
    def pause(self):
        if self.paused_at is not None:
            raise RuntimeError('nested observation pause')
        self.paused_at = self.clock()
        try:
            yield
        finally:
            self.excluded += self.clock()-self.paused_at
            self.paused_at = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_observer(checkpoint, inputs, protocol, output, chunk, timeout, *, self_test=False):
    output = Path(output)
    if self_test:
        # Explicit control-flow fixture: never a physical report or production qualification.
        return {'physical_acceptance_pass': output.name == 'physical-final-observation',
                'self_test_fixture_only': True, 'exit': 0, 'finished_unix': time.time()}
    argv = [sys.executable, '-m', 'box376.potential_physical_eta', '--inputs', str(inputs),
            '--protocol', str(protocol), '--checkpoint', str(checkpoint), '--output', str(output),
            '--observer-chunk', str(chunk)]
    if output.exists():
        raise RuntimeError('refusing duplicate physical observation: '+str(output))
    began = time.time()
    with (output.parent/(output.name+'.log')).open('x') as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'physical observer exited {result.returncode}; preserve partial report')
    report = json.loads((output/'report.json').read_text())
    assert report['state'] == 'COMPLETE'
    assert report['protocol_sha256'] == sha(protocol)
    assert report['checkpoint_sha256'] == sha(checkpoint)
    assert report['raw_sha256'] == sha(output/'raw-physical.npz')
    return {'physical_acceptance_pass': report['physical_acceptance_pass'], 'exit': result.returncode,
            'started_unix': began, 'finished_unix': time.time(), 'report_sha256': sha(output/'report.json'),
            'raw_sha256': report['raw_sha256'], 'checkpoint_sha256': report['checkpoint_sha256']}


def cohort_result(expected_seeds, audited_runs):
    expected = set(expected_seeds)
    if not expected or len(expected) != len(expected_seeds):
        raise ValueError('duplicate registered seeds')
    runs = {}
    for run in audited_runs:
        seed = run['seed']
        if seed not in expected or seed in runs:
            raise ValueError('unexpected or duplicate seed')
        runs[seed] = run
    entries = []
    for seed in expected_seeds:
        run = runs.get(seed)
        if not run or not run.get('independent_audit_pass'):
            entries.append({'seed': seed, 'state': 'INCOMPLETE_OR_INVALID', 'time_s': None})
        elif run.get('self_test'):
            raise ValueError('fixture cannot enter formal cohort')
        elif run.get('physical_acceptance_pass'):
            t = run['time_to_verified_accuracy_s']
            if not isinstance(t, (float, int)) or not 0 < t < float('inf'):
                raise ValueError('positive finite observed TTA required')
            entries.append({'seed': seed, 'state': 'PASS', 'time_s': t})
        else:
            entries.append({'seed': seed, 'state': 'NOT_REACHED_WITHIN_BUDGET', 'time_s': None,
                            'right_censor_elapsed_s': run.get('right_censor_elapsed_s')})
    n = len(entries); k = sum(e['state'] == 'PASS' for e in entries)
    z = 1.959963984540054; p = k/n; d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    radius = z*((p*(1-p)/n+z*z/(4*n*n))**.5)/d
    times = sorted(e['time_s'] for e in entries if e['state'] == 'PASS')
    return {'state': 'FORMAL_COHORT_PASS' if k == n else 'COHORT_NOT_FULLY_QUALIFIED',
            'denominator': n, 'passes': k, 'entries': entries,
            'success_fraction_wilson95': [max(0.,center-radius),min(1.,center+radius)],
            'cohort_median_tta_s': ((times[(n-1)//2]+times[n//2])/2 if k == n else None),
            'cohort_range_tta_s': [times[0],times[-1]] if k == n else None,
            'speedup_claim': False}
