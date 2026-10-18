"""Check that the cutoff-width generators reproduce the frozen eta = 1/32 inputs (C36 v6).

    PYTHONPATH=src python -m box376.check_eta_generators --frozen data --work DIR --output OUT.json

Runs box376.calibrate_source_eta and box376.prepare_training_data_eta at eta = 1/32 into DIR
and compares every array with the frozen calibration.npz and training-source.npz, and the
calibration scales with the frozen report. Arrays computed by the source may differ in the
last bits between NumPy builds; sampled indices, regions, weights, ids and targets must match
exactly. No GPU.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

EXACT = ('weights', 'regions', 'time_active', 'ids', 'target')


def compare(frozen, regenerated):
    rows = {}
    with np.load(frozen) as a, np.load(regenerated) as b:
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            x, y = a[key], b[key]
            assert x.shape == y.shape and x.dtype == y.dtype, key
            scale = float(np.abs(x).max()) if x.size else 0.0
            difference = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.size else 0.0
            rows[key] = {'identical': bool(np.array_equal(x, y)), 'differing_entries': int(np.count_nonzero(x != y)),
                         'max_abs_difference_over_max_abs': difference / scale if scale else difference}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = lambda *argv: subprocess.run([sys.executable, '-m', *argv], check=True, capture_output=True)
    run('box376.calibrate_source_eta', '--eta', '0.03125', '--output', str(args.work))
    run('box376.prepare_training_data_eta', '--eta', '0.03125', '--calibration', str(args.work / 'calibration.json'),
        '--output', str(args.work / 'training-source.npz'))
    report = {'numpy': np.__version__, 'arrays': {}}
    for name in ('calibration', 'training-source'):
        report['arrays'][name] = compare(args.frozen / f'{name}.npz', args.work / f'{name}.npz')
    frozen = json.loads((args.frozen / 'calibration.json').read_text())['candidate_fixed_physical_scales']
    regenerated = json.loads((args.work / 'calibration.json').read_text())['candidate_fixed_physical_scales']
    report['scales_relative_difference'] = {k: abs(frozen[k] - regenerated[k]) / abs(frozen[k]) for k in frozen}
    rows = [(name, key, row) for name, arrays in report['arrays'].items() for key, row in arrays.items()]
    report['max_relative_difference'] = max(row['max_abs_difference_over_max_abs'] for _, _, row in rows)
    report['exact_arrays_identical'] = all(row['identical'] for _, key, row in rows if key.split('_', 1)[1] in EXACT)
    report['passed'] = (report['exact_arrays_identical'] and report['max_relative_difference'] < 1e-12
                        and max(report['scales_relative_difference'].values()) < 1e-12)
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'max_relative_difference', 'exact_arrays_identical', 'scales_relative_difference')}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
