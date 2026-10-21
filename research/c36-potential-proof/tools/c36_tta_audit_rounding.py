"""Post-hoc diagnosis of the frozen TTA auditor's spatial-residual-gradient failures (CPU only).

    PYTHONPATH=src python3 tools/c36_tta_audit_rounding.py --root ROOT --output OUT.json

The frozen auditor (box376.audit_c36_tta, line 206) recomputes each saved raw array of the
physical consumer and requires np.allclose(rtol=1e-10, atol=1e-13) against the value saved on
the GPU host. The spatial residual gradient is a sum of terms whose magnitudes reach about 1e5
and cancel to about 1e-2, so a tolerance relative to the result is far below the rounding of
the terms. For every complete observation of every sealed archive this tool reports, per
dataset: the frozen tolerance's failures, the error in units of the spacing of the largest
summand, and the error over 2*gamma_n*S, where S is the elementwise sum of the summands'
absolute values, n = 10 roundings and gamma_n = n*u/(1-n*u). Two evaluations that each satisfy
the standard bound |fl(x)-x| <= gamma_n*S differ by at most 2*gamma_n*S. This does not change the
frozen audit, its outputs or the registered cohort result.
"""
import argparse
import datetime
import hashlib
import io
import json
import platform
import tarfile
from pathlib import Path

import numpy as np

from box376.audit_c36_tta import metrics

U = 2.0 ** -53
N_ROUNDINGS = 10  # 1 + 3 + 3 + 1 + 1 + 1 summands; each product adds one rounding
GAMMA = N_ROUNDINGS * U / (1 - N_ROUNDINGS * U)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summands(state, df):
    h = np.empty((len(state), 4, 3, 3))
    for k, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        h[:, :, i, j] = h[:, :, j, i] = state[:, :, 4 + k]
    u, g = state[:, :3, 0], state[:, :3, 1:4]
    return {'time_gradient': np.abs(state[:, :3, 14:17]),
            'velocity_gradient_squared': np.einsum('nij,njk->nik', np.abs(g), np.abs(g)),
            'velocity_times_hessian': np.einsum('nj,nijk->nik', np.abs(u), np.abs(h[:, :3])),
            'viscous': .01 * np.abs(state[:, :3, 10:13]),
            'pressure_hessian': np.abs(h[:, 3]),
            'force_gradient': np.abs(df)}


def dataset_row(state, saved, recomputed, df):
    err = np.abs(recomputed - saved)
    frozen = 1e-13 + 1e-10 * np.abs(saved)
    terms = summands(state, df)
    S = sum(terms.values())
    largest = np.maximum.reduce(list(terms.values()))
    bound = np.where(S > 0, err / np.where(S > 0, S, 1) / (2 * GAMMA), np.where(err > 0, np.inf, 0.0))
    worst = np.unravel_index(int(np.argmax(err / frozen)), saved.shape)
    return {'elements': int(saved.size), 'frozen_tolerance_failures': int(np.count_nonzero(err > frozen)),
            'max_abs_error': float(err.max()), 'max_abs_value': float(np.abs(saved).max()),
            'max_error_over_spacing_of_largest_summand': float(np.max(err / np.maximum(np.spacing(largest), 5e-324))),
            'max_error_over_2gamma_S': float(np.max(bound)), 'elements_over_2gamma_S': int(np.count_nonzero(bound > 1)),
            'worst_frozen_element': {'value': float(saved[worst]), 'error': float(err[worst]),
                                     'summands': {k: float(v[worst]) for k, v in terms.items()}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.root / 'packet' / 'protocol-formal-tta-20261021.json').read_text())
    record = {'state': 'POSTHOC_DIAGNOSTIC_NOT_AUDIT_PASS', 'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'machine': platform.machine(), 'numpy': np.__version__, 'gamma_n': GAMMA, 'roundings': N_ROUNDINGS,
              'frozen_tolerance': {'rtol': 1e-10, 'atol': 1e-13}, 'seeds': {}}
    for seed in protocol['seeds']:
        [archive] = sorted((args.root / 'archives').glob(f'seed{seed}-job*.tar.gz'))
        receipt = json.loads(archive.with_name(archive.name[:-len('.tar.gz')] + '.json').read_text())
        assert sha(archive) == receipt['sha256']
        rows = {}
        with tarfile.open(archive, 'r:gz') as t:
            def read(name):
                data = t.extractfile(name).read()
                assert hashlib.sha256(data).hexdigest() == receipt['member_sha256'][name], name
                return data
            scales = json.loads(read('packet/data/calibration.json'))['candidate_fixed_physical_scales']
            for name in sorted(receipt['member_sha256']):
                if not (name.startswith('run/physical-') and name.endswith('/report.json')):
                    continue
                report = json.loads(read(name))
                if report['state'] != 'COMPLETE':
                    rows[name.split('/')[1]] = {'state': report['state']}
                    continue
                with np.load(io.BytesIO(read(name.replace('report.json', 'raw-physical.npz'))), allow_pickle=False) as z:
                    datasets = {}
                    for ds in report['datasets']:
                        state = z[ds + '_native_state']
                        reference = {k: z[ds + '_source_' + k] for k in ('U', 'f', 'df')}
                        _, raw, _ = metrics(state, reference, z[ds + '_weights'], z[ds + '_gauge'], scales)
                        datasets[ds] = dataset_row(state, z[ds + '_spatial_residual_gradient'], raw['spatial_residual_gradient'], reference['df'])
                rows[name.split('/')[1]] = {'state': 'COMPLETE', 'datasets': datasets}
        record['seeds'][str(seed)] = {'archive_sha256': receipt['sha256'], 'observations': rows}
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + '\n')
    for seed, entry in record['seeds'].items():
        for tag, obs in entry['observations'].items():
            if obs['state'] != 'COMPLETE':
                print(seed, tag, obs['state']); continue
            print(seed, tag, {ds: (r['frozen_tolerance_failures'], r['max_error_over_spacing_of_largest_summand'],
                                   round(r['max_error_over_2gamma_S'], 4), r['elements_over_2gamma_S']) for ds, r in obs['datasets'].items()})


if __name__ == '__main__':
    main()
