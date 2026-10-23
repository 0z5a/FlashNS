"""CPU check of two structural properties of the frozen single-box source (source.py).

    PYTHONPATH=src python -m box376.check_source_structure --output source-structure-check-20261018.json

(a) In the box plateau (chi == 1) during the motion window the velocity is affine, U = c' + L y,
    its Laplacian vanishes and the force f = c'' + (L' + L^2) y is a gradient (L is diagonal), so
    the spatial force Jacobian is symmetric. In the cutoff layer the Jacobian is not symmetric.
(b) At the rest slices t = 0, 1/4, 3/4, 1 the exact velocity, force and their derivatives vanish
    identically, as they do for every t outside (1/4, 3/4).
(c) The fraction of uniform space-time points at which the exact velocity is nonzero.
Documentation only: no training or observation is rerun.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import source as S

REST_SLICES = (0.0, 0.25, 0.75, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261018)
    report = {'source': 'src/box376/source.py', 'source_sha256': hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
              'numpy': np.__version__, 'rng_seed': 20261018, 'eta': 1 / 32, 'nu': 0.01}

    t = rng.uniform(0.27, 0.73, 4000)
    m = S.motion(t)
    y = rng.uniform(-1, 1, (len(t), 3)) * m['h']
    r = S.source(np.concatenate([m['c'] + y, t[:, None]], 1))
    inside = r['chi'] == 1.0
    f, df = r['f'][inside], r['df'][inside]
    predicted = m['acceleration'][inside] + (m['ell_t'][inside] + m['ell'][inside] ** 2) * y[inside]
    report['a_plateau_motion_window'] = {
        'points': int(inside.sum()),
        'max_abs_f': float(np.abs(f).max()),
        'max_abs_f_minus_gradient_form': float(np.abs(f - predicted).max()),
        'max_abs_df': float(np.abs(df).max()),
        'max_abs_antisymmetric_df': float(np.abs(df - np.transpose(df, (0, 2, 1))).max()),
        'max_abs_U_minus_affine': float(np.abs(r['U'][inside] - (m['v'][inside] + m['ell'][inside] * y[inside])).max()),
        'max_abs_laplacian_U': float(np.abs(r['lap'][inside]).max())}

    t = rng.uniform(0.27, 0.73, 4000)
    m = S.motion(t)
    layer = rng.choice([-1, 1], (len(t), 3)) * (m['h'] + (1 + rng.uniform(0, 1, (len(t), 3))) / 32)
    y = np.where(rng.uniform(size=(len(t), 3)) < 0.5, layer, rng.uniform(-1, 1, (len(t), 3)) * m['h'])
    r = S.source(np.concatenate([m['c'] + y, t[:, None]], 1))
    shell = (r['chi'] > 0) & (r['chi'] < 1)
    df = r['df'][shell]
    report['a_cutoff_layer_motion_window'] = {
        'points': int(shell.sum()), 'max_abs_df': float(np.abs(df).max()),
        'max_abs_antisymmetric_df': float(np.abs(df - np.transpose(df, (0, 2, 1))).max())}

    report['b_rest_slices'] = {}
    for slice_t in REST_SLICES:
        x = rng.uniform(0, 1, (20000, 3))
        r = S.source(np.concatenate([x, np.full((len(x), 1), slice_t)], 1))
        entry = {k: float(np.abs(r[k]).max()) for k in ('U', 'G', 'H', 'Ut', 'lap', 'f', 'df')}
        entry.update(points=len(x), max_chi=float(r['chi'].max()))
        report['b_rest_slices'][str(slice_t)] = entry

    x = rng.uniform(0, 1, (200000, 4))
    r = S.source(x)
    nonzero = np.abs(r['U']).max(1) > 0
    window = (x[:, 3] > 0.25) & (x[:, 3] < 0.75)
    report['c_uniform_space_time'] = {
        'points': len(x), 'fraction_nonzero_U': float(nonzero.mean()),
        'fraction_nonzero_U_inside_motion_window': float(nonzero[window].mean()),
        'nonzero_U_time_range': [float(x[nonzero, 3].min()), float(x[nonzero, 3].max())]}

    a, b = report['a_plateau_motion_window'], report['b_rest_slices']
    report['passed'] = {
        'plateau_force_is_gradient': a['max_abs_antisymmetric_df'] <= 1e-12 * a['max_abs_df'] and a['max_abs_f_minus_gradient_form'] <= 1e-12 * a['max_abs_f'],
        'plateau_velocity_affine': a['max_abs_U_minus_affine'] <= 1e-12 and a['max_abs_laplacian_U'] <= 1e-12,
        'rest_slices_vanish': all(v == 0.0 for e in b.values() for k, v in e.items() if k not in ('points', 'max_chi'))}
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report['passed']))
    if not all(report['passed'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
