"""Post-hoc structure of the stratified corrected-pressure error in saved C36 observations.

    PYTHONPATH=src python tools/c36_pressure_structure.py --observations OBS_DIR --output OUT.json [--tags final,LBFGS-call800]
    PYTHONPATH=src python tools/c36_pressure_structure.py --runs RUNS --eta 0.0625 --pressure-scale S --output OUT.json

Reads each observation's saved native state (`stratified-fields.npz`) and rebuilds the frozen
stratified holdout (seed 794701). The exact pressure is zero, so the corrected-pressure error is
dp = p(x, t) - p(gauge, t). For the motion window (time region 0) it reports, per spatial region,
the weighted RMS of dp normalized by the frozen pressure scale and how much of it a per-time offset
explains: dp is binned in t (10 equal bins over (1/4, 3/4)) and the bin mean is removed. In the
plateau it also fits a per-bin offset plus a linear term in y = x - c(t), and compares the RMS of
the learned pressure gradient times the plateau half-width with the dp RMS. It reproduces each
observation's recorded regional and global corrected-pressure RMS from the saved state (point-order
check), and reports the layers-only floor: the global stratified value with the plateau error set
to zero. --runs reads a v2/v6 batch layout (RUNS/<run>/physical-<tag>/) instead of an observation
batch, and --eta selects the instance (default 1/32). With --eta-comparison it also reports the frozen source's force and force-gradient RMS per
region on the stratified holdout geometry at eta = 1/32 and 1/16. No observation is rerun.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from box376.source import motion, source
from box376.validation import fields_from_state
from box376.validation_data import make_holdouts

REGIONS = ('plateau', 'face', 'edge', 'corner', 'outside')
BINS = np.linspace(0.25, 0.75, 11)


def wrms(values, weights):
    return float(np.sqrt(np.sum(weights * values ** 2) / np.sum(weights)))


def explained(dp, weights, design):
    """Weighted least-squares fit of dp on per-bin design columns; returns residual and R^2."""
    sw = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], dp * sw, rcond=None)
    rest = dp - design @ coef
    total = float(np.sum(weights * dp ** 2))
    return rest, (1 - float(np.sum(weights * rest ** 2)) / total) if total > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--eta', type=float, default=1 / 32)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tags', default='final')
    parser.add_argument('--holdout-seed', type=int, default=794701)
    parser.add_argument('--pressure-scale', type=float, default=0.141864656295504)
    parser.add_argument('--eta-comparison', action='store_true')
    args = parser.parse_args()
    if (args.observations is None) == (args.runs is None):
        parser.error('give exactly one of --observations and --runs')
    datasets, _, _ = make_holdouts(seed=args.holdout_seed, eta=args.eta)
    holdout = datasets['stratified']
    points, regions, times = holdout['points'], holdout['regions'], holdout['time_regions']
    weights = np.asarray(holdout['weights'])
    m = motion(points[:, 3])
    y = points[:, :3] - m['c']
    tbin = np.clip(np.digitize(points[:, 3], BINS) - 1, 0, len(BINS) - 2)
    tags = args.tags.split(',')
    result = {'scope': __doc__.strip().splitlines()[0], 'holdout_seed': args.holdout_seed, 'eta': args.eta,
              'pressure_scale': args.pressure_scale, 'time_bins': BINS.tolist(), 'observations': {}}
    if args.runs is not None:
        found = [(f'{run.name}__{tag}', run / f'physical-{tag}') for run in sorted(args.runs.iterdir()) for tag in tags
                 if (run / f'physical-{tag}' / 'stratified-fields.npz').is_file()]
    else:
        found = [(path.name, path) for path in sorted(args.observations.iterdir())
                 if any(path.name.endswith('__' + tag) for tag in tags)]
    for label, path in found:
        with np.load(path / 'stratified-fields.npz', allow_pickle=False) as raw:
            state, gauge = raw['state'], raw['gauge']
        fields = fields_from_state(state)
        dp = fields['p'] - gauge
        entry = {'regions': {}}
        for index, name in enumerate(REGIONS[:4]):
            sel = (regions == index) & (times == 0)
            design = np.eye(len(BINS) - 1)[tbin[sel]]
            rest, r2 = explained(dp[sel], weights[sel], design)
            row = {'points': int(sel.sum()), 'normalized_rms': wrms(dp[sel], weights[sel]) / args.pressure_scale,
                   'signed_mean_normalized': float(np.average(dp[sel], weights=weights[sel])) / args.pressure_scale,
                   'per_time_offset_R2': r2, 'normalized_rms_after_offset': wrms(rest, weights[sel]) / args.pressure_scale}
            if index == 0:
                affine = np.concatenate([design] + [design * y[sel, k:k + 1] for k in range(3)], 1)
                rest2, r2a = explained(dp[sel], weights[sel], affine)
                half = m['h'][sel] + args.eta
                grad = fields['grad_p'][sel]
                row.update(offset_plus_linear_R2=r2a, normalized_rms_after_offset_and_linear=wrms(rest2, weights[sel]) / args.pressure_scale,
                           gradient_times_half_width_normalized_rms=wrms(np.linalg.norm(grad * half, axis=1), weights[sel]) / args.pressure_scale)
            entry['regions'][name] = row
        recorded = json.loads((path / 'report.json').read_text())['datasets']['stratified']
        reproduced = {str(r): wrms(dp[regions == r], weights[regions == r]) / args.pressure_scale for r in range(5) if np.any(regions == r)}
        reproduced['global'] = wrms(dp, weights) / args.pressure_scale
        expected = {r: recorded['regions'][r]['normalized_corrected_pressure_rms'] for r in reproduced if r != 'global'}
        expected['global'] = recorded['global']['normalized_corrected_pressure_rms']
        entry['reproduction'] = {'max_relative_difference': max(abs(reproduced[k] - expected[k]) / max(abs(expected[k]), 1e-300)
                                                                for k in expected if expected[k] != 0),
                                 'zero_regions_match': all(reproduced[k] == 0.0 for k in expected if expected[k] == 0)}
        assert entry['reproduction']['max_relative_difference'] < 1e-9 and entry['reproduction']['zero_regions_match'], label
        layers = (regions >= 1) & (regions <= 3)
        entry['global_normalized_rms'] = reproduced['global']
        entry['layers_only_floor_normalized_rms'] = float(np.sqrt(np.sum(weights[layers] * dp[layers] ** 2) / np.sum(weights))) / args.pressure_scale
        flat = (times != 0) & (regions < 4)
        entry['flat_time_normalized_rms'] = wrms(dp[flat], weights[flat]) / args.pressure_scale
        entry['per_bin_plateau_mean_normalized'] = [
            float(np.average(dp[(regions == 0) & (times == 0) & (tbin == b)], weights=weights[(regions == 0) & (times == 0) & (tbin == b)])) / args.pressure_scale
            if np.any((regions == 0) & (times == 0) & (tbin == b)) else None for b in range(len(BINS) - 1)]
        result['observations'][label] = entry
    if args.eta_comparison:
        result['eta_comparison'] = {}
        for eta in (1 / 32, 1 / 16):
            sets, _, _ = make_holdouts(seed=args.holdout_seed, eta=eta)
            pts, reg, tr = sets['stratified']['points'], sets['stratified']['regions'], sets['stratified']['time_regions']
            src = source(pts, eta=eta)
            rows = {}
            for index, name in enumerate(REGIONS[:4]):
                sel = (reg == index) & (tr == 0)
                rows[name] = {'force_component_rms': float(np.sqrt(np.mean(src['f'][sel] ** 2))),
                              'force_gradient_component_rms': float(np.sqrt(np.mean(src['df'][sel] ** 2))),
                              'velocity_component_rms': float(np.sqrt(np.mean(src['U'][sel] ** 2)))}
            result['eta_comparison'][str(eta)] = rows
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    for name, entry in result['observations'].items():
        p = entry['regions']['plateau']
        print(name, 'global', round(entry['global_normalized_rms'], 5), 'layers-only floor', round(entry['layers_only_floor_normalized_rms'], 5),
              'plateau', round(p['normalized_rms'], 5), 'offset R2', round(p['per_time_offset_R2'], 3), 'repro', entry['reproduction'])
    if args.eta_comparison:
        print(json.dumps(result['eta_comparison']))


if __name__ == '__main__':
    main()
