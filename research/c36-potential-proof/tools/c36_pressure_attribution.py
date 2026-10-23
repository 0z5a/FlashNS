"""Post-hoc attribution of the C36 pressure error to velocity error or unpenalized residual.

    PYTHONPATH=src python tools/c36_pressure_attribution.py --observations OBS_DIR --calibration CAL.json --output OUT.json [--tags final]
    PYTHONPATH=src python tools/c36_pressure_attribution.py --runs RUNS --eta 0.0625 --calibration CAL.json --output OUT.json

The exact pressure is zero, so every learned pressure gradient is error. With R the momentum
residual of the learned field (the observer's own formula, box376.validation.observations) and
dI = inertia(U_net) - f the inertia error (zero for the exact velocity), the identities

    grad p_net = R - dI                     (gradient level)
    lap p_net  = div R - div dI             (Poisson level; div dI = Q below)

hold pointwise. dI and div dI contain only velocity errors. For each region in the motion window
the tool reports the weighted RMS of each term (the observer's component-mean convention), the
residual over the loss's own normalization, the weighted cosine between the pressure term and
the velocity term (-dI or -div dI) with its noise floor 1/sqrt(entries), the pressure gradient's
fraction of the residual energy |grad p|^2/|R|^2, and the region's share of the stratified
momentum energy. If the velocity error forced the pressure error, the gradient cosine would be
near one. In the cutoff layers div R and div dI are large and nearly cancel, so Poisson-level
cosines there are flagged as not interpretable. The tool also reproduces the observer's recorded
regional momentum and pressure RMS from the saved state (point-order check) and checks both
identities to rounding. No observation is rerun.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from box376.validation import fields_from_state, observations
from box376.validation_data import make_holdouts

REGIONS = ('plateau', 'face', 'edge', 'corner')


def inner(a, b, w):
    return float(np.sum(w * np.sum((a * b).reshape(len(a), -1), axis=1)))


def wrms(a, w):
    """Weighted RMS with the observer's convention (mean over components, box376.validation.weighted_rms)."""
    a = np.asarray(a).reshape(len(a), -1)
    return float(np.sqrt(np.sum(w * np.mean(a * a, axis=1)) / np.sum(w)))


def attribute(state, gauge, holdout, scales, nu):
    fields = fields_from_state(state)
    reference = holdout['reference']
    residual, spatial, _, _ = observations(fields, reference, nu)
    grad_p, lap_p = fields['grad_p'], np.trace(fields['H_p'], axis1=1, axis2=2)
    u, g, h = fields['U'], fields['G'], fields['H']
    # Inertia error and its divergence from the velocity state alone (no pressure channels).
    inertia_error = fields['Ut'] + np.einsum('nj,nij->ni', u, g) - nu * fields['lap'] - reference['f']
    div_inertia_error = (np.trace(fields['GUt'], axis1=1, axis2=2) + np.einsum('nij,nji->n', g, g)
                         + np.einsum('nj,niji->n', u, h) - nu * np.trace(fields['gradlap'], axis1=1, axis2=2)
                         - np.trace(reference['df'], axis1=1, axis2=2))
    div_residual = np.trace(spatial, axis1=1, axis2=2)
    exact_inertia = reference['Ut'] + np.einsum('nj,nij->ni', reference['U'], reference['G']) - nu * reference['lap']
    checks = {'gradient_identity_over_max': float(np.abs(grad_p - (residual - inertia_error)).max() / np.abs(grad_p).max()),
              'poisson_identity_over_max': float(np.abs(lap_p - (div_residual - div_inertia_error)).max() / np.abs(lap_p).max()),
              'exact_inertia_minus_force_over_force': float(np.abs(exact_inertia - reference['f']).max() / np.abs(reference['f']).max())}
    weights, regions, times = holdout['weights'], holdout['regions'], holdout['time_regions']
    total_momentum_energy = inner(residual, residual, weights)
    out = {}
    for index, name in enumerate(REGIONS):
        sel = (regions == index) & (times == 0)
        w = weights[sel]
        row = {'points': int(sel.sum()),
               'share_of_stratified_momentum_energy': inner(residual[sel], residual[sel], w) / total_momentum_energy}
        for level, p_term, r_term, v_term, r_scale in (
                ('gradient', grad_p[sel], residual[sel], -inertia_error[sel], scales['momentum_residual']),
                ('poisson', lap_p[sel], div_residual[sel], -div_inertia_error[sel], scales['momentum_residual_gradient'])):
            entries = p_term.size
            row[level] = {'pressure_term_rms': wrms(p_term, w), 'residual_term_rms': wrms(r_term, w), 'velocity_term_rms': wrms(v_term, w),
                          'residual_term_over_loss_scale': wrms(r_term, w) / r_scale,
                          'velocity_over_pressure_term': wrms(v_term, w) / wrms(p_term, w),
                          'cosine_pressure_velocity': inner(p_term, v_term, w) / np.sqrt(inner(p_term, p_term, w) * inner(v_term, v_term, w)),
                          'cosine_noise_floor': 1 / np.sqrt(entries),
                          'pressure_fraction_of_residual_energy': inner(p_term, p_term, w) / inner(r_term, r_term, w)}
        row['poisson']['interpretable'] = name in ('plateau', 'face')
        delta_u = fields['U'][sel] - reference['U'][sel]
        delta_g = fields['G'][sel] - reference['G'][sel]
        row['velocity_relative_l2'] = wrms(delta_u, w) / wrms(reference['U'][sel], w)
        row['velocity_gradient_relative_l2'] = wrms(delta_g, w) / wrms(reference['G'][sel], w)
        row['corrected_pressure_over_limit'] = wrms(fields['p'][sel] - gauge[sel], w) / scales['pressure'] / 1e-3
        out[name] = row
    reproduction = {}
    for index in range(5):
        sel = regions == index
        if np.any(sel):
            reproduction[str(index)] = {'momentum': wrms(residual[sel], weights[sel]) / scales['momentum_residual'],
                                        'pressure': wrms(fields['p'][sel] - gauge[sel], weights[sel]) / scales['pressure']}
    return out, checks, reproduction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--eta', type=float, default=1 / 32)
    parser.add_argument('--nu', type=float, default=.01)
    parser.add_argument('--holdout-seed', type=int, default=794701)
    parser.add_argument('--tags', default='final')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.observations is None) == (args.runs is None):
        parser.error('give exactly one of --observations and --runs')
    calibration = json.loads(args.calibration.read_text())
    assert calibration['source_eta'] == args.eta
    scales = calibration['candidate_fixed_physical_scales']
    datasets, _, _ = make_holdouts(seed=args.holdout_seed, eta=args.eta)
    holdout = datasets['stratified']
    tags = args.tags.split(',')
    if args.runs is not None:
        found = [(f'{run.name}__{tag}', run / f'physical-{tag}') for run in sorted(args.runs.iterdir()) for tag in tags
                 if (run / f'physical-{tag}' / 'stratified-fields.npz').is_file()]
    else:
        found = [(p.name, p) for p in sorted(args.observations.iterdir()) if any(p.name.endswith('__' + t) for t in tags)]
    result = {'scope': __doc__.strip().splitlines()[0], 'eta': args.eta, 'nu': args.nu, 'holdout_seed': args.holdout_seed,
              'scales': scales, 'observations': {}}
    for label, path in found:
        with np.load(path / 'stratified-fields.npz', allow_pickle=False) as raw:
            state, gauge = raw['state'], raw['gauge']
        regions, checks, reproduction = attribute(state, gauge, holdout, scales, args.nu)
        recorded = json.loads((path / 'report.json').read_text())['datasets']['stratified']['regions']
        diff = max(abs(reproduction[r][k] - recorded[r][m]) / max(abs(recorded[r][m]), 1e-300)
                   for r in reproduction for k, m in (('momentum', 'normalized_momentum_rms'), ('pressure', 'normalized_corrected_pressure_rms'))
                   if recorded[r][m] != 0)
        assert diff < 1e-9, (label, diff)
        result['observations'][label] = {'regions': regions, 'identity_checks': checks, 'reproduction_max_relative_difference': diff}
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    for label, entry in result['observations'].items():
        cells = []
        for name in REGIONS:
            g, q = entry['regions'][name]['gradient'], entry['regions'][name]['poisson']
            cells.append('%s cos %.2f (floor %.2f) p/R %.3f dI/p %.1f%s' % (
                name[:4], g['cosine_pressure_velocity'], g['cosine_noise_floor'], g['pressure_fraction_of_residual_energy'],
                g['velocity_over_pressure_term'], ' | Poisson cos %.2f divdI/lap %.2f' % (q['cosine_pressure_velocity'], q['velocity_over_pressure_term']) if q['interpretable'] else ''))
        print(label, 'plateau momentum share %.4f' % entry['regions']['plateau']['share_of_stratified_momentum_energy'])
        for c in cells:
            print('   ', c)


if __name__ == '__main__':
    main()
