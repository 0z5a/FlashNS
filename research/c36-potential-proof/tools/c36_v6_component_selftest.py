"""CPU component self-test of the C36 v6 additions (login-node sized).

    PYTHONPATH=src python3 tools/c36_v6_component_selftest.py --inputs DATA --protocol PROTOCOL --part trainer|observer --output OUT.json [--interior N]

The complete trainer self-test exceeds the login node's process time limit, so the parts that
v6 adds are exercised separately; the training loop, checkpoints and observation calls are the
unchanged v2/v5 code already run on H100.
- trainer: box376.potential_train_v6.subset_problem and native_objective on the first N of the
  registered 131 subset points (default 3) and every value point, compared under the protocol
  budget with both eager engines on the same subset, at the stored-equivalent initial and at
  changed parameters; and the canonical denominators are kept.
- observer: box376.potential_physical_eta binds the protocol width into the unchanged observer
  (holdout metadata and the native field evaluator), and refuses a calibration of another width.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def trainer_part(args, protocol):
    import box376.potential_train_v6 as trainer
    from torch_c36_packed import activate_packed
    from torch_c36_packed_vjp import activate_packed_vjp
    from box376.data import FAMILIES, flatten, initialize, views
    from box376.potential_ad_objective import PreparedPotentialADObjective
    from box376.potential_objective import PreparedPotentialNormalizedObjective

    trainer.SUBSET_INTERIOR = trainer.SUBSET_INTERIOR[:args.interior]
    data, calibration = trainer._cpu_problem(args, protocol)
    assert data['eta'] == protocol['instance']['eta']
    subset = trainer.subset_problem(data)
    report = {'interior_points': args.interior, 'value_points': int(sum(len(subset['families'][n]['points']) for n in FAMILIES[1:])),
              'canonical_denominators_kept': all(subset['families'][n]['W'] == data['families'][n]['W'] for n in FAMILIES),
              'comparisons': []}
    topology = tuple(protocol['topology'])
    theta = flatten(initialize(topology, 361901))
    engines = {'analytic': PreparedPotentialNormalizedObjective(subset, calibration, activation=activate_packed,
                                                                activation_vjp=activate_packed_vjp),
               'parameter_ad': PreparedPotentialADObjective(subset, calibration, activation=activate_packed)}
    for label, shift in (('original', 0.0), ('changed', 1e-3)):
        current = theta + shift
        started = time.perf_counter()
        native = trainer.native_objective(current, subset, calibration, topology)
        seconds = time.perf_counter() - started
        for engine, objective in engines.items():
            actual = objective(views(current, topology), 128)
            for kind, a, b in zip(trainer.KINDS, actual, native):
                a, b = a.detach().numpy(), b.detach().numpy()
                error = float(np.max(np.abs(a - b) / (protocol['atol'] + protocol['rtol'] * np.abs(b))))
                report['comparisons'].append({'parameters': label, 'engine': engine, 'kind': kind, 'over_budget': error})
        report[f'native_seconds_{label}'] = seconds
    report['passed'] = report['canonical_denominators_kept'] and all(c['over_budget'] <= 1 for c in report['comparisons'])
    return report


def observer_part(args, protocol):
    import functools
    import box376.potential_physical as observer
    import box376.potential_physical_eta as wrapper
    eta = protocol['instance']['eta']
    calls = {}
    original_main = observer.main
    observer.main = lambda: calls.update(make=observer.make_holdouts, field=observer.native_field)
    sys.argv = ['potential_physical_eta', '--inputs', str(args.inputs), '--protocol', str(args.protocol),
                '--checkpoint', 'unused', '--output', 'unused']
    wrapper.main()
    observer.main = original_main
    report = {'make_holdouts_bound_eta': calls['make'].keywords.get('eta'), 'native_field_bound_eta': calls['field'].keywords.get('eta')}
    datasets, _, metadata = calls['make'](seed=protocol['physical']['holdout_seed'])
    report['holdout_metadata_eta'] = metadata['eta']
    report['stratified_points'] = len(datasets['stratified']['points'])
    refused = False
    other = Path(args.output).with_suffix('.wrong-calibration')
    other.mkdir(exist_ok=True)
    wrong = json.loads((args.inputs / 'calibration.json').read_text())
    wrong['source_eta'] = 1 / 32
    (other / 'calibration.json').write_text(json.dumps(wrong))
    sys.argv[2] = str(other)
    try:
        wrapper.main()
    except ValueError:
        refused = True
    report['refuses_other_width_calibration'] = refused
    report['passed'] = (report['make_holdouts_bound_eta'] == eta == report['native_field_bound_eta'] == report['holdout_metadata_eta']
                        and refused and isinstance(calls['field'], functools.partial))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--part', choices=('trainer', 'observer'), required=True)
    parser.add_argument('--interior', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    protocol = json.loads(args.protocol.read_text())
    report = {'part': args.part, 'torch': torch.__version__, 'numpy': np.__version__}
    report.update(trainer_part(args, protocol) if args.part == 'trainer' else observer_part(args, protocol))
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'comparisons'}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
