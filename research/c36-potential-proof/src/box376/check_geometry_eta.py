"""CPU qualification of the C36 geometry at a selectable cutoff width (C36 v6).

    PYTHONPATH=src python -m box376.check_geometry_eta --inputs DIR --eta 0.0625 --output OUT.json

The model's geometry depends on the cutoff width only through its support gate. This check
(1) compares the prepared 36-channel feature and gate states with fresh native coordinate AD on
a region-stratified subset of the training interior (plateau, face, edge, corner and outside in
the motion window, plus resting times), under the original atol/rtol budget, and with a second
chunking; (2) verifies that the gate vanishes exactly at every initial, boundary and gauge
point of the training data, and at the gauge position for 2,001 times; (3) bounds the support
inside the unit cube and away from the gauge position over t in [0, 1]; (4) records the largest
feature magnitude inside the support (input conditioning). No GPU and no training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from box376.c36_coordinate_oracle import native_c36
from box376.data import torch_data
from box376.frozen_data import load_cache
from box376.potential_geometry import features_and_gate
from box376.potential_model import prepare_geometry
from box376.source import motion
from box376.validation_data import classify

ATOL, RTOL = 1e-11, 1e-9


def budget_ratio(actual, expected):
    return float(np.max(np.abs(actual - expected) / (ATOL + RTOL * np.abs(expected))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--eta', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    source_file = args.inputs / 'training-source.npz'
    data = torch_data(load_cache(source_file))
    assert data['eta'] == args.eta, (data['eta'], args.eta)
    eta = args.eta
    report = {'eta': eta, 'training_source_sha256': hashlib.sha256(source_file.read_bytes()).hexdigest(),
              'atol': ATOL, 'rtol': RTOL, 'torch': torch.__version__, 'numpy': np.__version__}

    points = data['families']['interior']['points']
    raw = points.numpy()
    regions = classify(raw, eta)
    active = (raw[:, 3] > .25) & (raw[:, 3] < .75)
    chosen = []
    for region in range(5):
        pool = np.flatnonzero((regions == region) & active)
        chosen += pool[np.linspace(0, len(pool) - 1, 16, dtype=np.int64)].tolist()
    flat = np.flatnonzero(~active)
    chosen += flat[np.linspace(0, len(flat) - 1, 8, dtype=np.int64)].tolist()
    indices = np.unique(np.asarray(chosen, dtype=np.int64))
    report['subset'] = {'points': len(indices), 'per_region_motion_window': 16, 'resting_times': 8,
                        'indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest()}
    subset = points[indices]
    inputs, gate = prepare_geometry(subset, eta, chunk_size=13)
    inputs64, gate64 = prepare_geometry(subset, eta, chunk_size=64)
    native = []
    for component in (0, 1):
        x = subset.detach().clone().requires_grad_(True)
        native.append(native_c36(lambda p: features_and_gate(p, eta)[component], x).detach().numpy())
    report['native_feature_states_over_budget'] = budget_ratio(inputs.numpy(), native[0])
    report['native_gate_states_over_budget'] = budget_ratio(gate.numpy(), native[1])
    report['chunk13_vs_chunk64_over_budget'] = max(budget_ratio(inputs.numpy(), inputs64.numpy()),
                                                   budget_ratio(gate.numpy(), gate64.numpy()))
    report['gate_nonzero_points_in_subset'] = int(np.count_nonzero(np.abs(gate.numpy()[:, 0, 0]) > 0))

    zero = {}
    for name in ('initial', 'boundary', 'gauge'):
        _, g = features_and_gate(data['families'][name]['points'], eta)
        zero[name] = {'points': len(g), 'max_abs_gate': float(g.abs().max())}
    times = torch.linspace(0, 1, 2001)
    gauge_points = torch.cat((torch.full((2001, 3), 1 / 16), times[:, None]), 1)
    _, g = features_and_gate(gauge_points, eta)
    zero['gauge_position_2001_times'] = {'points': 2001, 'max_abs_gate': float(g.abs().max())}
    report['gate_zero'] = zero

    t = np.linspace(0, 1, 20001)
    m = motion(t)
    low, high = m['c'] - (m['h'] + 2 * eta), m['c'] + (m['h'] + 2 * eta)
    report['support'] = {'minimum_coordinate': float(low.min()), 'maximum_coordinate': float(high.max()),
                         'gauge_coordinate': 1 / 16}

    features, g = features_and_gate(points, eta)
    inside = g[:, 0] != 0
    report['support_feature_max_abs'] = [float(v) for v in features[inside, :3].abs().max(0).values]
    report['interior_points_with_nonzero_gate'] = int(inside.sum())

    report['passed'] = {
        'native_states': max(report['native_feature_states_over_budget'], report['native_gate_states_over_budget']) <= 1,
        'chunking': report['chunk13_vs_chunk64_over_budget'] <= 1,
        'gate_exactly_zero': all(v['max_abs_gate'] == 0.0 for v in zero.values()),
        'support_inside_cube_and_away_from_gauge': report['support']['minimum_coordinate'] > 1 / 16
                                                   and report['support']['maximum_coordinate'] < 1}
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'native_feature_states_over_budget', 'native_gate_states_over_budget',
                                             'support', 'support_feature_max_abs')}))
    if not all(report['passed'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
