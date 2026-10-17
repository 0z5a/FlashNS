"""Distance between the exact width-w and original C36 velocity fields on the physical holdouts.

CPU, NumPy only. Uses the observer's metric (weighted relative L2 of velocity) with the
original field as reference, so it can be compared with observed checkpoint errors.

    python -m box376.schedule_field_distance --protocol PROTOCOL --output OUT.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from box376 import schedule_family as family
from box376.source import source
from box376.validation_data import make_holdouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--widths', default='0.75,1.0')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    datasets, _, _ = make_holdouts(seed=protocol['physical']['holdout_seed'])
    result = {'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              'metric': 'weighted RMS of (U_w - U_1/2) over weighted RMS of U_1/2 on each holdout',
              'widths': {}}
    for width in (float(w) for w in args.widths.split(',')):
        rows = {}
        for name, dataset in datasets.items():
            points, weights = dataset['points'], dataset['weights']
            reference = source(points)['U']
            changed = family.source_w(points, width)['U']
            rms = lambda a: float(np.sqrt(np.sum(weights * np.sum(a * a, 1)) / np.sum(weights)))
            denominator = rms(reference)
            rows[name] = rms(changed - reference) / denominator if denominator > 0 else None
        result['widths'][f'{width:g}'] = rows
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['widths'], indent=1))


if __name__ == '__main__':
    main()
