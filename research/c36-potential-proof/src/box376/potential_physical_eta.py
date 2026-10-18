"""The unchanged C36 physical observer on an instance with another cutoff width (C36 v6).

    python -m box376.potential_physical_eta --inputs DIR --protocol PROTOCOL --checkpoint CKPT --output OUT [--observer-chunk N]

box376.potential_physical builds its holdouts with make_holdouts and evaluates the trained
field with native_field, both at their default cutoff width 1/32; those are its only
cutoff-dependent calls. This wrapper binds both names in the observer's module to the
protocol's instance width and then runs the observer unchanged. Gates, scales (from the
instance's calibration.json), holdout seed and trajectory checks are the observer's own.
"""
import functools
import json
import sys
from pathlib import Path

import box376.potential_physical as observer
from box376.potential_geometry import native_field
from box376.validation_data import make_holdouts


def main():
    protocol = json.loads(Path(sys.argv[sys.argv.index('--protocol') + 1]).read_text())
    eta = protocol['instance']['eta']
    inputs = Path(sys.argv[sys.argv.index('--inputs') + 1])
    calibration = json.loads((inputs / 'calibration.json').read_text())
    if calibration['source_eta'] != eta:
        raise ValueError('Calibration belongs to another cutoff width')
    observer.make_holdouts = functools.partial(make_holdouts, eta=eta)
    observer.native_field = functools.partial(native_field, eta=eta)
    observer.main()


if __name__ == '__main__':
    main()
