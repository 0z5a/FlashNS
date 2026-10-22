"""CPU checks of the traced and compiled external routes on a tiny network and data set (fixtures only)."""
import json
import os
import unittest
from pathlib import Path

import torch

from box376.data import flatten, initialize, make_data, torch_data, views
from box376.external_c36 import PreparedPotentialExternalObjective, build_chunked_traced_external, build_compiled_external
from box376.normalized_objective import normalization_key

ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY = (4, 3, 4)


def tiny():
    torch.set_default_dtype(torch.float64)
    calibration = json.loads((ROOT/'data/calibration.json').read_text())
    data = torch_data(make_data(3))
    data['normalization_key'] = normalization_key(calibration)
    theta = flatten(initialize(TOPOLOGY, 361901))
    return data, calibration, theta


def close(a, b):
    return all(torch.allclose(x, y, rtol=1e-9, atol=1e-11) for x, y in zip(a, b))


class ExternalCompile(unittest.TestCase):
    def test_traced_matches_eager_and_reads_parameters_in_place(self):
        data, calibration, initial = tiny()
        # Nested AD stays eager-only: tracing its fourth-order graph is not part of the comparison.
        for backend in ('func_jacfwd', 'jet_collapsed'):
            if backend == 'jet_collapsed' and not os.environ.get('FLASHNS_TORCHJET'):
                continue
            eager = PreparedPotentialExternalObjective(data, calibration, backend)
            theta = initial.clone().requires_grad_(True)
            traced = build_compiled_external(data, calibration, backend, theta, TOPOLOGY, 2, compile=False)
            self.assertTrue(close(traced(), eager(views(initial, TOPOLOGY), 2)), backend)
            with torch.no_grad():
                theta.add_(1e-3)
            self.assertTrue(close(traced(), eager(views(initial + 1e-3, TOPOLOGY), 2)), backend)


    def test_chunked_traced_matches_eager_and_reads_parameters_in_place(self):
        data, calibration, initial = tiny()
        eager = PreparedPotentialExternalObjective(data, calibration, 'func_jacfwd')
        theta = initial.clone().requires_grad_(True)
        evaluation = build_chunked_traced_external(data, calibration, 'func_jacfwd', theta, TOPOLOGY, 2)
        self.assertTrue(close(evaluation(), eager(views(initial, TOPOLOGY), 2)))
        with torch.no_grad():
            theta.add_(1e-3)
        self.assertTrue(close(evaluation(), eager(views(initial + 1e-3, TOPOLOGY), 2)))


if __name__ == '__main__':
    unittest.main()
