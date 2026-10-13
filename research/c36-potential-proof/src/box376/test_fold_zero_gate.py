"""Regression checks for forcing, targets and unchanged full normalization."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from box376.data import FAMILIES, flatten, initialize, torch_data, views
from box376.fold_zero_gate import ZeroGateFoldedObjective
from box376.frozen_data import load_cache
from box376.normalized_objective import normalization_key
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
from torch_c36_packed import activate_packed
from torch_c36_packed_vjp import activate_packed_vjp


class ZeroGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[2] / "data"
        self.calibration = json.loads((root / "calibration.json").read_text())
        data = torch_data(load_cache(root / "training-source.npz"))
        data["normalization_key"] = normalization_key(self.calibration)
        self.data = copy.deepcopy(data)
        for index, name in enumerate(FAMILIES):
            self.data["families"][name] = {
                "points": data["families"]["interior"]["points"][[1965, 2489, 1079]].clone(),
                "weights": torch.tensor([.3, 1.2, 2.1], dtype=torch.float64),
                "target": torch.full((3, 4), .02 * (index + 1), dtype=torch.float64),
                "W": 3.6 * (index + 1)}
            if name == "interior":
                self.data["families"][name].update(
                    force=torch.full((3, 3), .17, dtype=torch.float64),
                    force_gradient=torch.full((3, 3, 3), .23, dtype=torch.float64))

    def engines(self):
        return [PreparedPotentialNormalizedObjective(self.data, self.calibration,
                    activation=activate_packed, activation_vjp=activate_packed_vjp),
                PreparedPotentialADObjective(self.data, self.calibration, activation=activate_packed)]

    def test_nonzero_forcing_targets_and_changed_parameters(self):
        topology = (4, 5, 4)
        for engine in self.engines():
            original_sizes = [len(engine._data["families"][n]["points"]) for n in FAMILIES]
            folded = ZeroGateFoldedObjective(engine)
            self.assertTrue(torch.all(folded._constant_numerators > 0))
            torch.testing.assert_close(folded._consumer.denominators,
                torch.tensor([3.6, 7.2, 10.8, 14.4], dtype=torch.float64))
            for seed in (361901, 361902):
                theta = flatten(initialize(topology, seed))
                for offset in (0., .007):
                    parameters = views(theta + offset, topology)
                    for chunk in (1, 2, 4):
                        for actual, expected in zip(folded(parameters, chunk), engine(parameters, chunk)):
                            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)
            self.assertEqual(original_sizes, [len(engine._data["families"][n]["points"]) for n in FAMILIES])

    def test_zero_value_with_nonzero_derivative_is_retained(self):
        for engine in self.engines():
            for name in FAMILIES:
                inputs, gate = engine._geometry[name]
                gate = torch.zeros_like(gate)
                gate[0, 0, 1] = 1e-30
                engine._geometry[name] = inputs, gate
            folded = ZeroGateFoldedObjective(engine)
            for name in FAMILIES:
                self.assertEqual(folded.counts[name]["active"], 1)
                self.assertTrue(folded._masks[name][0])
            parameters = initialize((4, 5, 4), 361901)
            for a, b in zip(folded(parameters, 2), engine(parameters, 2)):
                torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)

    def test_all_zero_geometry_keeps_complete_constant_objective(self):
        for engine in self.engines():
            for name in FAMILIES:
                inputs, gate = engine._geometry[name]
                engine._geometry[name] = inputs, torch.zeros_like(gate)
            folded = ZeroGateFoldedObjective(engine)
            parameters = initialize((4, 5, 4), 361901)
            actual, expected = folded(parameters, 2), engine(parameters, 2)
            self.assertGreater(actual[0].item(), 0)
            self.assertEqual(torch.count_nonzero(actual[1]).item(), 0)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)

    def test_native_engine_is_rejected(self):
        native = PreparedPotentialADObjective(self.data, self.calibration, coordinate_ad=True)
        with self.assertRaises(ValueError):
            ZeroGateFoldedObjective(native)


if __name__ == "__main__":
    unittest.main()
