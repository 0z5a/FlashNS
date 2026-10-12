"""Compare the new training baseline with independent coordinate AD."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from box376.data import FAMILIES, flatten, initialize, torch_data, views
from box376.frozen_data import load_cache
from box376.normalized_objective import normalization_key
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.verify_potential_gradient import compare


class PotentialAutogradTests(unittest.TestCase):
    def test_three_implementations_and_changed_parameters(self):
        torch.set_num_threads(1)
        inputs = Path(__file__).resolve().parents[2] / "data"
        calibration = json.loads((inputs / "calibration.json").read_text())
        data = torch_data(load_cache(inputs / "training-source.npz"))
        data["normalization_key"] = normalization_key(calibration)
        indices = np.array([1965, 2489, 1079])
        probe = copy.deepcopy(data)
        for index, name in enumerate(FAMILIES):
            probe["families"][name] = {
                "points": data["families"]["interior"]["points"][indices].clone(),
                "weights": torch.tensor([.3, 1.2, 2.1], dtype=torch.float64),
                "target": torch.full((3, 4), .02 * (index + 1), dtype=torch.float64),
                "W": 3.6 * (index + 1)}
            if name == "interior":
                for key in ("force", "force_gradient"):
                    probe["families"][name][key] = data["families"][name][key][indices].clone()
        explicit = PreparedPotentialNormalizedObjective(probe, calibration)
        taylor = PreparedPotentialADObjective(probe, calibration)
        native = PreparedPotentialADObjective(probe, calibration, coordinate_ad=True)
        topology = (4, 5, 4)
        theta = flatten(initialize(topology, 361901))
        for offset in (0., .007):
            parameters = views(theta + offset, topology)
            expected = native(parameters, 2)
            for objective in (explicit, taylor):
                for actual, reference in zip(objective(parameters, 2), expected):
                    compare(actual, reference)


if __name__ == "__main__":
    unittest.main()
