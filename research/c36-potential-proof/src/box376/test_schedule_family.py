"""CPU checks of the temporal-smoothness family (box376.schedule_family)."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

import box376.potential_geometry as geometry_module
import box376.source as source_module
from box376 import schedule_family as family
from box376.data import FAMILIES, flatten, initialize, torch_data, views
from box376.frozen_data import load_cache
from box376.normalized_objective import normalization_key
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_geometry import features_and_gate
from box376.potential_model import prepare_geometry
from box376.potential_objective import PreparedPotentialNormalizedObjective

DATA = Path(__file__).resolve().parents[2] / "data"
TOPOLOGY = (4, 128, 128, 128, 128, 4)


def problem(subset=None):
    calibration = json.loads((DATA / "calibration.json").read_text())
    data = torch_data(load_cache(DATA / "training-source.npz"))
    data["normalization_key"] = normalization_key(calibration)
    if subset:
        for name in FAMILIES:
            fam = data["families"][name]
            keep = min(subset, len(fam["points"]))
            for key, value in list(fam.items()):
                if isinstance(value, torch.Tensor) and value.shape[:1] == fam["points"].shape[:1]:
                    fam[key] = value[:keep].clone()
    return data, calibration


class ScheduleFamilyTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.t = np.r_[np.linspace(0, 1, 2001), .25, .75, .5, .2500000001, .7499999999]

    def test_original_width_motion_is_bitwise(self):
        expected = source_module.motion(self.t)
        actual = family.motion(self.t, family.ORIGINAL_WIDTH)
        for key, value in expected.items():
            self.assertTrue(np.array_equal(actual[key], value), key)

    def test_swapped_source_is_bitwise_at_original_width(self):
        data, _ = problem()
        points = data["families"]["interior"]["points"][:257].numpy()
        expected = source_module.source(points)
        with family._swapped(source_module, "motion", lambda t: family.motion(t, family.ORIGINAL_WIDTH)):
            actual = source_module.source(points)
        for key, value in expected.items():
            self.assertTrue(np.array_equal(actual[key], value), key)

    def test_geometry_copies_are_bitwise_at_original_width(self):
        data, _ = problem()
        points = data["families"]["interior"]["points"][:300]
        f0, g0 = features_and_gate(points)
        f1, g1 = family.features_and_gate_w(points, family.ORIGINAL_WIDTH)
        self.assertTrue(torch.equal(f0, f1) and torch.equal(g0, g1))
        i0, gate0 = prepare_geometry(points, 1/32, chunk_size=64)
        i1, gate1 = family.prepare_geometry_w(points, 1/32, width=family.ORIGINAL_WIDTH, chunk_size=64)
        self.assertTrue(torch.equal(i0, i1) and torch.equal(gate0, gate1))

    def test_smoothness_scaling_and_window(self):
        reference = family.motion(self.t, .5)
        for width in (.75, 1.):
            m = family.motion(self.t, width)
            ratio = np.abs(m["sp"]).max() / np.abs(reference["sp"]).max()
            self.assertAlmostEqual(ratio, .5 / width, delta=2e-3)
            low, high = family.window(width)
            outside = (self.t <= low) | (self.t >= high)
            self.assertTrue(np.all(m["sp"][outside] == 0) and np.all(m["spp"][outside] == 0))
            self.assertTrue(np.allclose(m["s"][self.t >= high], 1) and np.all(m["s"][self.t <= low] == 0))

    def test_width_data_keeps_value_targets_and_changes_forcing(self):
        data, _ = problem()
        for width in (.75, 1.):
            changed, record = family.data_at_width(data, width)
            self.assertEqual(record["width"], width)
            force = changed["families"]["interior"]["force"]
            self.assertFalse(torch.equal(force, data["families"]["interior"]["force"]))
            self.assertTrue(torch.isfinite(force).all())

    def test_width_objective_engines_and_native_oracle_agree(self):
        data, calibration = problem(subset=24)
        theta = flatten(initialize(TOPOLOGY, 361901))
        params = views(theta, TOPOLOGY)
        width = .75
        analytic, _ = family.prepared_objective(PreparedPotentialNormalizedObjective, data, calibration, width)
        dense, _ = family.prepared_objective(PreparedPotentialADObjective, data, calibration, width)
        with family._swapped(geometry_module, "features_and_gate",
                             lambda p, eta=1/32: family.features_and_gate_w(p, width, eta)):
            native, _ = family.prepared_objective(PreparedPotentialADObjective, data, calibration, width,
                                                  coordinate_ad=True)
            reference = native(params, 24)
        for engine in (analytic, dense):
            for a, b in zip(engine(params, 24), reference):
                error = (a - b).abs() / (1e-11 + 1e-9 * b.abs())
                self.assertLessEqual(error.max().item(), 1.0)
        original, _ = family.prepared_objective(PreparedPotentialNormalizedObjective, data, calibration, .5)
        self.assertNotEqual(original(params, 24)[0].item(), reference[0].item())


if __name__ == "__main__":
    unittest.main()
