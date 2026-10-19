"""Synthetic checks of the registered v7 decision (box376.decide_c36_v7)."""
import json
import unittest
from pathlib import Path

from box376.decide_c36_v7 import decide

PROTOCOL = json.loads((Path(__file__).resolve().parents[2] / "protocol-v7-ssbroyden-development-20261019.json").read_text())
RUNS = PROTOCOL["decision_rule"]["runs"]
THRESHOLDS = [f"{t:g}" for t in PROTOCOL["training"]["loss_thresholds"]]


def observation(passed=False, pressure=2e-3, endpoint=5e-4):
    fields = {"velocity_relative_l2": 2e-3, "velocity_reference_is_zero": False, "normalized_velocity_error_rms": 2e-3,
              "normalized_divergence_rms": 1e-14, "normalized_momentum_rms": 1e-3, "normalized_momentum_gradient_rms": 1e-3,
              "normalized_divergence_gradient_rms": 1e-12, "normalized_corrected_pressure_rms": pressure,
              "normalized_pressure_gauge_rms": 0.0}
    return {"state": "COMPLETE", "physical_acceptance_pass": passed,
            "datasets": {"stratified": {"field_gate_pass": passed, "global": dict(fields)},
                         "uniform": {"field_gate_pass": True, "global": dict(fields, normalized_corrected_pressure_rms=pressure / 4)}},
            "initial_boundary_normalized_velocity_rms": [0.0, 0.0],
            "trajectory": {"maximum_endpoint_error": endpoint, "maximum_refinement_change": 1e-9}}


def reader(finals=None, extra=None, drop=(), hits=None):
    runs = []
    for key in RUNS:
        arm, seed = key.split(":")
        physical = {} if key in drop else {"final": (finals or {}).get(key, observation())}
        physical.update((extra or {}).get(key, {}))
        first = {t: ({"evaluation": (hits or {}).get(t)} if (hits or {}).get(t) else None) for t in THRESHOLDS}
        runs.append({"arm": arm, "seed": int(seed), "training_state": "COMPLETE", "training_gate_first": {"loss": 1e-5},
                     "physical": physical, "final_loss": 5e-7, "continuity": {"start_loss": 1.5e-6} if arm == "continue" else None,
                     "loss_curve": {"first_evaluation_at_or_below": first}})
    return {"runs": runs}


class DecideV7Tests(unittest.TestCase):
    def key(self, data):
        return decide(PROTOCOL, data)["outcome"]["key"]

    def test_outcomes_in_registered_order(self):
        self.assertEqual(self.key(reader()), "F3_fail")
        self.assertEqual(self.key(reader({"fresh:361902": observation(pressure=1.4e-3)})), "F2_near_miss")
        self.assertEqual(self.key(reader(extra={"continue:361901": {"SSBroyden-iteration10000": observation(passed=True, pressure=8e-4)}})), "F1_pass")
        self.assertEqual(self.key(reader(drop={"fresh:361901"})), "F0_incomplete")
        self.assertEqual(self.key(reader({"fresh:361902": observation(pressure=1.2e-3)}, drop={"fresh:361901"})), "F2_near_miss")

    def test_secondary_readouts(self):
        result = decide(PROTOCOL, reader(hits={"0.001": 4100, "1e-05": 9000, "1e-06": 20000}))
        fresh = result["runs"]["fresh:361901"]["first_evaluation_versus_v5"]
        self.assertAlmostEqual(fresh["1e-05"]["v7_over_v5"], 9000 / 12487)
        self.assertEqual((fresh["1e-06"]["v7"], fresh["1e-06"]["v5"], fresh["1e-06"]["v7_over_v5"]), (20000, None, None))
        self.assertIsNone(fresh["3e-07"]["v7"])
        self.assertAlmostEqual(result["runs"]["continue:361902"]["final_over_start_loss"], 5e-7 / 1.5e-6)
        self.assertNotIn("first_evaluation_versus_v5", result["runs"]["continue:361901"])


if __name__ == "__main__":
    unittest.main()
