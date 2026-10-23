"""Synthetic checks of the registered v6 decision (box376.decide_c36_v6)."""
import json
import unittest
from pathlib import Path

from box376.decide_c36_v6 import decide

PROTOCOL = json.loads((Path(__file__).resolve().parents[2] / "protocol-v6-eta16-development-20261018.json").read_text())
RUNS = PROTOCOL["decision_rule"]["runs"]


def observation(passed=False, pressure=2e-3, endpoint=5e-4, velocity=2e-3):
    fields = {"velocity_relative_l2": velocity, "velocity_reference_is_zero": False, "normalized_velocity_error_rms": velocity,
              "normalized_divergence_rms": 1e-14, "normalized_momentum_rms": 1e-3, "normalized_momentum_gradient_rms": 1e-3,
              "normalized_divergence_gradient_rms": 1e-12, "normalized_corrected_pressure_rms": pressure,
              "normalized_pressure_gauge_rms": 0.0}
    return {"state": "COMPLETE", "physical_acceptance_pass": passed,
            "datasets": {"stratified": {"field_gate_pass": passed, "global": dict(fields)},
                         "uniform": {"field_gate_pass": True, "global": dict(fields, normalized_corrected_pressure_rms=pressure / 4)}},
            "initial_boundary_normalized_velocity_rms": [0.0, 0.0],
            "trajectory": {"maximum_endpoint_error": endpoint, "maximum_refinement_change": 1e-9}}


def reader(finals=None, extra=None, drop=()):
    runs = []
    for key in RUNS:
        engine, seed = key.split(":")
        physical = {} if key in drop else {"final": (finals or {}).get(key, observation())}
        physical.update((extra or {}).get(key, {}))
        runs.append({"engine": engine, "seed": int(seed), "training_state": "COMPLETE",
                     "training_gate_first": {"loss": 1e-5}, "physical": physical})
    return {"runs": runs}


class DecideV6Tests(unittest.TestCase):
    def key(self, data):
        return decide(PROTOCOL, data)["outcome"]["key"]

    def test_fail(self):
        result = decide(PROTOCOL, reader())
        self.assertEqual(result["outcome"]["key"], "F3_fail")
        self.assertAlmostEqual(result["outcome"]["best_final_worst_over_limit"], 2.0)
        self.assertEqual(result["runs"]["analytic:361901"]["final_worst_test"], "stratified:normalized_corrected_pressure_rms")

    def test_near_miss_counts_every_test(self):
        self.assertEqual(self.key(reader({"analytic:361902": observation(pressure=1.4e-3)})), "F2_near_miss")
        # The same pressure with a trajectory endpoint 1.6 times its limit is not a near miss.
        self.assertEqual(self.key(reader({"analytic:361902": observation(pressure=1.4e-3, endpoint=1.6e-3)})), "F3_fail")

    def test_pass_at_any_observed_checkpoint(self):
        result = decide(PROTOCOL, reader(extra={"parameter_ad:361901": {"gate-first": observation(passed=True, pressure=8e-4)}}))
        self.assertEqual(result["outcome"]["key"], "F1_pass")
        self.assertEqual(result["outcome"]["runs"], ["parameter_ad:361901"])
        self.assertTrue(result["follow_up"].startswith("register a formal"))

    def test_observation_not_run_is_incomplete(self):
        result = decide(PROTOCOL, reader(drop={"analytic:361901"}))
        self.assertEqual(result["runs_without_complete_final_observation"], ["analytic:361901"])
        self.assertEqual((result["outcome"]["key"], result["outcome"]["runs"]), ("F0_incomplete", ["analytic:361901"]))
        self.assertTrue(result["follow_up"].startswith("no scientific reading"))
        failed = reader(extra={"analytic:361902": {"final": {"state": "FAILED"}}})
        self.assertEqual(self.key(failed), "F0_incomplete")

    def test_training_failed_is_incomplete(self):
        data = reader()
        data["runs"] = [r for r in data["runs"] if (r["engine"], r["seed"]) != ("parameter_ad", 361902)]
        self.assertEqual(self.key(data), "F0_incomplete")
        killed = reader(drop={k: None for k in RUNS})
        self.assertEqual(self.key(killed), "F0_incomplete")

    def test_near_miss_and_pass_need_no_complete_set(self):
        self.assertEqual(self.key(reader({"analytic:361902": observation(pressure=1.2e-3)}, drop={"analytic:361901"})), "F2_near_miss")
        passing = reader(extra={"analytic:361902": {"gate-first": observation(passed=True, pressure=8e-4)}}, drop={"analytic:361901"})
        self.assertEqual(self.key(passing), "F1_pass")


if __name__ == "__main__":
    unittest.main()
