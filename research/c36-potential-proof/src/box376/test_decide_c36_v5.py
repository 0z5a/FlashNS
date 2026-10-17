"""Synthetic checks of the registered v5 decision (box376.decide_c36_v5)."""
import json
import unittest
from pathlib import Path

from box376.decide_c36_v5 import decide

PROTOCOL = json.loads((Path(__file__).resolve().parents[2] / "protocol-v5-budget-development-20261015.json").read_text())
TAGS = {"LBFGS-call1600": 1600, "LBFGS-call2400": 2400, "LBFGS-call3200": 3200, "LBFGS-call4000": 4000, "final": 4800}
RUNS = PROTOCOL["decision_rule"]["runs"]


def summary(pass_at=None, pressure=None, drop=()):
    """pass_at[key]: smallest total call count that passes; pressure[key]: function of calls; drop: (key, tag) not observed."""
    rows = []
    for key, run in RUNS.items():
        first = (pass_at or {}).get(key)
        p = (pressure or {}).get(key, lambda c: 2e-3 * (1600 / c) ** .4)
        for tag, calls in TAGS.items():
            if (key, tag) in drop:
                continue
            rows.append({"id": f"{run}__{tag}", "tags": [f"{run}:{tag}"], "observation_state": "COMPLETE",
                         "LBFGS_calls": calls, "physical_acceptance_pass": first is not None and calls >= first,
                         "datasets": {"stratified": {"global": {"normalized_corrected_pressure_rms": p(calls),
                                                                 "velocity_relative_l2": 1e-3}}}})
    return {"rows": rows}


class DecideV5Tests(unittest.TestCase):
    def key(self, **kw):
        return decide(PROTOCOL, [summary(**kw)])["outcome"]["key"]

    def test_none_still_falling(self):
        result = decide(PROTOCOL, [summary()])
        self.assertEqual(result["outcome"]["key"], "E4_none_still_falling")
        self.assertTrue(result["outcome"]["stopped_falling_evaluable"])
        self.assertEqual(result["follow_up"], "no formal claim")

    def test_stopped_needs_both_seeds(self):
        flat = lambda c: 1.5e-3
        self.assertEqual(self.key(pressure={"analytic:361901": flat}), "E4_none_still_falling")
        self.assertEqual(self.key(pressure={"analytic:361901": flat, "analytic:361902": flat}), "E5_none_stopped")

    def test_all_pass_and_one_engine(self):
        result = decide(PROTOCOL, [summary(pass_at={k: 3200 for k in RUNS})])
        self.assertEqual(result["outcome"]["key"], "E1_all_pass")
        self.assertEqual(result["follow_up"], "freeze a formal protocol with new seeds")
        out = decide(PROTOCOL, [summary(pass_at={"analytic:361901": 2400, "analytic:361902": 4800})])["outcome"]
        self.assertEqual((out["key"], out["engine"]), ("E2_one_engine_passes", "analytic"))

    def test_partial(self):
        self.assertEqual(self.key(pass_at={"parameter_ad:361902": 4800}), "E3_partial")

    def test_call_800_and_missing_tags(self):
        extra = {"rows": [{"id": "x", "tags": ["seed361901-analytic:LBFGS-call800"], "observation_state": "COMPLETE",
                           "LBFGS_calls": 800, "physical_acceptance_pass": False,
                           "datasets": {"stratified": {"global": {"normalized_corrected_pressure_rms": 2.5e-3,
                                                                   "velocity_relative_l2": 1.3e-3}}}}]}
        result = decide(PROTOCOL, [summary(drop={("analytic:361902", "final")}), extra])
        run = result["runs"]["seed361901-analytic"]
        self.assertEqual(run["observed_tags"][0], "LBFGS-call800")
        self.assertEqual(result["runs"]["seed361902-analytic"]["missing_tags"], ["final"])
        self.assertIsNone(result["stopped_falling"]["analytic"])


if __name__ == "__main__":
    unittest.main()
