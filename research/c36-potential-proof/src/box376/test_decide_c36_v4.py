"""Synthetic checks of the registered v4 decision (box376.decide_c36_v4)."""
import json
import unittest
from pathlib import Path

from box376.decide_c36_v4 import decide

PROTOCOL = json.loads((Path(__file__).resolve().parents[2] / "protocol-v4-combination-development-20261015.json").read_text())
TAGS = {"LBFGS-call800": 800, "LBFGS-call1600": 1600, "LBFGS-call2400": 2400, "LBFGS-call3200": 3200, "final": 4000}


def summary(pass_at=None, pressure=None):
    """pass_at[(arm, seed)]: smallest extra-call count that passes (None: never);
    pressure[(arm, seed)]: function of extra calls."""
    rows = []
    for arm in ("baseline", "C2"):
        for seed in PROTOCOL["seeds"]:
            first = (pass_at or {}).get((arm, seed))
            p = (pressure or {}).get((arm, seed), lambda c: 2e-3 * (800 / c) ** .3)
            for tag, calls in TAGS.items():
                rows.append({"id": f"seed{seed}-resume-{arm}__{tag}", "tags": [f"seed{seed}-resume-{arm}:{tag}"],
                             "observation_state": "COMPLETE", "LBFGS_calls": calls,
                             "physical_acceptance_pass": first is not None and calls >= first,
                             "datasets": {"stratified": {"global": {"normalized_corrected_pressure_rms": p(calls),
                                                                     "velocity_relative_l2": 1e-3}}}})
    return {"rows": rows}


S1, S2 = PROTOCOL["seeds"]


class DecideTests(unittest.TestCase):
    def outcome(self, **kw):
        return decide(PROTOCOL, summary(**kw))["outcome"]

    def test_none_still_falling(self):
        result = decide(PROTOCOL, summary())
        self.assertEqual(result["outcome"]["key"], "O5_none_still_falling")
        self.assertTrue(result["outcome"]["stopped_falling_evaluable"])
        self.assertEqual(result["follow_up"], "no formal claim")

    def test_none_stopped(self):
        flat = lambda c: 2e-3 if c >= 3200 else 3e-3
        out = self.outcome(pressure={("C2", S1): flat, ("C2", S2): flat})
        self.assertEqual(out["key"], "O6_none_stopped")

    def test_one_seed_flat_is_not_stopped(self):
        flat = lambda c: 2e-3 if c >= 3200 else 3e-3
        self.assertEqual(self.outcome(pressure={("C2", S1): flat})["key"], "O5_none_still_falling")

    def test_both_pass_and_order(self):
        out = self.outcome(pass_at={("baseline", S1): 3200, ("baseline", S2): 2400, ("C2", S1): 1600, ("C2", S2): 1600})
        self.assertEqual((out["key"], out["sub_result"]), ("O1_both_arms_pass", "C2_sooner"))
        out = self.outcome(pass_at={("baseline", S1): 1600, ("baseline", S2): 2400, ("C2", S1): 1600, ("C2", S2): 3200})
        self.assertEqual((out["key"], out["sub_result"]), ("O1_both_arms_pass", "same_or_mixed"))

    def test_single_arm_and_partial(self):
        self.assertEqual(self.outcome(pass_at={("C2", S1): 2400, ("C2", S2): 4000, ("baseline", S1): 4000})["key"],
                         "O2_only_C2_passes")
        self.assertEqual(self.outcome(pass_at={("baseline", S1): 2400, ("baseline", S2): 4000})["key"],
                         "O3_only_baseline_passes")
        result = decide(PROTOCOL, summary(pass_at={("C2", S1): 800}))
        self.assertEqual(result["outcome"]["key"], "O4_partial")
        self.assertEqual(result["follow_up"], "no formal claim")

    def test_secondary_band(self):
        result = decide(PROTOCOL, summary(pressure={("C2", S1): lambda c: 1.25e-3}))
        pair = result["secondary_stratified_pressure"]["final"][str(S1)]
        self.assertFalse(pair["interpreted"])
        result = decide(PROTOCOL, summary(pressure={("C2", S1): lambda c: 1e-4}))
        self.assertTrue(result["secondary_stratified_pressure"]["final"][str(S1)]["interpreted"])


if __name__ == "__main__":
    unittest.main()
