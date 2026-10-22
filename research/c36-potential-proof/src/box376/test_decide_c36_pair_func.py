"""Synthetic tests of box376.decide_c36_pair_func."""
import unittest

from box376.decide_c36_pair_func import decide, pair_entry

SEEDS = [1, 2, 3, 4, 5]


def arm(t, ok=True):
    return {'independent_audit_pass': ok, 'threshold_reached': ok, 'physical_acceptance_pass': ok, 'active_training_seconds': t}


class PairFuncDecision(unittest.TestCase):
    def test_outcomes_use_the_func_arm(self):
        rows = [pair_entry(s, {'analytic': arm(1000.), 'func_jacfwd': arm(3500.)}) for s in SEEDS]
        r = decide(SEEDS, rows)
        self.assertEqual(r['outcome'], 'ANALYTIC_FASTER')
        self.assertAlmostEqual(r['geometric_mean_ratio'], 3.5)
        rows = [pair_entry(s, {'analytic': arm(1000.), 'func_jacfwd': arm(800.)}) for s in SEEDS]
        r = decide(SEEDS, rows)
        self.assertEqual((r['outcome'], r['pairs_favouring_torch_func']), ('TORCH_FUNC_FASTER', 5))

    def test_missing_func_arm_is_not_accepted(self):
        rows = [pair_entry(s, {'analytic': arm(1000.)}) for s in SEEDS]
        self.assertEqual(decide(SEEDS, rows)['outcome'], 'INSUFFICIENT_PAIRS')


if __name__ == '__main__':
    unittest.main()
