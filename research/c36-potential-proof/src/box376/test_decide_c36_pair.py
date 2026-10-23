"""Synthetic tests of box376.decide_c36_pair."""
import unittest

from box376.decide_c36_pair import decide, pair_entry

SEEDS = [1, 2, 3, 4, 5]


def arm(t, audit=True, reached=True, physical=True):
    return {'independent_audit_pass': audit, 'threshold_reached': reached, 'physical_acceptance_pass': physical,
            'active_training_seconds': t}


def entries(ratios):
    return [pair_entry(s, {'analytic': arm(1000.), 'parameter_ad': arm(1000. * r)}) for s, r in zip(SEEDS, ratios)]


class PairDecision(unittest.TestCase):
    def test_analytic_faster(self):
        r = decide(SEEDS, entries([1.25, 1.3, 1.2, 1.28, 0.99]))
        self.assertEqual(r['outcome'], 'ANALYTIC_FASTER')
        self.assertEqual(r['pairs_favouring_analytic'], 4)

    def test_parameter_ad_faster(self):
        self.assertEqual(decide(SEEDS, entries([0.8, 0.85, 0.9, 0.95, 1.01]))['outcome'], 'PARAMETER_AD_FASTER')

    def test_small_or_split_effect_is_undecided(self):
        self.assertEqual(decide(SEEDS, entries([1.02, 1.01, 1.03, 1.02, 1.01]))['outcome'], 'NO_DECIDED_DIFFERENCE')
        self.assertEqual(decide(SEEDS, entries([1.4, 1.4, 1.4, 0.9, 0.9]))['outcome'], 'NO_DECIDED_DIFFERENCE')

    def test_unaccepted_pairs_are_excluded_and_can_leave_too_few(self):
        rows = entries([1.3, 1.3, 1.3, 1.3])
        rows.append(pair_entry(5, {'analytic': arm(1000.), 'parameter_ad': arm(1300., physical=False)}))
        r = decide(SEEDS, rows)
        self.assertEqual((r['outcome'], r['jointly_accepted_pairs']), ('ANALYTIC_FASTER', 4))
        rows[0] = pair_entry(1, {'analytic': arm(1000., audit=False), 'parameter_ad': arm(1300.)})
        self.assertEqual(decide(SEEDS, rows)['outcome'], 'INSUFFICIENT_PAIRS')

    def test_missing_seed_counts_against_and_duplicates_fail(self):
        r = decide(SEEDS, entries([1.3, 1.3, 1.3]))
        self.assertEqual(r['outcome'], 'INSUFFICIENT_PAIRS')
        self.assertEqual(r['denominator'], 5)
        with self.assertRaises(ValueError):
            decide([1, 1, 2, 3, 4], entries([1.3]))
        with self.assertRaises(ValueError):
            pair_entry(1, {'analytic': arm(0.), 'parameter_ad': arm(1.)})


if __name__ == '__main__':
    unittest.main()
