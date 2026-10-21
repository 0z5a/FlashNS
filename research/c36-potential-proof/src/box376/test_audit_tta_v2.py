"""CPU tests of the rounding bound in box376.audit_c36_tta_v2.

The v1 auditor failed on arrays saved on an ARM GPU host and recomputed on x86. These tests model
that failure: terms of about 1e5 that cancel to about 1e-2, with the other host's result a few ulp
of the largest term away. They are fixtures, not physical evidence."""
import unittest

import numpy as np

from box376.audit_c36_tta_v2 import GAMMA, metrics, rounding_bounds, same_metrics

SCALES = {'velocity': .377, 'continuity_residual': 1., 'continuity_residual_gradient': 1.,
          'momentum_residual': 164., 'momentum_residual_gradient': 50451., 'pressure': .142}


def cancelling_case(n=256, seed=0):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, 4, 17)) * np.r_[1., 1e2, 1e2, 1e2, np.full(6, 1e3), np.full(3, 1e4), 1e2, np.full(3, 1e5)]
    reference = {'U': rng.normal(size=(n, 3)), 'f': np.zeros((n, 3)), 'df': np.zeros((n, 3, 3))}
    _, raw, _ = metrics(state, reference, np.ones(n), np.zeros(n), SCALES)
    # Choose the force so that both residuals cancel to about 1e-2, as in the observed arrays.
    reference['f'] = raw['residual'] - 1e-2 * rng.normal(size=(n, 3))
    reference['df'] = raw['spatial_residual_gradient'] - 1e-2 * rng.normal(size=(n, 3, 3))
    return state, reference, np.ones(n), np.zeros(n)


class RoundingBound(unittest.TestCase):
    def test_bound_dominates_every_result(self):
        state, reference, w, gauge = cancelling_case()
        _, raw, _ = metrics(state, reference, w, gauge, SCALES)
        bounds = rounding_bounds(state, reference)
        for key, array in raw.items():
            self.assertTrue(np.all(np.abs(array) <= bounds[key] / (2 * GAMMA)), key)

    def test_reordered_evaluation_within_bound(self):
        state, reference, w, gauge = cancelling_case(seed=1)
        _, raw, _ = metrics(state, reference, w, gauge, SCALES)
        h = np.empty((len(state), 4, 3, 3))
        for k, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
            h[:, :, i, j] = h[:, :, j, i] = state[:, :, 4+k]
        u, g = state[:, :3, 0], state[:, :3, 1:4]
        # Same terms, opposite order of addition.
        other = (((h[:, 3] - reference['df']) - .01*state[:, :3, 10:13]) + np.einsum('nj,nijk->nik', u, h[:, :3])
                 + np.einsum('nij,njk->nik', g, g) + state[:, :3, 14:17])
        err = np.abs(other - raw['spatial_residual_gradient'])
        self.assertTrue(np.all(err <= rounding_bounds(state, reference)['spatial_residual_gradient']))

    def test_few_ulp_difference_fails_v1_and_passes_v2(self):
        state, reference, w, gauge = cancelling_case(seed=2)
        m, raw, _ = metrics(state, reference, w, gauge, SCALES)
        bounds = rounding_bounds(state, reference)
        a = raw['spatial_residual_gradient']
        largest = np.maximum.reduce([np.abs(state[:, :3, 14:17]), np.abs(reference['df'])])
        b = a + 4 * np.spacing(largest) * np.sign(np.random.default_rng(3).normal(size=a.shape))
        self.assertFalse(np.allclose(b, a, rtol=1e-10, atol=1e-13))
        self.assertTrue(np.all(np.abs(b - a) <= bounds['spatial_residual_gradient']))
        moved = dict(m, normalized_momentum_gradient_rms=float(np.sqrt(np.mean(np.sum(b.reshape(len(b), -1)**2, axis=1) / 9))) / SCALES['momentum_residual_gradient'])
        same_metrics(m, moved, bounds, w, SCALES, 'fixture')

    def test_difference_beyond_bound_fails(self):
        state, reference, w, gauge = cancelling_case(seed=4)
        _, raw, _ = metrics(state, reference, w, gauge, SCALES)
        bounds = rounding_bounds(state, reference)
        b = raw['residual'] + 1.5 * bounds['residual']
        self.assertFalse(np.all(np.abs(b - raw['residual']) <= bounds['residual']))

    def test_metric_beyond_bound_fails(self):
        state, reference, w, gauge = cancelling_case(seed=5)
        m, _, _ = metrics(state, reference, w, gauge, SCALES)
        bounds = rounding_bounds(state, reference)
        tol = float(np.sqrt(np.mean(np.mean(bounds['divergence'].reshape(len(w), -1)**2, axis=1)))) / SCALES['continuity_residual']
        moved = dict(m, normalized_divergence_rms=m['normalized_divergence_rms'] + 2 * tol + 1e-9)
        with self.assertRaises(AssertionError):
            same_metrics(m, moved, bounds, w, SCALES, 'fixture')


if __name__ == '__main__':
    unittest.main()
