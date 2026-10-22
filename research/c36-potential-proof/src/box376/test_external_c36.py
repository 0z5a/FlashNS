"""CPU checks that each external backend reproduces the C36 network state and its parameter gradient."""
import os
import time
import unittest
from pathlib import Path

import torch
import torch_c36 as c36

from box376.data import flatten, initialize, views
from box376.external_c36 import RAW
from box376.frozen_data import load_cache
from box376.objective import ORIGIN, SCALE
from box376.potential_model import prepare_geometry

ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY = None


def setup(points=6, seed=361901):
    import json
    torch.set_default_dtype(torch.float64)
    protocol = json.loads((ROOT/'protocol-formal-pair-20261021.json').read_text())
    data = load_cache(ROOT/'data/training-source.npz')
    interior = torch.as_tensor(data['families']['interior']['points'][:points] if isinstance(data['families']['interior'], dict)
                               else data['interior_points'][:points])
    inputs, gate = prepare_geometry(interior, protocol['instance']['eta'])
    theta = flatten(initialize(tuple(protocol['topology']), seed)).requires_grad_(True)
    alpha, origin = interior.new_tensor(SCALE), interior.new_tensor(ORIGIN)

    def physical():
        leaves = views(theta, tuple(protocol['topology']))
        weight, bias = leaves[0]
        return [(weight * alpha, bias - (weight * alpha) @ origin)] + list(leaves[1:])
    return theta, physical, inputs, gate


class ExternalBackends(unittest.TestCase):
    def test_raw_state_and_gradient_match_c36(self):
        theta, physical, inputs, gate = setup()
        from box376.external_c36 import assert_axis_aligned
        assert_axis_aligned(inputs)
        scale = inputs.new_tensor([.01, .01, .01, .1])[None, :, None]
        weights = torch.linspace(.5, 1.5, 4 * 17, dtype=inputs.dtype).reshape(4, 17)

        def functional(raw):
            state = c36.curl_pressure(c36.multiply(gate, raw * scale))
            return (state.square() * weights).sum()

        reference, _ = c36.forward(physical(), inputs)
        ref_grad = torch.autograd.grad(functional(reference), theta)[0]
        for name, backend in RAW.items():
            if name == 'jet_collapsed' and not os.environ.get('FLASHNS_TORCHJET'):
                print('jet_collapsed skipped: pinned snapshot not available')
                continue
            started = time.perf_counter()
            raw = backend(physical(), inputs)
            grad = torch.autograd.grad(functional(raw), theta)[0]
            seconds = time.perf_counter() - started
            error = ((raw - reference).abs() / (1e-11 + 1e-9 * reference.abs())).max().item()
            gerror = ((grad - ref_grad).abs() / (1e-11 + 1e-9 * ref_grad.abs())).max().item()
            print(f'{name}: state {error:.3e}, gradient {gerror:.3e} of the budget, {seconds:.2f} s for {len(inputs)} points')
            self.assertLessEqual(error, 1, name)
            self.assertLessEqual(gerror, 1, name)


if __name__ == '__main__':
    unittest.main()
