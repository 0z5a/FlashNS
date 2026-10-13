"""Compare packed nonlinear identities and their transposes with the dense path."""
import unittest

import torch

from torch_c36 import activate, state_vjp
from torch_c36_packed import activate_packed
from torch_c36_packed_vjp import activate_packed_vjp


class PackedActivationTests(unittest.TestCase):
    def test_analytic_transpose_at_saturation_and_absolute_value_origin(self):
        state = torch.randn(9, 5, 36, dtype=torch.float64) * .1
        state[:, :, 0] = torch.tensor([-40., -20., -1e-12, -1e-16, 0., 1e-16, 1e-12, 20., 40.])[:, None]
        cotangent = torch.randn_like(state) * 1e8
        torch.testing.assert_close(activate_packed_vjp(state, cotangent),
                                   state_vjp(activate, state, cotangent), rtol=1e-9, atol=1e-11)

    def test_values_and_complete_state_transpose(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(361901)
        for shape in ((3, 5, 36), (0, 4, 36), (2, 36)):
            for scale in (0., .1, 1., 3.):
                state = (torch.randn(shape[:-1] + (72,), generator=generator, dtype=torch.float64) * scale)[..., ::2]
                cotangent = torch.randn(shape, generator=generator, dtype=torch.float64)
                torch.testing.assert_close(activate_packed(state), activate(state), rtol=1e-9, atol=1e-11)
                torch.testing.assert_close(state_vjp(activate_packed, state, cotangent),
                                           state_vjp(activate, state, cotangent), rtol=1e-9, atol=1e-11)
                torch.testing.assert_close(activate_packed_vjp(state, cotangent),
                                           state_vjp(activate, state, cotangent), rtol=1e-9, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
