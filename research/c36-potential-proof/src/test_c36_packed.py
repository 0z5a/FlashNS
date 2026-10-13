"""Compare packed nonlinear identities and their transposes with the dense path."""
import unittest

import torch

from torch_c36 import activate, state_vjp
from torch_c36_packed import activate_packed


class PackedActivationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
