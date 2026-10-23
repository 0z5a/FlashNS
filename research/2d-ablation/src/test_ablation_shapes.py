"""CPU checks of the width/depth wrapper (ablation_bench.py).

At the original topology the generic views equal problem.views exactly. At other topologies the
compact GL8 state (explicit output-state VJP and generic autograd) and compact Q10 agree with the
dense Q10 oracle on the full loss and every parameter gradient (atol 2e-12, rtol 2e-11).
"""
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pinn_solver"))

import problem  # noqa: E402
from ablation_bench import ORIGINAL, generic_views  # noqa: E402
from flashns.pinn_contracted_packed import ContractedPackedPINN  # noqa: E402
from flashns.pinn_packed import PackedPINN  # noqa: E402

SHAPES = [(2, 32, 32, 3), (2, 128, 128, 3), (2, 64, 64, 64, 3), (2, 64, 64, 64, 64, 3)]


def flatten(gradients):
    return torch.cat([g.flatten() for g in gradients])


class AblationShapeTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.saved = problem.WIDTHS

    def tearDown(self):
        problem.WIDTHS = self.saved

    def test_original_views_identical(self):
        flat = problem.parameters(361901)
        w0, b0 = problem.views(flat)
        w1, b1 = generic_views(ORIGINAL)(flat)
        for a, b in zip(w0 + b0, w1 + b1):
            self.assertTrue(torch.equal(a, b) and a.data_ptr() == b.data_ptr() and a.shape == b.shape)

    def test_shapes_against_dense_q10(self):
        task = problem.Problem(interior_count=96, device="cpu")
        ni = task.interior_count
        data = (task.x, ni, task.pde_weights[:ni], task.boundary_weights[ni:], task.target[ni:])
        oracle = PackedPINN(*data, layout="dense")
        for widths in SHAPES:
            with self.subTest(widths=widths):
                problem.WIDTHS = widths
                flat = problem.parameters(361901)
                self.assertEqual(flat.numel(), sum(i * o + o for i, o in zip(widths, widths[1:])))
                views = generic_views(widths)
                expected_loss, expected = oracle(*views(flat))
                expected = flatten(expected)
                engines = {"gl8_compact_output": ContractedPackedPINN(*data, vjp="output"),
                           "q10_compact": PackedPINN(*data, layout="compact")}
                for name, engine in engines.items():
                    loss, gradients = engine(*views(flat))
                    torch.testing.assert_close(loss, expected_loss, atol=2e-12, rtol=2e-11, msg=name)
                    torch.testing.assert_close(flatten(gradients), expected, atol=2e-12, rtol=2e-11, msg=name)
                autograd = ContractedPackedPINN(*data, vjp="output")
                leaf = flat.clone().requires_grad_(True)
                loss = autograd.autograd_loss(*views(leaf))
                loss.backward()
                torch.testing.assert_close(loss.detach(), expected_loss, atol=2e-12, rtol=2e-11)
                torch.testing.assert_close(leaf.grad, expected, atol=2e-12, rtol=2e-11)


if __name__ == "__main__":
    unittest.main()
