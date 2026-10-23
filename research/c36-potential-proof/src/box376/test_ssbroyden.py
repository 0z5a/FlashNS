"""CPU checks of the dense self-scaled Broyden optimizer (box376.ssbroyden)."""
import math
import unittest

import torch

from box376.ssbroyden import MODES, SSBroyden, parameters, update, updated_product


def random_case(n, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g)
    H = a @ a.T / n + 0.1 * torch.eye(n)
    s, y = torch.randn(n, generator=g), torch.randn(n, generator=g)
    if float(y @ s) <= 0:
        y = -y
    return H, s, y


def independent_parameters(b, h, n):
    """SSBroyden tau and phi written separately from Kiyani et al., arXiv:2501.16371, Eq. (15) definitions."""
    a = b * h - 1
    c = (a / (1 + a)) ** 0.5
    rho_minus = min(1, h * (1 - c))
    theta_minus = (rho_minus - 1) / a
    theta_plus = 1 / rho_minus
    theta = max(theta_minus, min(theta_plus, (1 - b) / b))
    rho_plus = min(1, 1 / b)
    sigma = 1 + theta * a
    sigma_power = abs(sigma) ** (1 / (1 - n))
    if theta <= 0:
        tau = min(rho_plus * sigma_power, sigma)
    else:
        tau = rho_plus * min(sigma_power, 1 / theta)
    phi = (1 - theta) / (1 + (h * b - 1) * theta)
    return tau, phi, theta


class SSBroydenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls.previous_dtype)

    def test_bfgs_update_matches_textbook(self):
        for seed in range(5):
            H, s, y = random_case(9, seed)
            rho = 1 / float(y @ s)
            eye = torch.eye(9)
            expected = (eye - rho * torch.outer(s, y)) @ H @ (eye - rho * torch.outer(y, s)) + rho * torch.outer(s, s)
            actual = H.clone()
            update(actual, s, y, H @ y, 1.0, 1.0)
            self.assertLess(float((actual - expected).abs().max() / expected.abs().max()), 1e-12)

    def test_parameters_match_independent_computation(self):
        branches = set()
        for seed in range(200):
            H, s, y = random_case(6, seed)
            b = float(s @ torch.linalg.solve(H, s)) / float(y @ s)
            h = float(y @ H @ y) / float(y @ s)
            tau, phi, details = parameters(b, h, 50692, 'ssbroyden')
            tau_i, phi_i, theta_i = independent_parameters(b, h, 50692)
            self.assertAlmostEqual(tau, tau_i, delta=1e-12 * max(1, abs(tau_i)))
            self.assertAlmostEqual(phi, phi_i, delta=1e-12 * max(1, abs(phi_i)))
            self.assertAlmostEqual(details['theta'], theta_i, delta=1e-12 * max(1, abs(theta_i)))
            branches.add(theta_i > 0)
            self.assertTrue(0 < tau <= 1 + 1e-15)
            self.assertGreater(1 + details['a'] * details['theta'], 0)  # sigma > 0 keeps phi finite
        self.assertEqual(branches, {True, False})
        self.assertEqual(parameters(2.0, 3.0, 10, 'bfgs')[:2], (1.0, 1.0))
        self.assertEqual(parameters(2.0, 3.0, 10, 'ssbfgs')[:2], (0.5, 1.0))

    def test_secant_positive_definiteness_and_carried_product(self):
        for mode in MODES:
            for seed in range(20):
                H, s, y = random_case(8, 100 + seed)
                Hy = H @ y
                b = float(s @ torch.linalg.solve(H, s)) / float(y @ s)
                h = float(y @ Hy) / float(y @ s)
                tau, phi, _ = parameters(b, h, 8, mode)
                g = torch.randn(8, generator=torch.Generator().manual_seed(seed))
                Hg = H @ g
                updated = H.clone()
                v = update(updated, s, y, Hy, tau, phi)
                # v.y = 0 exactly; in floating point its rounding is amplified by phi (y.Hy) |v|^2 |y| / tau.
                amplification = abs(phi) * float(y @ Hy) * float(v.norm()) ** 2 * float(y.norm()) / tau / float(s.norm())
                self.assertLess(float((updated @ y - s).norm() / s.norm()), 1e-14 * max(1.0, amplification), mode)
                self.assertGreater(float(torch.linalg.eigvalsh((updated + updated.T) / 2).min()), 0, mode)
                carried = updated_product(Hg, s, y, Hy, v, g, tau, phi)
                self.assertLess(float((carried - updated @ g).norm() / (updated @ g).norm()), 1e-11, mode)

    def run_optimizer(self, mode, f_and_g, x0, iterations):
        state = {'x': x0.clone()}

        def objective(x):
            state['x'] = x.clone()
            return f_and_g(x)

        def assign(x):
            state['x'] = x.clone()

        optimizer = SSBroyden(len(x0), 'cpu', mode=mode)
        f, g = f_and_g(x0)
        optimizer.start(x0, f, g)
        for _ in range(iterations):
            record = optimizer.step(objective, assign)
            self.assertTrue(torch.equal(state['x'], optimizer.x))
            if record['stop'] or float(optimizer.g.norm()) < 1e-12:
                break
        self.assertLess(optimizer.product_error(), 1e-8)
        return optimizer

    def test_convex_quadratic(self):
        gen = torch.Generator().manual_seed(7)
        q, _ = torch.linalg.qr(torch.randn(30, 30, generator=gen))
        A = q @ torch.diag(torch.logspace(0, 3, 30)) @ q.T
        rhs = torch.randn(30, generator=gen)
        solution = torch.linalg.solve(A, rhs)
        for mode in MODES:
            optimizer = self.run_optimizer(mode, lambda x: (0.5 * x @ A @ x - rhs @ x, A @ x - rhs), torch.zeros(30), 400)
            self.assertLess(float((optimizer.x - solution).norm() / solution.norm()), 1e-8, mode)

    def test_rosenbrock(self):
        def rosenbrock(x):
            f = (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
            g = torch.zeros_like(x)
            g[:-1] = -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
            g[1:] += 200 * (x[1:] - x[:-1] ** 2)
            return f, g
        x0 = torch.tensor([-1.2, 1.0] * 5)
        for mode in MODES:
            optimizer = self.run_optimizer(mode, rosenbrock, x0, 3000)
            self.assertLess(float((optimizer.x - 1).abs().max()), 1e-6, mode)


if __name__ == '__main__':
    unittest.main()
