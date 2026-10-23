"""Self-scaled Broyden quasi-Newton optimizer with a dense inverse Hessian (C36 v7).

The inverse-Hessian update is Eq. (10) with Eqs. (13)-(23) of Urban, Stefanou and Pons (J. Comput.
Phys. 523, 113656, 2025; arXiv:2405.04230), identical to Eq. (15) and its definitions in Kiyani et
al. (arXiv:2501.16371):

    H+ = (1/tau) [H - (Hy)(Hy)^T/(y.Hy) + phi (y.Hy) v v^T] + s s^T/(y.s),
    v  = s/(y.s) - Hy/(y.Hy),
    b  = s.H^{-1}s/(y.s) = -alpha s.g/(y.s),  h = y.Hy/(y.s),  a = h b - 1,  c = sqrt(a/(a+1)),
    rho- = min(1, h(1-c)),  theta- = (rho- - 1)/a,  theta+ = 1/rho-,
    theta = max(theta-, min(theta+, (1-b)/b)),  sigma = 1 + a theta,  rho+ = min(1, 1/b),
    tau = rho+ min(|sigma|^(1/(1-n)), 1/theta)   if theta > 0,
          min(rho+ |sigma|^(1/(1-n)), sigma)     otherwise,
    phi = (1 - theta)/(1 + a theta).

mode 'ssbfgs' uses tau = rho+ and phi = 1; mode 'bfgs' uses tau = phi = 1. H0 = I. The step length
comes from torch's strong-Wolfe line search (c1 = 1e-4, c2 = 0.9 by default, tolerance_change 0, as
the L-BFGS runs use), starting from a unit step. Each iteration needs one dense matrix-vector
product: with d = -H g_k, H y = H g_{k+1} + d, and H+ g_{k+1} follows from the update in O(n).
Safeguards, recorded per iteration: the update is skipped when y.s <= 1e-12 |y||s| (no curvature);
H is reset to I when a direction is not a descent direction or the line search gives no decrease;
two consecutive resets end the run.
"""
import math

import torch
from torch.optim.lbfgs import _strong_wolfe

MODES = ('ssbroyden', 'ssbfgs', 'bfgs')


def parameters(b, h, n, mode='ssbroyden'):
    """(tau, phi, details) for scalars b = s.H^{-1}s/(y.s) > 0 and h = y.Hy/(y.s) > 0."""
    if mode == 'bfgs':
        return 1.0, 1.0, {}
    rho_plus = min(1.0, 1.0 / b)
    if mode == 'ssbfgs':
        return rho_plus, 1.0, {'rho_plus': rho_plus}
    a = max(h * b - 1.0, 0.0)  # h b >= 1 for SPD H (Cauchy-Schwarz); clamp rounding
    c = math.sqrt(a / (a + 1.0))
    rho_minus = min(1.0, h * (1.0 - c))
    theta_plus = 1.0 / rho_minus
    theta_minus = (rho_minus - 1.0) / a if a > 1e-14 else -math.inf
    theta = max(theta_minus, min(theta_plus, (1.0 - b) / b))
    sigma = 1.0 + a * theta
    power = abs(sigma) ** (1.0 / (1.0 - n))
    tau = rho_plus * min(power, 1.0 / theta) if theta > 0 else min(rho_plus * power, sigma)
    phi = (1.0 - theta) / (1.0 + a * theta)
    return tau, phi, {'a': a, 'c': c, 'rho_minus': rho_minus, 'rho_plus': rho_plus, 'theta': theta, 'sigma': sigma}


def update(H, s, y, Hy, tau, phi):
    """In place: H <- (1/tau)[H - Hy Hy^T/yHy + phi yHy v v^T] + s s^T/ys, in one GEMM pass."""
    ys, yHy = float(y @ s), float(y @ Hy)
    v = s / ys - Hy / yHy
    U = torch.stack((Hy, v, s), 1)
    W = torch.stack((-Hy / (tau * yHy), phi * yHy / tau * v, s / ys), 1)
    H.addmm_(U, W.T, beta=1.0 / tau, alpha=1.0)
    return v


def updated_product(Hg, s, y, Hy, v, g, tau, phi):
    """H+ g from H g with the same update, in O(n)."""
    ys, yHy = float(y @ s), float(y @ Hy)
    return (Hg - Hy * (float(Hy @ g) / yHy) + v * (phi * yHy * float(v @ g))) / tau + s * (float(s @ g) / ys)


class SSBroyden:
    def __init__(self, n, device, dtype=torch.float64, mode='ssbroyden', c1=1e-4, c2=0.9, max_ls=25):
        assert mode in MODES
        self.n, self.mode, self.c1, self.c2, self.max_ls = n, mode, c1, c2, max_ls
        self.H = torch.eye(n, device=device, dtype=dtype)
        self.resets_in_a_row = 0
        self.iteration = 0

    def start(self, x, f, g):
        self.x, self.f, self.g = x.clone(), float(f), g.clone()
        self.Hg = self.H @ self.g

    def reset(self):
        self.H.zero_()
        self.H.diagonal().fill_(1.0)
        self.Hg = self.g.clone()

    def step(self, objective, assign):
        """One iteration. objective(x) -> (f, g) evaluates at x; assign(x) writes x into the parameters.

        The strong-Wolfe search may return a bracket point other than its last evaluation, so the
        accepted point is assigned explicitly; on return the parameters hold self.x.
        """
        record = {'iteration': self.iteration + 1}
        d = -self.Hg
        gtd = float(self.g @ d)
        if not gtd < 0:
            record['reset'] = 'not a descent direction'
            self.reset()
            d = -self.Hg
            gtd = float(self.g @ d)

        def directional(x, t, direction):
            f, g = objective(x + t * direction)
            return float(f), g.clone()

        f_new, g_new, t, evaluations = _strong_wolfe(directional, self.x, 1.0, d, self.f, self.g, gtd,
                                                     c1=self.c1, c2=self.c2, tolerance_change=0.0, max_ls=self.max_ls)
        t, f_new = float(t), float(f_new)
        record.update(step_length=t, line_search_evaluations=int(evaluations), loss_before=self.f, loss=f_new)
        self.iteration += 1
        if not f_new < self.f:
            record['reset'] = 'no decrease in the line search'
            self.resets_in_a_row += 1
            self.reset()
            assign(self.x)
            record['stop'] = self.resets_in_a_row >= 2
            return record
        self.resets_in_a_row = 0
        x_new = self.x + t * d
        s, y = x_new - self.x, g_new - self.g
        ys = float(y @ s)
        Hg_new = self.H @ g_new
        if ys > 1e-12 * float(y.norm() * s.norm()):
            Hy = Hg_new + d  # H y = H g_{k+1} - H g_k, and H g_k = -d
            b = -t * float(s @ self.g) / ys
            h = float(y @ Hy) / ys
            tau, phi, details = parameters(b, h, self.n, self.mode)
            v = update(self.H, s, y, Hy, tau, phi)
            Hg_new = updated_product(Hg_new, s, y, Hy, v, g_new, tau, phi)
            record.update(tau=tau, phi=phi, b=b, h=h, **{k: details[k] for k in ('theta', 'sigma') if k in details})
        else:
            record['update_skipped'] = 'no curvature'
        self.x, self.f, self.g, self.Hg = x_new, f_new, g_new, Hg_new
        assign(self.x)
        record['stop'] = False
        return record

    def product_error(self):
        """Relative difference between the carried H g and an explicit matrix-vector product."""
        explicit = self.H @ self.g
        return float((explicit - self.Hg).norm() / explicit.norm())
