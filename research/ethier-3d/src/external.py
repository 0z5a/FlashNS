"""External three-dimensional baselines for the Ethier--Steinman objective (protocol v4).

Both arms compute the unchanged objective of `objective.reference_objective`: momentum and divergence
residuals, lambda_g times the squared spatial gradient of every residual, the boundary, initial and
gauge value terms, and all 4,740 parameter gradients. Neither uses FlashNS code.

- torch_ad: nested scalar-coordinate PyTorch autograd, as an ordinary PINN code writes it and as the
  2D `torch_ad` adapter does. Four first-derivative passes, nine for the three spatial Laplacians,
  then one per residual for its spatial gradient.
- jet_collapsed: Dangel et al.'s public `jet` package (collapsed Taylor mode, pinned snapshot). One
  collapsed second-order jet along the three spatial directions gives the spatial first derivatives
  and the summed spatial Laplacian; a first-order jet along t gives the time derivative. As in the
  2D adapter, the residual gradient and the parameter gradient use torch autograd.

`evaluate(parameters, data, protocol, arm)` copies the parameters into a private module, as the A24
and naive arms of v4 copy them, so a CUDA Graph captured around it reads the live parameter storage
on replay.
"""
import os
import sys
from pathlib import Path

import torch

from objective import values

JET_ROOT = os.environ.get('FLASHNS_TORCHJET', str(Path(__file__).resolve().parent / 'snapshots' / 'torchjet'))
ARMS = ('torch_ad', 'jet_collapsed')


class MLP(torch.nn.Module):
    """The v4 4-64-64-4 tanh network on a private copy of the given weights and biases."""

    def __init__(self, parameters):
        super().__init__()
        self.depth = len(parameters)
        for k, (W, b) in enumerate(parameters):
            self.register_parameter(f'W{k}', torch.nn.Parameter(W.detach().clone()))
            self.register_parameter(f'b{k}', torch.nn.Parameter(b.detach().clone()))

    def pairs(self):
        return [(getattr(self, f'W{k}'), getattr(self, f'b{k}')) for k in range(self.depth)]

    def forward(self, x):
        for k in range(self.depth):
            x = x @ getattr(self, f'W{k}').T + getattr(self, f'b{k}')
            if k < self.depth - 1:
                x = torch.tanh(x)
        return x


def _grad(z, x):
    return torch.autograd.grad(z, x, torch.ones_like(z), create_graph=True, retain_graph=True)[0]


def derivatives_torch_ad(model, x):
    """y [N,4]; spatial first derivatives D [N,4,3]; time derivative [N,4]; spatial Laplacian of u,v,w [N,3]."""
    y = model(x)
    first = [_grad(y[:, c], x) for c in range(4)]
    D = torch.stack([g[:, :3] for g in first], 1)
    dt = torch.stack([g[:, 3] for g in first], 1)
    lap = torch.stack([sum(_grad(first[c][:, j], x)[:, j] for j in range(3)) for c in range(3)], 1)
    return y, D, dt, lap


def derivatives_jet(model, x):
    if JET_ROOT not in sys.path:
        sys.path.insert(0, JET_ROOT)
    from jet import jet
    # Built per call: the jets bind this module's parameters. Under CUDA Graph replay this Python
    # runs only during warmup and capture.
    collapsed, standard = jet(model, (x,), collapsed=True), jet(model, (x,))
    eye = torch.eye(4, dtype=x.dtype, device=x.device)
    directions = eye[:3, None, :].expand(3, len(x), 4)
    y, first, lap = collapsed((x, directions, torch.zeros_like(x)))
    _, dt = standard((x, eye[3].expand(len(x), 4)))
    return y, first.permute(1, 2, 0), dt, lap[:, :3]


def physical_terms(y, D, dt, lap, x, nu, lambda_g):
    """Mean over interior points of sum R^2 + lambda_g * sum |grad_x R|^2 (zero force)."""
    R = [dt[:, i] + sum(y[:, j] * D[:, i, j] for j in range(3)) - nu * lap[:, i] + D[:, 3, i] for i in range(3)]
    R.append(sum(D[:, i, i] for i in range(3)))
    G = [_grad(r, x)[:, :3] for r in R]
    return (sum(r * r for r in R) + lambda_g * sum((g * g).sum(-1) for g in G)).mean()


def evaluate(parameters, data, protocol, arm):
    model = MLP(parameters)
    p = model.pairs()
    x = data['interior'].detach().clone().requires_grad_(True)
    derivatives = derivatives_torch_ad if arm == 'torch_ad' else derivatives_jet
    with torch.enable_grad():
        y, D, dt, lap = derivatives(model, x)
        loss = physical_terms(y, D, dt, lap, x, protocol['parameters']['nu'], protocol['lambda_g'])
        for key, idx, coef in [('boundary', slice(0, 3), protocol['boundary_weight']), ('initial', slice(0, 3), protocol['initial_weight']),
                               ('gauge', slice(3, 4), protocol['gauge_weight'])]:
            diff = values(p, data[key])[:, idx] - data[key + '_target'][:, idx]
            loss = loss + coef * (diff * diff).sum() / len(diff)
        g = torch.autograd.grad(loss, [v for layer in p for v in layer])
    return loss.detach(), [(g[i], g[i + 1]) for i in range(0, len(g), 2)]
