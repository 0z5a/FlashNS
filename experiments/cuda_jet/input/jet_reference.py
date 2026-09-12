"""FP64 reference for factorial-normalized Taylor jets and explicit parameter VJPs.

This is a correctness oracle, not a GPU benchmark or an optimized CUDA backend.
Requires Python >= 3.10 and PyTorch. Tests use CPU float64 only.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import factorial
import json
from pathlib import Path
import torch

Tensor = torch.Tensor

@dataclass(frozen=True)
class JetPlan:
    dim: int
    order: int
    indices: tuple[tuple[int, ...], ...]

    @classmethod
    def dense(cls, dim: int, order: int = 3) -> 'JetPlan':
        if dim < 1 or not 0 <= order <= 3:
            raise ValueError('This reference supports dim >= 1 and order 0..3.')
        inds = tuple(sorted((a for a in product(range(order + 1), repeat=dim)
                             if sum(a) <= order), key=lambda a: (sum(a), a)))
        return cls(dim, order, inds)

    @property
    def q(self) -> int:
        return len(self.indices)

    @property
    def lookup(self) -> dict[tuple[int, ...], int]:
        return {a: i for i, a in enumerate(self.indices)}

    def pairs(self, a: tuple[int, ...]) -> list[tuple[int, int]]:
        lookup = self.lookup
        return [(j, lookup[tuple(ai - bi for ai, bi in zip(a, b))])
                for j, b in enumerate(self.indices)
                if all(bi <= ai for ai, bi in zip(a, b))]


def multiply(a: Tensor, b: Tensor, plan: JetPlan) -> Tensor:
    """Cauchy product of factorial-normalized jets, shape [B,Q,C]."""
    if a.shape != b.shape or a.ndim != 3 or a.shape[1] != plan.q:
        raise ValueError('Expected matching [B,Q,C] tensors.')
    return torch.stack([sum((a[:, j] * b[:, k] for j, k in plan.pairs(alpha)),
                            torch.zeros_like(a[:, 0]))
                        for alpha in plan.indices], dim=1)


def seed_coordinates(x: Tensor, plan: JetPlan) -> Tensor:
    if x.ndim != 2 or x.shape[1] != plan.dim:
        raise ValueError('Expected x of shape [B,dim].')
    out = x.new_zeros(x.shape[0], plan.q, plan.dim)
    out[:, 0] = x
    for axis in range(plan.dim):
        a = tuple(int(i == axis) for i in range(plan.dim))
        if a in plan.lookup:
            out[:, plan.lookup[a], axis] = 1.0
    return out


def tanh_jet(z: Tensor, plan: JetPlan) -> Tensor:
    """Degree <=3 Taylor composition; generic, intentionally not optimized."""
    t = torch.tanh(z[:, 0])
    s = 1.0 - t * t
    delta = z.clone()
    delta[:, 0] = 0.0
    a1 = s
    a2 = -t * s
    a3 = s * (t * t - 1.0 / 3.0)
    h = a1[:, None] * delta
    if plan.order >= 2:
        delta2 = multiply(delta, delta, plan)
        h = h + a2[:, None] * delta2
    if plan.order >= 3:
        h = h + a3[:, None] * multiply(delta2, delta, plan)
    h[:, 0] = t
    return h


def tanh_jet_vjp(h: Tensor, bar_h: Tensor, plan: JetPlan) -> Tensor:
    """Exact real-arithmetic jet VJP from G=1-H*H; ordinary FP64 evaluation.

    Saturated tanh tails need a separately designed/stably evaluated G[0].
    This implementation deliberately follows 1-tanh(z)^2 arithmetic for
    comparison with the ordinary PyTorch tanh backward, not arbitrary precision.
    """
    g = -multiply(h, h, plan)
    g[:, 0] += 1.0
    lookup = plan.lookup
    bar_z = []
    for beta in plan.indices:
        terms = []
        for ia, alpha in enumerate(plan.indices):
            if all(b <= a for a, b in zip(alpha, beta)):
                diff = tuple(a-b for a, b in zip(alpha, beta))
                terms.append(bar_h[:, ia] * g[:, lookup[diff]])
        bar_z.append(sum(terms, torch.zeros_like(h[:, 0])))
    return torch.stack(bar_z, dim=1)


def forward_jets(x: Tensor, weights: list[Tensor], biases: list[Tensor],
                 plan: JetPlan) -> tuple[Tensor, list[Tensor]]:
    h = seed_coordinates(x, plan)
    checkpoints = [h]
    for ell, (w, b) in enumerate(zip(weights, biases)):
        z = h @ w.T
        z[:, 0] += b  # bias applies ONLY to the order-zero coefficient
        h = tanh_jet(z, plan) if ell + 1 < len(weights) else z
        checkpoints.append(h)
    return h, checkpoints


def backward_parameters(bar_out: Tensor, checkpoints: list[Tensor],
                        weights: list[Tensor], plan: JetPlan
                        ) -> tuple[list[Tensor], list[Tensor]]:
    """Manual VJP: no autograd over the MLP jet propagation."""
    dw: list[Tensor] = [torch.empty(0)] * len(weights)
    db: list[Tensor] = [torch.empty(0)] * len(weights)
    bar_h = bar_out
    for ell in reversed(range(len(weights))):
        d = (tanh_jet_vjp(checkpoints[ell+1], bar_h, plan)
             if ell + 1 < len(weights) else bar_h)
        hp = checkpoints[ell]
        dw[ell] = d.flatten(0, 1).T @ hp.flatten(0, 1)
        db[ell] = d[:, 0].sum(dim=0)
        bar_h = d @ weights[ell]
    return dw, db


def residual_loss(jet: Tensor, plan: JetPlan, nu: float = 0.07,
                  gradient_weight: float = 0.2) -> Tensor:
    """2-D steady incompressible momentum/divergence + residual-gradient loss.

    u,v,p are three outputs. All weights, forcing and coordinates are fixed.
    This small residual implementation may use autograd to obtain output seeds;
    it does not build nested input-derivative graphs.
    """
    if plan.dim != 2 or plan.order != 3 or jet.shape[-1] != 3:
        raise ValueError('Expected a 2-D order-3 jet with three outputs u,v,p.')
    idx = plan.lookup
    def val(field: int, a: tuple[int, int]) -> Tensor:
        return jet[:, idx[a], field]
    def der(field: int, a: tuple[int, int], axis: int, n: int = 1) -> Tensor:
        target = list(a)
        target[axis] += n
        factor = factorial(target[axis]) / factorial(a[axis])
        return factor * val(field, tuple(target))
    terms = []
    for alpha in [(0, 0), (1, 0), (0, 1)]:
        adv_u = torch.zeros_like(jet[:, 0, 0])
        adv_v = torch.zeros_like(adv_u)
        for ib, ig in plan.pairs(alpha):
            beta, gamma = plan.indices[ib], plan.indices[ig]
            adv_u = adv_u + val(0,beta)*der(0,gamma,0) + val(1,beta)*der(0,gamma,1)
            adv_v = adv_v + val(0,beta)*der(1,gamma,0) + val(1,beta)*der(1,gamma,1)
        ru = adv_u + der(2,alpha,0) - nu*(der(0,alpha,0,2)+der(0,alpha,1,2))
        rv = adv_v + der(2,alpha,1) - nu*(der(1,alpha,0,2)+der(1,alpha,1,2))
        div = der(0,alpha,0) + der(1,alpha,1)
        weight = 1.0 if alpha == (0,0) else gradient_weight
        terms.append(weight*(ru.square()+rv.square()+div.square()))
    return 0.5 * torch.stack(terms).sum(dim=0).mean()


def mlp(x: Tensor, weights: list[Tensor], biases: list[Tensor]) -> Tensor:
    h = x
    for ell, (w,b) in enumerate(zip(weights,biases)):
        h = h @ w.T + b
        if ell + 1 < len(weights):
            h = h.tanh()
    return h


def nested_derivative(y: Tensor, x: Tensor, alpha: tuple[int, ...]) -> Tensor:
    ans = y
    for axis, times in enumerate(alpha):
        for _ in range(times):
            ans = torch.autograd.grad(ans.sum(), x, create_graph=True,
                                       retain_graph=True)[0][:, axis]
    return ans


def nested_loss(x: Tensor, weights: list[Tensor], biases: list[Tensor],
                nu: float = 0.07, gradient_weight: float = 0.2) -> Tensor:
    uvp = mlp(x, weights, biases)
    u,v,p = uvp.unbind(-1)
    def grad(f: Tensor) -> Tensor:
        return torch.autograd.grad(f.sum(), x, create_graph=True,
                                   retain_graph=True)[0]
    gu,gv,gp = grad(u),grad(v),grad(p)
    lapu = grad(gu[:,0])[:,0] + grad(gu[:,1])[:,1]
    lapv = grad(gv[:,0])[:,0] + grad(gv[:,1])[:,1]
    ru = u*gu[:,0]+v*gu[:,1]+gp[:,0]-nu*lapu
    rv = u*gv[:,0]+v*gv[:,1]+gp[:,1]-nu*lapv
    div = gu[:,0]+gv[:,1]
    value = ru.square()+rv.square()+div.square()
    smooth = grad(ru).square().sum(-1)+grad(rv).square().sum(-1)+grad(div).square().sum(-1)
    return 0.5*(value+gradient_weight*smooth).mean()


def max_abs(a: Tensor, b: Tensor) -> float:
    return float((a-b).detach().abs().max())


def verify() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(7309)
    results = {}
    for dim in (2,3):
        plan = JetPlan.dense(dim,3)
        z = torch.randn(3,plan.q,5,dtype=torch.float64)*0.2
        z.requires_grad_()
        bar = torch.randn_like(z)
        h = tanh_jet(z,plan)
        gold, = torch.autograd.grad((h*bar).sum(),z)
        manual = tanh_jet_vjp(h.detach(),bar,plan)
        err = max_abs(gold,manual)
        assert torch.allclose(manual,gold,rtol=2e-12,atol=2e-12), err
        results[f'tanh_vjp_{dim}d_q{plan.q}_max_abs'] = err
    plan = JetPlan.dense(2,3)
    widths = [2,8,8,3]
    weights = [(torch.randn(o,i,dtype=torch.float64)*0.2).requires_grad_()
               for i,o in zip(widths[:-1],widths[1:])]
    biases = [(torch.randn(o,dtype=torch.float64)*0.1).requires_grad_()
              for o in widths[1:]]
    x = (torch.randn(5,2,dtype=torch.float64)*0.4).requires_grad_()
    with torch.no_grad():
        jets, checkpoints = forward_jets(x,weights,biases,plan)
    out = mlp(x,weights,biases)
    reference = torch.stack([
        torch.stack([nested_derivative(out[:,c],x,a) /
                     (factorial(a[0])*factorial(a[1])) for c in range(3)],dim=-1)
        for a in plan.indices],dim=1)
    results['all_output_jets_max_abs'] = max_abs(jets,reference)
    torch.testing.assert_close(jets,reference,rtol=2e-11,atol=2e-12)
    # Only the compact residual is differentiated by autograd for its seed.
    jleaf = jets.detach().requires_grad_()
    loss_fast = residual_loss(jleaf,plan)
    bar_out, = torch.autograd.grad(loss_fast,jleaf)
    with torch.no_grad():
        dws,dbs = backward_parameters(bar_out,checkpoints,weights,plan)
    loss_ref = nested_loss(x,weights,biases)
    ref_grads = torch.autograd.grad(loss_ref,weights+biases)
    results['loss_jet'] = float(loss_fast.detach())
    results['loss_nested'] = float(loss_ref.detach())
    results['loss_abs_error'] = float((loss_fast-loss_ref).abs().detach())
    results['parameter_gradient_max_abs'] = max(max_abs(a,b) for a,b in zip(dws+dbs,ref_grads))
    for a,b in zip(dws+dbs,ref_grads):
        torch.testing.assert_close(a,b,rtol=2e-10,atol=2e-12)
    results['validation'] = 'PASS: CPU float64; no CUDA compilation or GPU benchmark.'
    results['scope'] = 'Random small nonsaturated networks; not a proof of robustness or full scientific convergence.'
    results['torch_version'] = torch.__version__
    return results

if __name__ == '__main__':
    result = verify()
    print(json.dumps(result,indent=2))
    Path(__file__).with_name('cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n')
