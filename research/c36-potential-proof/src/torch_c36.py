"""Experimental partial jets for a curl-potential transient 3D model.

State: value, spatial gradient, symmetric Hessian, symmetric third derivative,
Hessian of spatial Laplacian, dt, spatial gradient of dt, spatial Hessian of dt.
The 36-state forward uses explicit derivative identities. Nonlinear pullbacks
currently use local state autograd, not coordinate AD. GPU capture, performance,
complete model gradients and solver convergence still require qualification.
"""
import itertools

import torch
import torch_c17 as c17

PAIRS = c17.PAIRS
TRIPLES = tuple(itertools.combinations_with_replacement(range(3), 3))
_UNPACK_INDICES = {}


def unpack(state):
    if state.shape[-1] != 36:
        raise ValueError("A 36-channel derivative state is required")
    # Gather the symmetric channels directly. In-place scalar slices create a
    # long CopySlices backward chain; these four gathers have the same transpose.
    key = state.device
    if key not in _UNPACK_INDICES:
        pairs = [PAIRS.index((min(i, j), max(i, j))) for i in range(3) for j in range(3)]
        triples = [TRIPLES.index(tuple(sorted((i, j, k))))
                   for i in range(3) for j in range(3) for k in range(3)]
        _UNPACK_INDICES[key] = tuple(torch.tensor(indices, dtype=torch.long, device=state.device)
                                    for indices in ([4 + v for v in pairs], [10 + v for v in triples],
                                                    [20 + v for v in pairs], [30 + v for v in pairs]))
    hi, ti, qi, ki = _UNPACK_INDICES[key]
    prefix = state.shape[:-1]
    hessian = state.index_select(-1, hi).reshape(prefix + (3, 3))
    third = state.index_select(-1, ti).reshape(prefix + (3, 3, 3))
    fourth = state.index_select(-1, qi).reshape(prefix + (3, 3))
    time_hessian = state.index_select(-1, ki).reshape(prefix + (3, 3))
    return (state[..., 0], state[..., 1:4], hessian, third, fourth,
            state[..., 26], state[..., 27:30], time_hessian)


def pack(value, gradient, hessian, third, fourth, time, time_gradient, time_hessian):
    return torch.cat((value[..., None], gradient,
                      torch.stack([hessian[..., i, j] for i, j in PAIRS], -1),
                      torch.stack([third[..., i, j, k] for i, j, k in TRIPLES], -1),
                      torch.stack([fourth[..., i, j] for i, j in PAIRS], -1),
                      time[..., None], time_gradient,
                      torch.stack([time_hessian[..., i, j] for i, j in PAIRS], -1)), -1)


def laplacian_gradient(third):
    return torch.stack([sum(third[..., i, j, j] for j in range(3)) for i in range(3)], -1)


def activate(state):
    value, g, h, t, q, m, n, k = unpack(state)
    y, a1, a2, a3, a4 = c17.coefficients(value)
    lap = h.diagonal(dim1=-2, dim2=-1).sum(-1)
    lgrad = laplacian_gradient(t)
    gg = (g * g).sum(-1)
    hg = torch.einsum("...ij,...j->...i", h, g)
    ht_h = torch.einsum("...ik,...jk->...ij", h, h)
    tg = torch.einsum("...ijk,...k->...ij", t, g)
    outer = g[..., :, None] * g[..., None, :]
    third_out = (a1[..., None, None, None] * t
                 + a2[..., None, None, None] * (
                     g[..., :, None, None] * h[..., None, :, :]
                     + g[..., None, :, None] * h[..., :, None, :]
                     + g[..., None, None, :] * h[..., :, :, None])
                 + a3[..., None, None, None] * (
                     g[..., :, None, None] * g[..., None, :, None] * g[..., None, None, :]))
    fourth_out = (a1[..., None, None] * q
                  + a2[..., None, None] * (
                      h * lap[..., None, None]
                      + g[..., :, None] * lgrad[..., None, :]
                      + lgrad[..., :, None] * g[..., None, :]
                      + 2 * tg + 2 * ht_h)
                  + a3[..., None, None] * (
                      outer * lap[..., None, None] + h * gg[..., None, None]
                      + 2 * g[..., :, None] * hg[..., None, :]
                      + 2 * hg[..., :, None] * g[..., None, :])
                  + a4[..., None, None] * outer * gg[..., None, None])
    return pack(y, a1[..., None] * g,
                a1[..., None, None] * h + a2[..., None, None] * outer,
                third_out, fourth_out, a1 * m,
                a1[..., None] * n + a2[..., None] * m[..., None] * g,
                a1[..., None, None] * k
                + a2[..., None, None] * (
                    m[..., None, None] * h + n[..., :, None] * g[..., None, :]
                    + g[..., :, None] * n[..., None, :])
                + a3[..., None, None] * m[..., None, None] * outer)


def multiply(left, right):
    """Leibniz product closed on the declared 36 channels."""
    a, ag, ah, at, aq, am, an, ak = unpack(left)
    b, bg, bh, bt, bq, bm, bn, bk = unpack(right)
    al = ah.diagonal(dim1=-2, dim2=-1).sum(-1)
    bl = bh.diagonal(dim1=-2, dim2=-1).sum(-1)
    alg, blg = laplacian_gradient(at), laplacian_gradient(bt)
    hessian = (a[..., None, None] * bh + b[..., None, None] * ah
               + ag[..., :, None] * bg[..., None, :] + bg[..., :, None] * ag[..., None, :])
    third = (a[..., None, None, None] * bt + b[..., None, None, None] * at
             + ag[..., :, None, None] * bh[..., None, :, :]
             + ag[..., None, :, None] * bh[..., :, None, :]
             + ag[..., None, None, :] * bh[..., :, :, None]
             + bg[..., :, None, None] * ah[..., None, :, :]
             + bg[..., None, :, None] * ah[..., :, None, :]
             + bg[..., None, None, :] * ah[..., :, :, None])
    fourth = (a[..., None, None] * bq + b[..., None, None] * aq
              + ag[..., :, None] * blg[..., None, :] + blg[..., :, None] * ag[..., None, :]
              + bg[..., :, None] * alg[..., None, :] + alg[..., :, None] * bg[..., None, :]
              + ah * bl[..., None, None] + bh * al[..., None, None]
              + 2 * torch.einsum("...ijk,...k->...ij", at, bg)
              + 2 * torch.einsum("...ijk,...k->...ij", bt, ag)
              + 2 * torch.einsum("...ik,...jk->...ij", ah, bh)
              + 2 * torch.einsum("...jk,...ik->...ij", ah, bh))
    time_hessian = (a[..., None, None] * bk + b[..., None, None] * ak
                    + am[..., None, None] * bh + bm[..., None, None] * ah
                    + ag[..., :, None] * bn[..., None, :] + bn[..., :, None] * ag[..., None, :]
                    + an[..., :, None] * bg[..., None, :] + bg[..., :, None] * an[..., None, :])
    return pack(a * b, a[..., None] * bg + b[..., None] * ag, hessian, third, fourth,
                a * bm + b * am,
                a[..., None] * bn + b[..., None] * an + am[..., None] * bg + bm[..., None] * ag,
                time_hessian)


def state_vjp(function, state, cotangent):
    """Local derivative-state transpose; no coordinate derivative graph."""
    with torch.enable_grad():
        leaf = state.detach().requires_grad_(True)
        result = function(leaf)
        return torch.autograd.grad(result, leaf, grad_outputs=cotangent.detach())[0].detach()


def coordinate_state(points):
    state = points.new_zeros((len(points), 4, 36))
    state[:, :, 0] = points
    for i in range(3):
        state[:, i, 1 + i] = 1
    state[:, 3, 26] = 1
    return state


def forward(parameters, input_state):
    state, tape = input_state, []
    for index, (weight, bias) in enumerate(parameters):
        z = torch.einsum("nwq,ow->noq", state, weight)
        z[:, :, 0] += bias
        tape.append((state, z))
        state = activate(z) if index < len(parameters) - 1 else z
    return state, tape


def parameter_vjp(parameters, tape, cotangent):
    gradients, bar = [], cotangent
    for index in range(len(parameters) - 1, -1, -1):
        state, z = tape[index]
        weight, _ = parameters[index]
        if index < len(parameters) - 1:
            bar = state_vjp(activate, z, bar)
        gradients.append((torch.einsum("noq,nwq->ow", bar, state), bar[:, :, 0].sum(0)))
        bar = torch.einsum("noq,ow->nwq", bar, weight)
    return list(reversed(gradients))


def curl_pressure(state):
    """Decode potential components0..2 and pressure3 into physical C17 jets."""
    value, g, h, t, q, m, n, k = unpack(state)
    values, gradients, hessians, lgrads, times, time_gradients = [], [], [], [], [], []
    for _, j, a in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        values.append(g[:, a, j] - g[:, j, a])
        gradients.append(h[:, a, j, :] - h[:, j, a, :])
        hessians.append(t[:, a, j, :, :] - t[:, j, a, :, :])
        lgrads.append(q[:, a, j, :] - q[:, j, a, :])
        times.append(n[:, a, j] - n[:, j, a])
        time_gradients.append(k[:, a, j, :] - k[:, j, a, :])
    values.append(value[:, 3]); gradients.append(g[:, 3]); hessians.append(h[:, 3])
    lgrads.append(laplacian_gradient(t[:, 3])); times.append(m[:, 3]); time_gradients.append(n[:, 3])
    return c17.pack(torch.stack(values, 1), torch.stack(gradients, 1), torch.stack(hessians, 1),
                    torch.stack(lgrads, 1), torch.stack(times, 1), torch.stack(time_gradients, 1))
