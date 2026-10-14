"""Experimental C36 tanh activation computed in symmetric storage.

This CPU-qualified candidate is opt-in. The default forward and GPU benchmark
sources keep using torch_c36.activate until full GPU qualification and paired
timing are available.
"""
import torch

import torch_c17 as c17
from torch_c36 import PAIRS, TRIPLES

_INDICES = {}


def _indices(device):
    if device not in _INDICES:
        pair = lambda i, j: PAIRS.index(tuple(sorted((i, j))))
        triple = lambda i, j, k: TRIPLES.index(tuple(sorted((i, j, k))))
        rows = {
            "i": [i for i, j in PAIRS], "j": [j for i, j in PAIRS],
            "ti": [i for i, j, k in TRIPLES],
            "tj": [j for i, j, k in TRIPLES],
            "tk": [k for i, j, k in TRIPLES],
            "hjk": [pair(j, k) for i, j, k in TRIPLES],
            "hik": [pair(i, k) for i, j, k in TRIPLES],
            "hij": [pair(i, j) for i, j, k in TRIPLES],
            "diag": [pair(i, i) for i in range(3)],
            "full_h": [pair(i, j) for i in range(3) for j in range(3)],
            "lgrad": [triple(i, j, j) for i in range(3) for j in range(3)],
            "tg": [triple(i, j, k) for i, j in PAIRS for k in range(3)],
            "hleft": [pair(i, k) for i, j in PAIRS for k in range(3)],
            "hright": [pair(j, k) for i, j in PAIRS for k in range(3)],
        }
        _INDICES[device] = {k: torch.tensor(v, dtype=torch.long, device=device)
                            for k, v in rows.items()}
    return _INDICES[device]


def activate_packed(state):
    """Same 36 derivative identities; compute ten third and six Hessian entries."""
    if state.shape[-1] != 36:
        raise ValueError("A 36-channel derivative state is required")
    ids = _indices(state.device)
    take = lambda tensor, name: tensor.index_select(-1, ids[name])
    value, g, h, t, q = state[..., 0], state[..., 1:4], state[..., 4:10], state[..., 10:20], state[..., 20:26]
    m, n, k = state[..., 26], state[..., 27:30], state[..., 30:36]
    y, a1, a2, a3, a4 = c17.coefficients(value)
    a1, a2, a3, a4 = (a[..., None] for a in (a1, a2, a3, a4))
    lap = take(h, "diag").sum(-1, keepdim=True)
    gg = g.square().sum(-1, keepdim=True)
    hg = (take(h, "full_h").reshape(h.shape[:-1] + (3, 3)) * g[..., None, :]).sum(-1)
    lgrad = take(t, "lgrad").reshape(t.shape[:-1] + (3, 3)).sum(-1)
    tg = (take(t, "tg").reshape(t.shape[:-1] + (6, 3)) * g[..., None, :]).sum(-1)
    hh = (take(h, "hleft").reshape(h.shape[:-1] + (6, 3))
          * take(h, "hright").reshape(h.shape[:-1] + (6, 3))).sum(-1)
    gi, gj = take(g, "i"), take(g, "j")
    gti, gtj, gtk = take(g, "ti"), take(g, "tj"), take(g, "tk")
    outer = gi * gj
    third = (a1 * t + a2 * (gti * take(h, "hjk") + gtj * take(h, "hik") + gtk * take(h, "hij"))
             + a3 * (gti * gtj * gtk))
    fourth = (a1 * q + a2 * (h * lap + gi * take(lgrad, "j") + take(lgrad, "i") * gj + 2 * tg + 2 * hh)
              + a3 * (outer * lap + h * gg + 2 * gi * take(hg, "j") + 2 * take(hg, "i") * gj)
              + a4 * outer * gg)
    time_hessian = (a1 * k + a2 * (m[..., None] * h + take(n, "i") * gj + gi * take(n, "j"))
                    + a3 * m[..., None] * outer)
    return torch.cat((y[..., None], a1 * g, a1 * h + a2 * outer, third, fourth,
                      a1 * m[..., None], a1 * n + a2 * m[..., None] * g, time_hessian), -1)
