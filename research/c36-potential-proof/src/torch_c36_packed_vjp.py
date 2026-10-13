"""Experimental analytic transpose of the packed C36 tanh activation."""
import torch

import torch_c17 as c17
from torch_c36_packed import _indices


def activate_packed_vjp(state, cotangent):
    if state.shape[-1] != 36 or state.shape != cotangent.shape:
        raise ValueError("Matching 36-channel state and cotangent required")
    ids, prefix = _indices(state.device), state.shape[:-1]
    take = lambda tensor, name: tensor.index_select(-1, ids[name])

    def pull(value, name, size):
        return value.new_zeros(prefix + (size,)).index_add(-1, ids[name], value)

    g, h, t, q = state[..., 1:4], state[..., 4:10], state[..., 10:20], state[..., 20:26]
    m, n, k = state[..., 26], state[..., 27:30], state[..., 30:36]
    bg, bh, bt, bq = cotangent[..., 1:4], cotangent[..., 4:10], cotangent[..., 10:20], cotangent[..., 20:26]
    bm, bn, bk = cotangent[..., 26], cotangent[..., 27:30], cotangent[..., 30:36]
    y, a1, a2, a3, a4 = c17.coefficients(state[..., 0])
    a1, a2, a3, a4 = (a[..., None] for a in (a1, a2, a3, a4))
    lap = take(h, "diag").sum(-1, keepdim=True)
    gg = g.square().sum(-1, keepdim=True)
    full_h = take(h, "full_h").reshape(prefix + (3, 3))
    hg = (full_h * g[..., None, :]).sum(-1)
    lg = take(t, "lgrad").reshape(prefix + (3, 3)).sum(-1)
    full_t = take(t, "tg").reshape(prefix + (6, 3))
    tg = (full_t * g[..., None, :]).sum(-1)
    hl, hr = (take(h, name).reshape(prefix + (6, 3)) for name in ("hleft", "hright"))
    hh = (hl * hr).sum(-1)
    gi, gj = take(g, "i"), take(g, "j")
    gti, gtj, gtk = (take(g, name) for name in ("ti", "tj", "tk"))
    hjk, hik, hij = (take(h, name) for name in ("hjk", "hik", "hij"))
    ni, nj = take(n, "i"), take(n, "j")
    hgi, hgj, lgi, lgj = take(hg, "i"), take(hg, "j"), take(lg, "i"), take(lg, "j")
    outer = gi * gj
    third_bilinear = gti * hjk + gtj * hik + gtk * hij
    third_cubic = gti * gtj * gtk
    qa = h * lap + gi * lgj + lgi * gj + 2 * tg + 2 * hh
    qb = outer * lap + h * gg + 2 * gi * hgj + 2 * hgi * gj
    qc = outer * gg
    km = m[..., None]
    time_bilinear = km * h + ni * gj + gi * nj
    b1 = ((bg * g).sum(-1, keepdim=True) + (bh * h).sum(-1, keepdim=True)
          + (bt * t).sum(-1, keepdim=True) + (bq * q).sum(-1, keepdim=True)
          + (bm * m)[..., None] + (bn * n).sum(-1, keepdim=True) + (bk * k).sum(-1, keepdim=True))
    b2 = ((bh * outer).sum(-1, keepdim=True) + (bt * third_bilinear).sum(-1, keepdim=True)
          + (bq * qa).sum(-1, keepdim=True) + (bn * km * g).sum(-1, keepdim=True)
          + (bk * time_bilinear).sum(-1, keepdim=True))
    b3 = ((bt * third_cubic).sum(-1, keepdim=True) + (bq * qb).sum(-1, keepdim=True)
          + (bk * km * outer).sum(-1, keepdim=True))
    b4 = (bq * qc).sum(-1, keepdim=True)
    # Differentiate the actual stable coefficient program. tanh's rounded
    # derivative and the exp-based a1 differ in the saturated region.
    yy = y[..., None]
    y_bar = cotangent[..., :1] - 2 * a1 * b2 + 8 * yy * a1 * b3
    y_bar = y_bar + (16 * a1.square() - 24 * yy.square() * a1) * b4
    a1_bar = b1 - 2 * yy * b2 + (4 * yy.square() - 4 * a1) * b3
    a1_bar = a1_bar + (32 * yy * a1 - 8 * yy.pow(3)) * b4
    exponent = torch.exp(-2 * torch.abs(state[..., 0]))[..., None]
    exponent_bar = 4 / (1 + exponent).square() - 8 * exponent / (1 + exponent).pow(3)
    a1_derivative = exponent_bar * (-2 * exponent * torch.sign(state[..., :1]))
    value_bar = y_bar * (1 - yy.square()) + a1_bar * a1_derivative

    # Reverse the scalar coefficient groups, then their packed contractions.
    sb, cb = a2 * bt, a3 * bt
    ab, bb, qb4 = a2 * bq, a3 * bq, a4 * bq
    outer_bar = a2 * bh + a3 * km * bk + bb * lap + qb4 * gg
    gi_bar = ab * lgj + 2 * bb * hgj + a2 * bk * nj + outer_bar * gj
    gj_bar = ab * lgi + 2 * bb * hgi + a2 * bk * ni + outer_bar * gi
    hg_bar = pull(2 * bb * gj, "i", 3) + pull(2 * bb * gi, "j", 3)
    lg_bar = pull(ab * gj, "i", 3) + pull(ab * gi, "j", 3)
    tg_bar, hh_bar = 2 * ab, 2 * ab
    lap_bar = (ab * h + bb * outer).sum(-1, keepdim=True)
    gg_bar = (bb * h + qb4 * outer).sum(-1, keepdim=True)
    g_bar = (a1 * bg + a2 * km * bn + pull(gi_bar, "i", 3) + pull(gj_bar, "j", 3)
             + pull(sb * hjk + cb * gtj * gtk, "ti", 3)
             + pull(sb * hik + cb * gti * gtk, "tj", 3)
             + pull(sb * hij + cb * gti * gtj, "tk", 3)
             + (tg_bar[..., :, None] * full_t).sum(-2)
             + (hg_bar[..., :, None] * full_h).sum(-2) + 2 * gg_bar * g)
    h_bar = (a1 * bh + ab * lap + bb * gg + a2 * km * bk
             + pull(sb * gti, "hjk", 6) + pull(sb * gtj, "hik", 6) + pull(sb * gtk, "hij", 6)
             + pull((hh_bar[..., :, None] * hr).flatten(-2), "hleft", 6)
             + pull((hh_bar[..., :, None] * hl).flatten(-2), "hright", 6)
             + pull((hg_bar[..., :, None] * g[..., None, :]).flatten(-2), "full_h", 6)
             + pull(lap_bar.expand(prefix + (3,)), "diag", 6))
    t_bar = (a1 * bt + pull(lg_bar[..., :, None].expand(prefix + (3, 3)).flatten(-2), "lgrad", 10)
             + pull((tg_bar[..., :, None] * g[..., None, :]).flatten(-2), "tg", 10))
    m_bar = a1 * bm[..., None] + a2 * (bn * g).sum(-1, keepdim=True)
    m_bar = m_bar + a2 * (bk * h).sum(-1, keepdim=True) + a3 * (bk * outer).sum(-1, keepdim=True)
    n_bar = a1 * bn + pull(a2 * bk * gj, "i", 3) + pull(a2 * bk * gi, "j", 3)
    return torch.cat((value_bar, g_bar, h_bar, t_bar, a1 * bq, m_bar, n_bar, a1 * bk), -1)
