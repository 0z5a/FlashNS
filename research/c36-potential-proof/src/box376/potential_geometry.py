"""Experimental geometry features and support gate; no exact velocity target.

Only the prescribed box motion and cutoff enter this representation. Learned
potential coefficients remain unknown. This changes model representation, not
forcing, viscosity, sampling, normalization or physical acceptance thresholds.
"""
import torch

from box376.objective import values
from box376.source import C0, DISPLACEMENT, H0, LOGS


def smooth_switch(q):
    """Stable switch value and first derivative, with finite flat branches."""
    with torch.no_grad():
        interior = (q > 0) & (q < 1)
        probe = torch.where(interior, q, torch.full_like(q, .5))
        logit = 1 / (1 - probe) - 1 / probe
        active = interior & (torch.abs(logit) <= 700)
    safe = torch.where(active, q, torch.full_like(q, .5))
    logit = 1 / (1 - safe) - 1 / safe
    positive = logit >= 0
    ep = torch.exp(-torch.where(positive, logit, torch.zeros_like(logit)))
    en = torch.exp(torch.where(positive, torch.zeros_like(logit), logit))
    sigmoid = torch.where(positive, 1 / (1 + ep), en / (1 + en))
    e = torch.where(positive, ep, en)
    first = e / (1 + e)**2 * (1 / safe**2 + 1 / (1 - safe)**2)
    value = torch.where(active, sigmoid, (q >= .5).to(q.dtype))
    return value, torch.where(active, first, torch.zeros_like(first))


def features_and_gate(points, eta=1/32):
    if eta <= 0 or points.shape[-1] != 4:
        raise ValueError("Positive cutoff width and physical Nx4 coordinates required")
    s, ds = smooth_switch(2 * points[:, 3] - .5)
    ds = 2 * ds
    logs = points.new_tensor(LOGS)
    first = torch.exp(s * logs[0])
    second = torch.exp(s * logs[1])
    dilation = torch.stack((first, second, 1 / (first * second)), 1)
    half_width = points.new_tensor(H0) * dilation
    center = points.new_tensor(C0) + s[:, None] * points.new_tensor(DISPLACEMENT)
    relative = points[:, :3] - center
    features = torch.cat((relative / half_width, points[:, 3:4]), 1)
    axes = []
    for i in range(3):
        minus, _ = smooth_switch((half_width[:, i] + 2 * eta - relative[:, i]) / eta)
        plus, _ = smooth_switch((half_width[:, i] + 2 * eta + relative[:, i]) / eta)
        axes.append(minus * plus)
    gate = (ds / 4) * torch.stack(axes, 1).prod(1)
    return features, gate[:, None]


def potentials(parameters, points, eta=1/32, potential_scale=.01, pressure_scale=.1):
    features, gate = features_and_gate(points, eta)
    scale = points.new_tensor([potential_scale] * 3 + [pressure_scale])
    return gate * values(parameters, features) * scale


def native_field(parameters, points, eta=1/32, potential_scale=.01, pressure_scale=.1):
    """Independent coordinate-AD curl for values and later physical checks."""
    potential = potentials(parameters, points, eta, potential_scale, pressure_scale)
    gradients = [torch.autograd.grad(potential[:, i].sum(), points, create_graph=True)[0]
                 for i in range(3)]
    velocity = torch.stack((gradients[2][:, 1] - gradients[1][:, 2],
                            gradients[0][:, 2] - gradients[2][:, 0],
                            gradients[1][:, 0] - gradients[0][:, 1]), 1)
    return torch.cat((velocity, potential[:, 3:4]), 1)
