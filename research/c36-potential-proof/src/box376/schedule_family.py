"""Temporal-smoothness family of the C36 box problem for continuation schedules.

The original box motion uses s(t) = switch(2t - 1/2), so the box moves for t in (1/4, 3/4).
A transition width w generalizes it to

    s_w(t) = switch(t / w - (1 / (2w) - 1/2)),    moving for t in (1/2 - w/2, 1/2 + w/2),

centred at t = 1/2 with the same total displacement and dilation. A larger w is a slower,
smoother motion: the k-th time derivative scales as w**-k. The width w = 1/2 reproduces the
original problem bit for bit.

Only new code lives here. The pinned source, geometry and model modules are unchanged:
``source_w`` swaps the source module's ``motion`` while it evaluates, and
``prepared_objective`` swaps the objective module's ``prepare_geometry`` while it builds a
problem at another width. Physical acceptance always uses the original schedule.
"""
import contextlib
import copy
import hashlib
import math

import numpy as np
import torch

import box376.potential_objective as objective_module
import box376.source as source_module
from box376.c36_coordinate_oracle import native_c36
from box376.potential_geometry import smooth_switch
from box376.source import C0, DISPLACEMENT, H0, LOGS, switch

ORIGINAL_WIDTH = 0.5


def phase(t, width):
    """Switch argument q(t) = t / w - (1/(2w) - 1/2); equals 2t - 1/2 exactly at w = 1/2."""
    return t * (1 / width) - (.5 / width - .5)


def window(width):
    return .5 - .5 * width, .5 + .5 * width


def motion(t, width):
    """box376.source.motion with the width-w schedule (verbatim otherwise)."""
    derivatives = switch(phase(np.asarray(t), width))
    s = derivatives[..., 0]
    sp = (1 / width) * derivatives[..., 1]
    spp = (1 / width) ** 2 * derivatives[..., 2]
    D = np.empty(s.shape + (3,))
    D[..., 0] = np.exp(s * LOGS[0])
    D[..., 1] = np.exp(s * LOGS[1])
    D[..., 2] = 1 / (D[..., 0] * D[..., 1])
    ell = sp[..., None] * LOGS
    ell[..., 2] = -ell[..., 0] - ell[..., 1]
    ell_t = spp[..., None] * LOGS
    ell_t[..., 2] = -ell_t[..., 0] - ell_t[..., 1]
    return {'s': s, 'sp': sp, 'spp': spp, 'D': D, 'h': H0 * D, 'c': C0 + s[..., None] * DISPLACEMENT,
            'v': sp[..., None] * DISPLACEMENT, 'acceleration': spp[..., None] * DISPLACEMENT,
            'ell': ell, 'ell_t': ell_t}


@contextlib.contextmanager
def _swapped(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def source_w(points, width, eta=1/32, nu=.01):
    """Exact field, forcing and forcing gradient of the width-w problem."""
    if width == ORIGINAL_WIDTH:
        return source_module.source(points, eta=eta, nu=nu)
    with _swapped(source_module, 'motion', lambda t: motion(t, width)):
        return source_module.source(points, eta=eta, nu=nu)


def features_and_gate_w(points, width, eta=1/32):
    """box376.potential_geometry.features_and_gate with the width-w schedule."""
    if eta <= 0 or points.shape[-1] != 4:
        raise ValueError("Positive cutoff width and physical Nx4 coordinates required")
    s, ds = smooth_switch(phase(points[:, 3], width))
    ds = (1 / width) * ds
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
    gate = (ds / 4) * axes[0] * axes[1] * axes[2]
    return features, gate[:, None]


def prepare_geometry_w(points, eta=1/32, *, width, chunk_size=256):
    """box376.potential_model.prepare_geometry with the width-w schedule."""
    if points.ndim != 2 or points.shape[1] != 4 or not points.is_floating_point():
        raise ValueError("Floating physical Nx4 coordinates required")
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("Positive finite cutoff width required")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("Positive integer chunk_size required")
    low, high = window(width)
    with torch.no_grad():
        inputs = points.new_zeros((len(points), 4, 36))
        gate = points.new_zeros((len(points), 1, 36))
        for begin in range(0, len(points), chunk_size):
            end = min(begin + chunk_size, len(points))
            local = points[begin:end].detach()
            switch_value, ds = smooth_switch(phase(local[:, 3], width))
            ds = (1 / width) * ds
            first = torch.exp(switch_value * LOGS[0])
            second = torch.exp(switch_value * LOGS[1])
            half_width = local.new_tensor(H0) * torch.stack((first, second, 1 / (first * second)), 1)
            center = local.new_tensor(C0) + switch_value[:, None] * local.new_tensor(DISPLACEMENT)
            relative = (local[:, :3] - center) / half_width
            log_rates = local.new_tensor([LOGS[0], LOGS[1], -LOGS[0] - LOGS[1]])
            state = inputs[begin:end]
            state[:, :3, 0] = relative
            state[:, 3, 0] = local[:, 3]
            state[:, :3, 26] = -ds[:, None] * (local.new_tensor(DISPLACEMENT) / half_width + relative * log_rates)
            state[:, 3, 26] = 1
            for axis in range(3):
                state[:, axis, 1 + axis] = 1 / half_width[:, axis]
                state[:, axis, 27 + axis] = -ds * log_rates[axis] / half_width[:, axis]
            active = (local[:, 3] > low) & (local[:, 3] < high)
            if active.any():
                with torch.enable_grad():
                    x = local[active].clone().requires_grad_(True)
                    gate[begin:end][active] = native_c36(lambda p: features_and_gate_w(p, width, eta)[1], x).detach()
    return inputs, gate


def data_at_width(data, width):
    """Training data of the width-w problem: same points, weights and calibration;
    interior forcing and forcing gradient from the width-w exact field. Value targets
    stay zero; this is checked."""
    if width == ORIGINAL_WIDTH:
        return data, None
    result = {key: value for key, value in data.items() if key != 'families'}
    result['families'] = {}
    record = {'width': width}
    for name, family in data['families'].items():
        family = dict(family)
        points = family['points'].detach().cpu().numpy()
        exact = source_w(points, width, eta=data['eta'], nu=data['nu'])
        if name == 'interior':
            force = torch.as_tensor(exact['f'], dtype=family['force'].dtype, device=family['force'].device)
            gradient = torch.as_tensor(exact['df'], dtype=family['force_gradient'].dtype, device=family['force_gradient'].device)
            family['force'], family['force_gradient'] = force, gradient
            record['interior_force_sha256'] = hashlib.sha256(exact['f'].tobytes()).hexdigest()
            record['interior_force_gradient_sha256'] = hashlib.sha256(exact['df'].tobytes()).hexdigest()
        else:
            assert np.all(exact['U'] == 0), f'{name} velocity target is no longer zero at width {width}'
            assert torch.count_nonzero(family['target']).item() == 0
        result['families'][name] = family
    return result, record


def prepared_objective(cls, data, calibration, width, **kwargs):
    """Build a prepared objective (analytic or parameter AD) for the width-w problem."""
    data_w, record = data_at_width(data, width)
    if width == ORIGINAL_WIDTH:
        return cls(data_w, calibration, **kwargs), record

    def geometry(points, eta, chunk_size=256):
        return prepare_geometry_w(points, eta, width=width, chunk_size=chunk_size)

    with _swapped(objective_module, 'prepare_geometry', geometry):
        return cls(data_w, calibration, **kwargs), record
