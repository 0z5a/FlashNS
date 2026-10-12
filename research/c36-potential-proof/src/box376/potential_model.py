"""Experimental curl-potential model using partial 36-channel derivative states."""
import math

import torch
import torch_c36 as c36

from box376.c36_coordinate_oracle import native_c36
from box376.objective import ORIGIN, SCALE, physical_parameters
from box376.potential_geometry import features_and_gate, smooth_switch
from box376.source import C0, DISPLACEMENT, H0, LOGS


def prepare_geometry(points, eta=1/32, *, chunk_size=256):
    """Prepare detached C36 geometry with bounded coordinate-AD batches.

    Features are affine in space, so only their value, spatial gradient, dt,
    and spatial gradient of dt can be nonzero. The support gate still uses
    native coordinate AD. ``chunk_size`` bounds that graph, not output storage;
    include preprocessing time and the returned arrays in model setup costs.
    """
    if points.ndim != 2 or points.shape[1] != 4 or not points.is_floating_point():
        raise ValueError("Floating physical Nx4 coordinates required")
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("Positive finite cutoff width required")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("Positive integer chunk_size required")
    with torch.no_grad():
        inputs = points.new_zeros((len(points), 4, 36))
        gate = points.new_zeros((len(points), 1, 36))
        for begin in range(0, len(points), chunk_size):
            end = min(begin + chunk_size, len(points))
            local = points[begin:end].detach()
            switch, ds = smooth_switch(2 * local[:, 3] - .5)
            ds = 2 * ds
            first = torch.exp(switch * LOGS[0])
            second = torch.exp(switch * LOGS[1])
            width = local.new_tensor(H0) * torch.stack((first, second, 1 / (first * second)), 1)
            center = local.new_tensor(C0) + switch[:, None] * local.new_tensor(DISPLACEMENT)
            relative = (local[:, :3] - center) / width
            log_rates = local.new_tensor([LOGS[0], LOGS[1], -LOGS[0] - LOGS[1]])
            state = inputs[begin:end]
            state[:, :3, 0] = relative
            state[:, 3, 0] = local[:, 3]
            state[:, :3, 26] = -ds[:, None] * (local.new_tensor(DISPLACEMENT) / width + relative * log_rates)
            state[:, 3, 26] = 1
            for axis in range(3):
                state[:, axis, 1 + axis] = 1 / width[:, axis]
                state[:, axis, 27 + axis] = -ds * log_rates[axis] / width[:, axis]
            # The smooth switch is exactly flat outside this time interval.
            active = (local[:, 3] > .25) & (local[:, 3] < .75)
            if active.any():
                with torch.enable_grad():
                    x = local[active].clone().requires_grad_(True)
                    gate[begin:end][active] = native_c36(lambda p: features_and_gate(p, eta)[1], x).detach()
    return inputs, gate


def forward(parameters, inputs, gate, potential_scale=.01, pressure_scale=.1, *, constants=None):
    """Use optional preconstructed device constants for the default model scales."""
    if constants is not None and (potential_scale != .01 or pressure_scale != .1):
        raise ValueError("Preconstructed constants require the default potential and pressure scales")
    if constants is None:
        physical = physical_parameters(parameters)
    else:
        alpha, origin, scale = constants
        weight, bias = parameters[0]
        physical = [(weight * alpha, bias - (weight * alpha) @ origin)] + parameters[1:]
    with torch.no_grad():
        raw, tape = c36.forward(physical, inputs)
        if constants is None:
            scale = raw.new_tensor([potential_scale] * 3 + [pressure_scale])[None, :, None]
        state = c36.curl_pressure(c36.multiply(gate, raw * scale))
    return state, (physical, raw, tape, gate, scale)


def parameter_vjp(tape, cotangent, *, constants=None):
    physical, raw, network_tape, gate, scale = tape
    bar = c36.state_vjp(lambda s: c36.curl_pressure(c36.multiply(gate, s * scale)), raw, cotangent)
    with torch.no_grad():
        derivatives = c36.parameter_vjp(physical, network_tape, bar)
        weight, bias = derivatives[0]
        if constants is None:
            alpha, origin = weight.new_tensor(SCALE), weight.new_tensor(ORIGIN)
        else:
            alpha, origin, _ = constants
        derivatives[0] = (weight * alpha - bias[:, None] * (alpha * origin)[None, :], bias)
    return derivatives
