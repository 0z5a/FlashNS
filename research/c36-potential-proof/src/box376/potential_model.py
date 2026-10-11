"""Experimental curl-potential model using partial 36-channel derivative states."""
import torch
import torch_c36 as c36

from box376.c36_coordinate_oracle import native_c36
from box376.objective import ORIGIN, SCALE, physical_parameters
from box376.potential_geometry import features_and_gate


def prepare_geometry(points, eta=1/32):
    """CPU/reference preprocessing; include its actual cost in model setup."""
    with torch.enable_grad():
        x = points.detach().clone().requires_grad_(True)
        inputs = native_c36(lambda p: features_and_gate(p, eta)[0], x).detach()
        x = points.detach().clone().requires_grad_(True)
        gate = native_c36(lambda p: features_and_gate(p, eta)[1], x).detach()
    return inputs, gate


def forward(parameters, inputs, gate, potential_scale=.01, pressure_scale=.1):
    physical = physical_parameters(parameters)
    with torch.no_grad():
        raw, tape = c36.forward(physical, inputs)
        scale = raw.new_tensor([potential_scale] * 3 + [pressure_scale])[None, :, None]
        state = c36.curl_pressure(c36.multiply(gate, raw * scale))
    return state, (physical, raw, tape, gate, scale)


def parameter_vjp(tape, cotangent):
    physical, raw, network_tape, gate, scale = tape
    bar = c36.state_vjp(lambda s: c36.curl_pressure(c36.multiply(gate, s * scale)), raw, cotangent)
    with torch.no_grad():
        derivatives = c36.parameter_vjp(physical, network_tape, bar)
        weight, bias = derivatives[0]
        alpha, origin = weight.new_tensor(SCALE), weight.new_tensor(ORIGIN)
        derivatives[0] = (weight * alpha - bias[:, None] * (alpha * origin)[None, :], bias)
    return derivatives
