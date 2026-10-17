"""Naive ablation: all four-variable monomials through degree three.

No residual-aware pruning, contracted layer state, or explicit parameter VJP.
CUDA graph use is matched to the candidate and recorded separately.
"""
import math
import torch
from taylor_oracle import MULTI, INDEX, PRODUCT


def forward(parameters, points):
    state = points.new_zeros((len(points), 4, len(MULTI)))
    state[..., 0] = points
    for i in range(4):
        state[:, i, INDEX[tuple(int(j == i) for j in range(4))]] = 1

    def multiply(a, b):
        return torch.stack([
            sum((a[..., i] * b[..., j] for i, j, k in PRODUCT if k == out),
                torch.zeros_like(a[..., 0]))
            for out in range(len(MULTI))
        ], -1)

    for layer, (weight, bias) in enumerate(parameters):
        state = torch.einsum('nwq,ow->noq', state, weight)
        z = state[..., 0] + bias
        state = torch.cat((z[..., None], state[..., 1:]), -1)
        if layer < len(parameters) - 1:
            y = torch.tanh(z)
            a1 = 1 - y * y
            a2 = -2 * y * a1
            a3 = (6 * y * y - 2) * a1
            delta = torch.cat((torch.zeros_like(z[..., None]), state[..., 1:]), -1)
            square = multiply(delta, delta)
            cube = multiply(square, delta)
            state = a1[..., None] * delta + a2[..., None] * square / 2 + a3[..., None] * cube / 6
            state = torch.cat((y[..., None], state[..., 1:]), -1)
    return state


def contract(state):
    def derivative(axes):
        multi = tuple(axes.count(i) for i in range(4))
        return state[..., INDEX[multi]] * math.prod(math.factorial(i) for i in multi)
    hp = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    return torch.stack([state[..., 0]] + [derivative([i]) for i in range(3)] +
        [derivative([i, j]) for i, j in hp] +
        [sum(derivative([i, j, j]) for j in range(3)) for i in range(3)] +
        [derivative([3])] + [derivative([i, 3]) for i in range(3)], -1)
