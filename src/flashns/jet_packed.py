"""Static full-cubic/value-only row layout with explicit FP64 activation VJP."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from flashns.jet_spec import JetSpec
from flashns.jet_stable import tanh_jet


@dataclass(frozen=True)
class PackedLayout:
    full_points: int
    value_points: int
    dimension: int = 2

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.full_points, self.value_points)):
            raise ValueError("nonnegative integer point counts required")
        JetSpec(self.dimension)

    @property
    def q(self):
        return JetSpec(self.dimension).q

    @property
    def points(self):
        return self.full_points + self.value_points

    @property
    def full_rows(self):
        return self.full_points * self.q

    @property
    def rows(self):
        return self.full_rows + self.value_points

    def zero_rows(self, device):
        return torch.cat((torch.arange(0, self.full_rows, self.q, device=device),
                          torch.arange(self.full_rows, self.rows, device=device))).long()

    def coordinates(self, x):
        if x.shape != (self.points, self.dimension) or x.dtype != torch.float64:
            raise ValueError("FP64 coordinates must match the layout")
        result = x.new_zeros((self.rows, self.dimension))
        result[self.zero_rows(x.device)] = x
        lookup = {a: k for k, a in enumerate(JetSpec(self.dimension).coefficient_order)}
        full = result[:self.full_rows].view(self.full_points, self.q, self.dimension)
        for axis in range(self.dimension):
            alpha = tuple(int(axis == j) for j in range(self.dimension))
            full[:, lookup[alpha], axis] = 1.0
        return result

    def check(self, tensor):
        if tensor.ndim != 2 or tensor.shape[0] != self.rows or tensor.shape[1] < 1:
            raise ValueError("packed row/channel shape mismatch")
        if tensor.dtype != torch.float64 or not tensor.is_contiguous():
            raise ValueError("contiguous FP64 packed storage required")


def stable_a1(value):
    r = torch.exp(-value.abs())
    return ((2 * r) / (1 + r * r)).square()


@lru_cache(None)
def convolution_terms(dimension):
    indices = JetSpec(dimension).coefficient_order
    lookup = {a: k for k, a in enumerate(indices)}
    return tuple(tuple((i, lookup[tuple(x-y for x, y in zip(a, b))])
                       for i, b in enumerate(indices) if all(y <= x for x, y in zip(a, b)))
                 for a in indices)


def full_vjp(hidden, auxiliary, seed, dimension):
    terms = convolution_terms(dimension)
    derivative = [auxiliary]
    for alpha in range(1, len(terms)):
        value = torch.zeros_like(auxiliary)
        for beta, gamma in terms[alpha]:
            value = value - hidden[:, beta] * hidden[:, gamma]
        derivative.append(value)
    output = [torch.zeros_like(auxiliary) for _ in terms]
    for alpha, row in enumerate(terms):
        for beta, gamma in row:
            output[beta] = output[beta] + seed[:, alpha] * derivative[gamma]
    return torch.stack(output, dim=1)


class TensorActivation:
    """Reference/compiled tensor path; CUDA implementation writes packed output directly."""

    def __init__(self, layout, *, compiled=False, compile_options=None):
        self.layout = layout
        self.forward = self._forward
        self.vjp = self._vjp
        if compiled:
            options = dict(fullgraph=True, dynamic=False, **(compile_options or {}))
            self.forward = torch.compile(self._forward, **options)
            self.vjp = torch.compile(self._vjp, **options)

    def _forward(self, value):
        layout = self.layout
        layout.check(value)
        channels = value.shape[1]
        hidden = torch.empty_like(value)
        aux = value.new_empty((layout.points, channels))
        if layout.full_points:
            part, a1 = tanh_jet(value[:layout.full_rows].view(layout.full_points, layout.q, channels), layout.dimension)
            hidden[:layout.full_rows] = part.reshape(layout.full_rows, channels)
            aux[:layout.full_points] = a1
        if layout.value_points:
            boundary = value[layout.full_rows:]
            hidden[layout.full_rows:] = torch.tanh(boundary)
            aux[layout.full_points:] = stable_a1(boundary)
        return hidden, aux

    def _vjp(self, hidden, auxiliary, seed):
        layout = self.layout
        layout.check(hidden)
        layout.check(seed)
        if auxiliary.shape != (layout.points, hidden.shape[1]):
            raise ValueError("one stable auxiliary value per point/channel required")
        result = torch.empty_like(hidden)
        if layout.full_points:
            shape = (layout.full_points, layout.q, hidden.shape[1])
            full = full_vjp(hidden[:layout.full_rows].view(shape), auxiliary[:layout.full_points],
                            seed[:layout.full_rows].view(shape), layout.dimension)
            result[:layout.full_rows] = full.reshape(layout.full_rows, hidden.shape[1])
        if layout.value_points:
            result[layout.full_rows:] = seed[layout.full_rows:] * auxiliary[layout.full_points:]
        return result
