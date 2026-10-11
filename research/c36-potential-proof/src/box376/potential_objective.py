"""Prepared normalized objective for the C36 curl-potential representation."""
import copy
import math

import torch

from box376.data import FAMILIES, flatten
from box376.normalized_objective import PreparedNormalizedObjective
from box376.potential_model import forward, parameter_vjp, prepare_geometry


class PreparedPotentialNormalizedObjective:
    """Freeze physical inputs and geometry before evaluating parameter gradients.

    Source tensors are cloned, so later changes to the caller's data cannot make
    the prepared geometry disagree with its force, targets or weights. This
    explicit-gradient interface returns detached loss, gradient and family
    numerators; it does not install an autograd backward or update parameters.
    """
    def __init__(self, data, calibration, *, geometry_chunk=256):
        self._data = copy.deepcopy({key: value for key, value in data.items() if key != "families"})
        self._data["families"] = {}
        self._geometry = {}
        for name in FAMILIES:
            family = data["families"][name]
            if not math.isfinite(family["W"]) or family["W"] <= 0:
                raise ValueError("Positive finite family weight sums required")
            frozen = {key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                      for key, value in family.items()}
            self._data["families"][name] = frozen
        self._consumer = PreparedNormalizedObjective(self._data, calibration)
        for name in FAMILIES:
            points = self._data["families"][name]["points"]
            self._geometry[name] = prepare_geometry(points, self._data["eta"], chunk_size=geometry_chunk)

    def contributions(self, parameters, chunk):
        """Yield family index, raw weighted numerator and normalized chunk gradient."""
        if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
            raise ValueError("Positive integer chunk required")
        self._consumer.guard(self._data)
        for index, name in enumerate(FAMILIES):
            family = self._data["families"][name]
            inputs, gate = self._geometry[name]
            factor = self._consumer.coefficients[index] / self._consumer.denominators[index]
            for begin in range(0, len(inputs), chunk):
                sl = slice(begin, begin + chunk)
                state, tape = forward(parameters, inputs[sl], gate[sl])
                weights = family["weights"][sl]
                if name == "interior":
                    numerator, bar = self._consumer.interior(state, family["force"][sl],
                                                            family["force_gradient"][sl], weights,
                                                            self._data["nu"], self._data["lambda_g"])
                else:
                    components = slice(3, 4) if name == "gauge" else slice(0, 3)
                    scale = self._consumer.pressure_inverse_square if name == "gauge" else self._consumer.velocity_inverse_square
                    difference = state[:, components, 0] - family["target"][sl, components]
                    numerator = (weights * difference.square().sum(-1) * scale).sum()
                    bar = torch.zeros_like(state)
                    bar[:, components, 0] = 2 * weights[:, None] * difference * scale
                gradient = flatten(parameter_vjp(tape, bar * factor))
                yield index, numerator.detach(), gradient.detach()

    def __call__(self, parameters, chunk):
        numerators = parameters[0][0].new_zeros(len(FAMILIES))
        gradient = torch.zeros_like(flatten(parameters))
        for index, numerator, contribution in self.contributions(parameters, chunk):
            numerators[index] += numerator
            gradient += contribution
        loss = (numerators * self._consumer.coefficients / self._consumer.denominators).sum()
        return loss, gradient, numerators
