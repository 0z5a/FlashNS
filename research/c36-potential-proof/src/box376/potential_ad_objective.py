"""Parameter-autograd baseline over the same frozen C36 Taylor states.

This is a parameter-gradient reference, not an independent coordinate-AD
oracle. Native coordinate AD remains the independent physical-state oracle.
"""
import torch
import torch_c36 as c36

from box376.data import FAMILIES, flatten
from box376.normalized_objective import PreparedNormalizedADObjective
from box376.potential_geometry import native_field
from box376.c36_coordinate_oracle import native_c17
from box376.potential_objective import PreparedPotentialNormalizedObjective


class PreparedPotentialADObjective(PreparedPotentialNormalizedObjective):
    def __init__(self, data, calibration, *, coordinate_ad=False, geometry_chunk=256):
        super().__init__(data, calibration, geometry_chunk=geometry_chunk)
        self.independent = PreparedNormalizedADObjective(self._data, calibration)
        self.coordinate_ad = coordinate_ad

    def contributions(self, parameters, chunk):
        if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
            raise ValueError("Positive integer chunk required")
        self._consumer.guard(self._data)
        theta = flatten(parameters).detach().clone().requires_grad_(True)
        sizes = [w.shape for w, _ in parameters]
        offset, leaves = 0, []
        for shape in sizes:
            count = shape.numel()
            weight = theta[offset:offset + count].view(shape)
            offset += count
            bias = theta[offset:offset + shape[0]]
            offset += shape[0]
            leaves.append((weight, bias))
        for index, name in enumerate(FAMILIES):
            family = self._data["families"][name]
            inputs, gate = self._geometry[name]
            factor = self._consumer.coefficients[index] / self._consumer.denominators[index]
            for begin in range(0, len(inputs), chunk):
                sl = slice(begin, begin + chunk)
                if self.coordinate_ad:
                    x = family["points"][sl].detach().clone().requires_grad_(True)
                    if name == "interior":
                        state = native_c17(lambda p: native_field(leaves, p, self._data["eta"]), x)
                    else:
                        state = native_field(leaves, x, self._data["eta"])[:, :, None]
                else:
                    alpha, origin, scale = self._model_constants
                    weight, bias = leaves[0]
                    physical = [(weight * alpha, bias - (weight * alpha) @ origin)] + leaves[1:]
                    raw, _ = c36.forward(physical, inputs[sl])
                    state = c36.curl_pressure(c36.multiply(gate[sl], raw * scale))
                weights = family["weights"][sl]
                if name == "interior":
                    numerator = self.independent.independent_interior(
                        state, family["force"][sl], family["force_gradient"][sl],
                        weights, self._data["nu"], self._data["lambda_g"])
                else:
                    components = slice(3, 4) if name == "gauge" else slice(0, 3)
                    scale = self._consumer.pressure_inverse_square if name == "gauge" else self._consumer.velocity_inverse_square
                    difference = state[:, components, 0] - family["target"][sl, components]
                    numerator = (weights * difference.square().sum(-1) * scale).sum()
                gradient = torch.autograd.grad(numerator * factor, theta)[0]
                yield index, numerator.detach(), gradient.detach()
