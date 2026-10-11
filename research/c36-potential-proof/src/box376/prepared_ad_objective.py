"""A24 / Naive complete AD objective using preconstructed device constants.

Both baselines use their independent Taylor propagation and native autograd.
They share only the physical problem, batching and fixed normalization with C17.
"""
import torch

from box376.capture_objective import PreparedObjective
from box376.data import COEFFICIENTS, FAMILIES, flatten
from torch_references import a24_contract, a24_forward, physical_loss


class PreparedADObjective(PreparedObjective):
    def __init__(self, data, backend):
        super().__init__(data)
        if backend == "a24":
            self.propagate, self.contract = a24_forward, a24_contract
        elif backend == "naive_q35":
            from e2e.naive_q35 import forward, contract
            self.propagate, self.contract = forward, contract
        else:
            raise ValueError(backend)
        self.backend = backend

    def physical_parameters(self, parameters):
        weight, bias = parameters[0]
        physical_weight = weight*self.scale
        return [(physical_weight, bias-physical_weight@self.origin)]+parameters[1:]

    def values(self, parameters, points):
        value = self.transform(points)
        for index, (weight, bias) in enumerate(parameters):
            value = value@weight.T+bias
            if index < len(parameters)-1:
                value = torch.tanh(value)
        return value

    def __call__(self, parameters, data, chunk):
        if chunk < 1:
            raise ValueError("Positive chunk required")
        if data["metadata"]["cache_key"] != self.source_cache_key or data["metadata"]["source_version"] != self.source_version:
            raise ValueError("Prepared constants belong to a different source cache")
        if tuple(value.data_ptr() for value in self.constants) != self.constant_addresses:
            raise ValueError("Prepared constant storage changed")
        p = [(weight.detach().clone().requires_grad_(True), bias.detach().clone().requires_grad_(True))
             for weight, bias in parameters]
        leaves = [value for layer in p for value in layer]
        numerators = parameters[0][0].new_zeros(len(FAMILIES))
        gradient = torch.zeros_like(flatten(parameters))
        for index, name in enumerate(FAMILIES):
            family = data["families"][name]
            for begin in range(0, len(family["points"]), chunk):
                end = min(begin+chunk, len(family["points"]))
                points = family["points"][begin:end]
                weights = family["weights"][begin:end]
                if name == "interior":
                    state = self.contract(self.propagate(self.physical_parameters(p), points))
                    numerator = physical_loss(state, family["force"][begin:end],
                                              family["force_gradient"][begin:end], weights,
                                              nu=data["nu"], lambda_g=data["lambda_g"])
                else:
                    indices = slice(3, 4) if name == "gauge" else slice(0, 3)
                    difference = self.values(p, points)[:, indices]-family["target"][begin:end, indices]
                    numerator = (weights*(difference*difference).sum(-1)).sum()
                derivatives = torch.autograd.grad(numerator*(COEFFICIENTS[index]/family["W"]), leaves)
                gradient += torch.cat([value.reshape(-1) for value in derivatives])
                numerators[index] += numerator.detach()
        loss = (numerators*self.coefficients/self.denominators).sum()
        return loss, gradient, numerators
