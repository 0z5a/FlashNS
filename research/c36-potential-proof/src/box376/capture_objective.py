"""Complete C17 objective with fixed constants constructed before capture.

The coordinate transform, physical derivative seeds, PDE consumer and parameter
transpose are the same as box376.objective. Only host-to-device constant creation
is moved into preparation. NCCL and optimizer control remain outside this class.
"""
import torch

import torch_c17 as c17
from box376.data import COEFFICIENTS, FAMILIES, flatten
from box376.objective import ORIGIN, SCALE


class PreparedObjective:
    def __init__(self, data):
        prototype = data["families"]["interior"]["points"]
        self.origin = prototype.new_tensor(ORIGIN)
        self.scale = prototype.new_tensor(SCALE)
        self.denominators = prototype.new_tensor([data["families"][name]["W"] for name in FAMILIES])
        self.coefficients = prototype.new_tensor(COEFFICIENTS)
        self.source_cache_key = data["metadata"]["cache_key"]
        self.source_version = data["metadata"]["source_version"]
        self.constants = (self.origin, self.scale, self.denominators, self.coefficients)
        self.constant_addresses = tuple(value.data_ptr() for value in self.constants)

    def transform(self, points):
        return (points-self.origin)*self.scale

    def forward(self, parameters, points):
        state = c17.coordinate_state(self.transform(points))
        for i in range(3):
            state[:, i, 1+i] = SCALE[i]
        state[:, 3, 13] = SCALE[3]
        tape = []
        for k, (weight, bias) in enumerate(parameters):
            pre = torch.einsum("nwq,ow->noq", state, weight)
            pre[:, :, 0] += bias
            tape.append((state, pre))
            state = c17.activate(pre) if k < len(parameters)-1 else pre
        return state, tape

    def value_gradient(self, parameters, points, target, indices, weights):
        x = self.transform(points)
        tape = []
        for k, (weight, bias) in enumerate(parameters):
            pre = x@weight.T+bias
            tape.append((x, pre))
            x = torch.tanh(pre) if k < len(parameters)-1 else pre
        difference = x[:, indices]-target[:, indices]
        loss = (weights*(difference*difference).sum(-1)).sum()
        adjoint = torch.zeros_like(x)
        adjoint[:, indices] = 2*weights[:, None]*difference
        gradients = []
        for k in range(len(parameters)-1, -1, -1):
            previous, pre = tape[k]
            weight, _ = parameters[k]
            if k < len(parameters)-1:
                adjoint = adjoint*c17.coefficients(pre)[1]
            gradients.append((adjoint.T@previous, adjoint.sum(0)))
            adjoint = adjoint@weight
        return loss, list(reversed(gradients))

    def __call__(self, parameters, data, chunk):
        if chunk < 1:
            raise ValueError("Positive chunk required")
        if data["metadata"]["cache_key"] != self.source_cache_key or data["metadata"]["source_version"] != self.source_version:
            raise ValueError("Prepared constants belong to a different source cache")
        if tuple(value.data_ptr() for value in self.constants) != self.constant_addresses:
            raise ValueError("Prepared constant storage changed")
        numerators = parameters[0][0].new_zeros(len(FAMILIES))
        gradient = torch.zeros_like(flatten(parameters))
        for family_index, name in enumerate(FAMILIES):
            family = data["families"][name]
            denominator = family["W"]
            coefficient = COEFFICIENTS[family_index]
            for begin in range(0, len(family["points"]), chunk):
                end = min(begin+chunk, len(family["points"]))
                points, weights = family["points"][begin:end], family["weights"][begin:end]
                if name == "interior":
                    state, tape = self.forward(parameters, points)
                    numerator, seed = c17.loss_and_seed(state, family["force"][begin:end],
                                                      family["force_gradient"][begin:end], weights,
                                                      nu=data["nu"], lambda_g=data["lambda_g"])
                    gradients = c17.parameter_vjp(parameters, tape, seed*(coefficient/denominator))
                else:
                    indices = slice(3, 4) if name == "gauge" else slice(0, 3)
                    numerator, gradients = self.value_gradient(parameters, points, family["target"][begin:end], indices, weights)
                    gradients = [(gW*(coefficient/denominator), gb*(coefficient/denominator)) for gW, gb in gradients]
                numerators[family_index] += numerator
                gradient += flatten(gradients)
        loss = (numerators*self.coefficients/self.denominators).sum()
        return loss, gradient, numerators
