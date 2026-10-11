"""Complete fixed-source, globally normalized CPU objective and explicit VJP.

Physical input seeds are applied before contracting C17 derivatives. Reference
paths absorb the coordinate transform into the first affine layer and use AD.
"""
import torch

import torch_c17 as c17
from box376.data import COEFFICIENTS, FAMILIES, flatten
from torch_references import a24_contract, a24_forward, physical_loss, scalar_ad_state

SCALE = (1.25, .75, 1.5, .625)
ORIGIN = (.375, .375, .375, .5)


def transform(points, scale=SCALE, origin=ORIGIN):
    return (points-points.new_tensor(origin))*points.new_tensor(scale)


def forward(parameters, points, scale=SCALE, origin=ORIGIN):
    z = transform(points, scale, origin)
    s = c17.coordinate_state(z)
    for i in range(3):
        s[:, i, 1+i] = scale[i]
    s[:, 3, 13] = scale[3]
    tape = []
    for k, (W, b) in enumerate(parameters):
        pre = torch.einsum("nwq,ow->noq", s, W)
        pre[:, :, 0] += b
        tape.append((s, pre))
        s = c17.activate(pre) if k < len(parameters)-1 else pre
    return s, tape


def values(parameters, points, scale=SCALE, origin=ORIGIN):
    x = transform(points, scale, origin)
    for k, (W, b) in enumerate(parameters):
        x = x@W.T+b
        if k < len(parameters)-1:
            x = torch.tanh(x)
    return x


def physical_parameters(parameters, scale=SCALE, origin=ORIGIN):
    # An independent algebraic route to physical input derivatives. AD flows
    # through this map, so the resulting VJP is w.r.t. the original parameters.
    W, b = parameters[0]
    alpha = W.new_tensor(scale)
    shift = W.new_tensor(origin)
    return [(W*alpha, b-(W*alpha)@shift)] + parameters[1:]


def value_gradient(parameters, points, target, indices, weights):
    x = transform(points)
    tape = []
    for k, (W, b) in enumerate(parameters):
        pre = x@W.T+b
        tape.append((x, pre))
        x = torch.tanh(pre) if k < len(parameters)-1 else pre
    delta = x[:, indices]-target[:, indices]
    loss = (weights*(delta*delta).sum(-1)).sum()
    adjoint = torch.zeros_like(x)
    adjoint[:, indices] = 2*weights[:, None]*delta
    gradients = []
    for k in range(len(parameters)-1, -1, -1):
        previous, pre = tape[k]
        W, _ = parameters[k]
        if k < len(parameters)-1:
            adjoint = adjoint*c17.coefficients(pre)[1]
        gradients.append((adjoint.T@previous, adjoint.sum(0)))
        adjoint = adjoint@W
    return loss, list(reversed(gradients))


def local_objective(parameters, data, chunk):
    if chunk < 1:
        raise ValueError("Positive chunk required")
    numerators = parameters[0][0].new_zeros(len(FAMILIES))
    gradient = torch.zeros_like(flatten(parameters))
    for family_index, name in enumerate(FAMILIES):
        family = data["families"][name]
        denominator = family["W"]  # Frozen GLOBAL weight sum, even on empty ranks.
        coefficient = COEFFICIENTS[family_index]
        for begin in range(0, len(family["points"]), chunk):
            end = min(begin+chunk, len(family["points"]))
            x, w = family["points"][begin:end], family["weights"][begin:end]
            if name == "interior":
                state, tape = forward(parameters, x)
                numerator, seed = c17.loss_and_seed(state, family["force"][begin:end],
                                                  family["force_gradient"][begin:end], w,
                                                  nu=data["nu"], lambda_g=data["lambda_g"])
                gradients = c17.parameter_vjp(parameters, tape, seed*(coefficient/denominator))
            else:
                indices = slice(3, 4) if name == "gauge" else slice(0, 3)
                numerator, gradients = value_gradient(parameters, x, family["target"][begin:end], indices, w)
                gradients = [(gW*(coefficient/denominator), gb*(coefficient/denominator)) for gW, gb in gradients]
            numerators[family_index] += numerator
            gradient += flatten(gradients)
    sums = numerators.new_tensor([data["families"][name]["W"] for name in FAMILIES])
    loss = (numerators*numerators.new_tensor(COEFFICIENTS)/sums).sum()
    return loss, gradient, numerators


def reference_objective(parameters, data, backend="a24"):
    p = [(W.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True)) for W, b in parameters]
    physical = physical_parameters(p)
    family = data["families"]["interior"]
    if backend == "scalar_ad":
        state = scalar_ad_state(physical, family["points"])
    elif backend == "naive_q35":
        from e2e.naive_q35 import forward as qforward, contract
        state = contract(qforward(physical, family["points"]))
    elif backend == "a24":
        state = a24_contract(a24_forward(physical, family["points"]))
    else:
        raise ValueError(backend)
    numerators = [physical_loss(state, family["force"], family["force_gradient"], family["weights"],
                               nu=data["nu"], lambda_g=data["lambda_g"])]
    for name in FAMILIES[1:]:
        family = data["families"][name]
        selected = slice(3, 4) if name == "gauge" else slice(0, 3)
        delta = values(p, family["points"])[:, selected]-family["target"][:, selected]
        numerators.append((family["weights"]*(delta*delta).sum(-1)).sum())
    numerators = torch.stack(numerators)
    sums = numerators.new_tensor([data["families"][name]["W"] for name in FAMILIES])
    loss = (numerators*numerators.new_tensor(COEFFICIENTS)/sums).sum()
    gradients = torch.autograd.grad(loss, [value for layer in p for value in layer])
    return loss.detach(), torch.cat([value.reshape(-1) for value in gradients]).detach(), numerators.detach(), state.detach()
