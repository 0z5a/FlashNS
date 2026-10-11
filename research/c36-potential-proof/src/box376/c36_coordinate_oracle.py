"""Coordinate-AD oracle for experimental 36-state potential algebra only.

Not a production fast kernel. Retain native derivative graphs when a complete
parameter gradient must be compared. Spatial fourth derivatives are independent
of the explicit activation/product identities in torch_c36.
"""
import torch
import torch_c36 as c36


def derivative_table(function, points):
    values = function(points)
    if values.ndim == 1:
        values = values[:, None]
    cache = {(): values}

    def derivative(indices):
        indices = tuple(sorted(indices))
        if indices in cache:
            return cache[indices]
        parent = derivative(indices[:-1])
        columns = []
        for component in range(parent.shape[1]):
            value = parent[:, component]
            gradient = torch.autograd.grad(value.sum(), points, create_graph=True,
                                           retain_graph=True, allow_unused=True)[0] if value.requires_grad else None
            columns.append(points[:, indices[-1]] * 0 if gradient is None else gradient[:, indices[-1]])
        cache[indices] = torch.stack(columns, 1)
        return cache[indices]

    return values, derivative


def native_c36(function, points):
    values, derivative = derivative_table(function, points)
    columns = [values] + [derivative((i,)) for i in range(3)]
    columns += [derivative(pair) for pair in c36.PAIRS]
    columns += [derivative(triple) for triple in c36.TRIPLES]
    columns += [sum(derivative(tuple(pair) + (k, k)) for k in range(3)) for pair in c36.PAIRS]
    columns += [derivative((3,))] + [derivative((i, 3)) for i in range(3)]
    columns += [derivative(tuple(pair) + (3,)) for pair in c36.PAIRS]
    return torch.stack(columns, -1)


def native_c17(function, points):
    values, derivative = derivative_table(function, points)
    columns = [values] + [derivative((i,)) for i in range(3)]
    columns += [derivative(pair) for pair in c36.PAIRS]
    columns += [sum(derivative((i, k, k)) for k in range(3)) for i in range(3)]
    columns += [derivative((3,))] + [derivative((i, 3)) for i in range(3)]
    return torch.stack(columns, -1)
