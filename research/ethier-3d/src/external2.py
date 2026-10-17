"""Compiled C17 builder for the second attempt of the compiled Ethier batch (protocol-ext3-20261017).

The first attempt's builder (external.build_compiled) traced C17 with make_fx under no_grad; every
compiled C17 case then failed during CUDA Graph capture ("Attempt to trace generator"). A one-GPU
diagnostic reproduced that failure and found that compiling a function which takes the six parameter
tensors as explicit arguments, called under no_grad, survives capture and matches eager C17 after
in-place parameter updates. This module provides that builder for C17 and delegates the external
methods to external.build_compiled unchanged.
"""
import torch

import external


def build_compiled(parameters, data, protocol, method):
    if method != 'c17':
        return external.build_compiled(parameters, data, protocol, method)
    from objective import explicit_objective
    flat = [t for pair in parameters for t in pair]

    def objective(W0, b0, W1, b1, W2, b2):
        return explicit_objective([(W0, b0), (W1, b1), (W2, b2)], data, protocol)
    compiled = torch.compile(objective, fullgraph=True, dynamic=False)

    def call():
        with torch.no_grad():
            return compiled(*flat)
    return call
