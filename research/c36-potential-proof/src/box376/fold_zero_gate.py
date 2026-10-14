"""Opt-in exact constant folding of parameter-independent zero-gate rows."""
import copy

import torch

from box376.data import FAMILIES
from box376.potential_objective import PreparedPotentialNormalizedObjective


class ZeroGateFoldedObjective:
    """Keep full family sums while evaluating the network only on nonzero jets.

    A row is constant only when every one of the 36 gate channels is exactly
    zero. Its forcing, forcing-gradient or target loss is retained at the
    original family weight and normalization. No numerical threshold is used.
    Preparation is outside objective-gradient replay and does not change the
    supplied prepared engine or its default execution path.
    """
    def __init__(self, prepared):
        if not isinstance(prepared, PreparedPotentialNormalizedObjective):
            raise TypeError("A prepared C36 potential objective is required")
        if getattr(prepared, "coordinate_ad", False):
            raise ValueError("Zero-gate folding requires the Taylor-state engine")
        engine = copy.copy(prepared)
        engine._data = dict(prepared._data, families={})
        engine._geometry = {}
        self._engine = engine
        self._consumer = prepared._consumer
        self._constant_numerators = self._consumer.denominators.new_zeros(len(FAMILIES))
        self._masks, self.counts = {}, {}
        with torch.no_grad():
            for index, name in enumerate(FAMILIES):
                family = prepared._data["families"][name]
                inputs, gate = prepared._geometry[name]
                if gate.shape != (len(inputs), 1, 36) or not torch.isfinite(gate).all():
                    raise ValueError("Finite complete C36 gate geometry required")
                active = (gate != 0).any(dim=(1, 2))
                self._masks[name] = active.detach().clone()
                engine._geometry[name] = (inputs[active].clone(), gate[active].clone())
                engine._data["families"][name] = {
                    key: value[active].clone() if isinstance(value, torch.Tensor)
                    and value.ndim and value.shape[0] == len(inputs) else value
                    for key, value in family.items()}
                inactive = ~active
                weights = family["weights"][inactive]
                if name == "interior":
                    force = family["force"][inactive]
                    derivative = family["force_gradient"][inactive]
                    terms = (force.square() * self._consumer.residual_inverse_square[:3]).sum(1)
                    terms += prepared._data["lambda_g"] * (
                        derivative.square() * self._consumer.gradient_inverse_square[:3]).sum((1, 2))
                else:
                    components = slice(3, 4) if name == "gauge" else slice(0, 3)
                    scale = self._consumer.pressure_inverse_square if name == "gauge" else self._consumer.velocity_inverse_square
                    terms = family["target"][inactive, components].square().sum(1) * scale
                self._constant_numerators[index] = (weights * terms).sum()
                self.counts[name] = {"full": len(inputs), "active": int(active.sum().item()),
                    "folded": int(inactive.sum().item()), "original_W": family["W"]}

    def __call__(self, parameters, chunk):
        _, gradient, numerators = self._engine(parameters, chunk)
        numerators = numerators + self._constant_numerators
        loss = (numerators * self._consumer.coefficients / self._consumer.denominators).sum()
        return loss, gradient, numerators

    def proof_arrays(self, prepared):
        """Detached preparation evidence for a separate NumPy-only reader."""
        result = {"constant_numerators": self._constant_numerators,
            "denominators": self._consumer.denominators,
            "coefficients": self._consumer.coefficients,
            "residual_inverse_square": self._consumer.residual_inverse_square,
            "gradient_inverse_square": self._consumer.gradient_inverse_square,
            "velocity_inverse_square": self._consumer.velocity_inverse_square,
            "pressure_inverse_square": self._consumer.pressure_inverse_square}
        for name in FAMILIES:
            result[name + "_active"] = self._masks[name]
            result[name + "_gate"] = prepared._geometry[name][1]
        return {name: value.detach().cpu().numpy().copy() for name, value in result.items()}
