"""Versioned full-Taylor jet contract for the independent CUDA experiments."""

from dataclasses import dataclass
from itertools import product
from math import factorial, prod


@dataclass(frozen=True)
class JetSpec:
    coordinate_dimension: int
    schema_version: int = 1
    basis_revision: int = 1
    max_derivative_order: int = 3

    def __post_init__(self):
        if (
            self.coordinate_dimension not in (2, 3)
            or self.max_derivative_order != 3
            or self.schema_version != 1
            or self.basis_revision != 1
        ):
            raise NotImplementedError(
                "Only full Taylor 2-D/3-D order-3 basis revision 1 is implemented"
            )

    @property
    def basis_id(self):
        return f"full_taylor_{self.coordinate_dimension}d3"

    @property
    def coefficient_order(self):
        return tuple(
            sorted(
                (
                    a
                    for a in product(range(4), repeat=self.coordinate_dimension)
                    if sum(a) <= 3
                ),
                key=lambda a: (sum(a), a),
            )
        )

    @property
    def q(self):
        return len(self.coefficient_order)

    @property
    def derivative_recovery_diagonal(self):
        return tuple(prod(factorial(v) for v in a) for a in self.coefficient_order)

    def recover_raw_derivatives(self, coefficients):
        import numpy as np

        values = np.asarray(coefficients)
        if values.shape[-1] != self.q:
            raise ValueError("last axis must contain the complete ordered jet")
        return values * np.asarray(self.derivative_recovery_diagonal)

    def pullback_raw_derivative_seed(self, raw_seed):
        # The recovery map is diagonal, so its transpose has the same factors.
        return self.recover_raw_derivatives(raw_seed)

    def require_operation(self, operation):
        if operation not in {
            "affine",
            "tanh",
            "pointwise_fixed_loss",
            "first_parameter_vjp",
            "dgrad_jet_vjp",
        }:
            raise NotImplementedError(
                f"JetSpec {self.basis_id} does not support {operation}"
            )

    def manifest(self, *, generated_source_hash):
        if len(generated_source_hash) != 64 or any(
            c not in "0123456789abcdef" for c in generated_source_hash
        ):
            raise ValueError("generated source SHA-256 required")
        return {
            "schema_version": self.schema_version,
            "basis_id": self.basis_id,
            "basis_revision": self.basis_revision,
            "coordinate_dimension": self.coordinate_dimension,
            "max_derivative_order": self.max_derivative_order,
            "coefficient_order": [list(a) for a in self.coefficient_order],
            "coefficient_normalization": "partial^alpha / alpha!",
            "q": self.q,
            "primal_projection": "identity on the specified full Taylor basis",
            "output_recovery": {
                "type": "diagonal",
                "entries": self.derivative_recovery_diagonal,
            },
            "cotangent_pairing": "sum_alpha bar_H_alpha * H_alpha; transpose of derivative recovery maps raw derivative seeds",
            "forward_rule": "truncated Taylor tanh composition through total degree three",
            "vjp_rule": "bar_Z_beta = sum_{alpha>=beta} bar_H_alpha * (1-H*H)_{alpha-beta}",
            "activation_aux_state": {
                "strategy": "H_only_ordinary_fp64",
                "stored": ["H"],
                "tail_accuracy": "unsupported; recorded high-precision saturation failures retained",
            },
            "tensor_layout": "contiguous [B,Q,C]; C contiguous; no alias",
            "loss_reduction": "normalization over B is specified by the loss; never introduce division by Q",
            "operator_scope": "affine+tanh, fixed pointwise loss, first parameter VJP",
            "unsupported_operations": [
                "double_backward",
                "hvp",
                "c13",
                "nonlinear_coordinate_transform",
                "nonlocal_loss",
                "batch_norm",
            ],
            "symbolic_certificate_hash": None,
            "generated_source_hash": generated_source_hash,
        }
