import numpy as np
import pytest

from flashns.jet_spec import JetSpec


@pytest.mark.parametrize("dim,q", [(2, 10), (3, 20)])
def test_full_jet_recovery_and_cotangent_duality(dim, q):
    spec = JetSpec(dim)
    assert spec.q == q
    assert spec.coefficient_order[0] == (0,) * dim
    # Independent polynomial x_0^3 + 2*x_0*x_1^2: raw derivatives are 6 and 4.
    coefficients = np.zeros(q)
    cube = (3,) + (0,) * (dim - 1)
    mixed = (1, 2) + (0,) * (dim - 2)
    coefficients[spec.coefficient_order.index(cube)] = 1
    coefficients[spec.coefficient_order.index(mixed)] = 2
    derivatives = spec.recover_raw_derivatives(coefficients)
    assert derivatives[spec.coefficient_order.index(cube)] == 6
    assert derivatives[spec.coefficient_order.index(mixed)] == 4
    raw_seed = np.arange(q) / 7
    assert np.dot(raw_seed, derivatives) == pytest.approx(
        np.dot(spec.pullback_raw_derivative_seed(raw_seed), coefficients)
    )


@pytest.mark.parametrize(
    "operation", ["hvp", "double_backward", "c13", "nonlocal_loss"]
)
def test_unsupported_transforms_are_explicit(operation):
    with pytest.raises(NotImplementedError, match=operation):
        JetSpec(3).require_operation(operation)


def test_manifest_does_not_invent_certificates_or_tail_accuracy():
    manifest = JetSpec(2).manifest(generated_source_hash="a" * 64)
    assert manifest["symbolic_certificate_hash"] is None
    assert manifest["activation_aux_state"]["tail_accuracy"].startswith("unsupported")
    assert manifest["generated_source_hash"] == "a" * 64
