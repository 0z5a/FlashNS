import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flashns.jet_packed import PackedLayout, TensorActivation
from flashns.ns_seed import NSLoss, compile_seed, explicit_seed, residuals
from flashns.pinn_packed import PackedPINN

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "sources/flashns_v8_reference"
sys.path.insert(0, str(REFERENCE / "src"))
spec = importlib.util.spec_from_file_location("v8_math_validation_reference", REFERENCE / "tests/verify_math.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
ref = reference.ref


def close(a, b):
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()
    return reference.close(a, b)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("counts", [(2, 3), (0, 3), (3, 0), (0, 0)])
def test_mixed_activation_transpose(dimension, counts):
    layout = PackedLayout(*counts, dimension)
    gen = torch.Generator().manual_seed(7421)
    z = torch.randn(layout.rows, 7, generator=gen, dtype=torch.float64) * 0.3
    seed = torch.randn(z.shape, generator=gen, dtype=z.dtype)
    ops = TensorActivation(layout)
    h, a = ops.forward(z)
    dz = ops.vjp(h, a, seed)
    if layout.full_points:
        shape = (layout.full_points, layout.q, 7)
        hr, ar = ref.tanh_forward(z[:layout.full_rows].reshape(shape).numpy(), dimension)
        dr = ref.tanh_vjp(hr, ar, seed[:layout.full_rows].reshape(shape).numpy(), dimension)
        close(h[:layout.full_rows].reshape(shape), hr)
        close(a[:layout.full_points], ar)
        close(dz[:layout.full_rows].reshape(shape), dr)
    if layout.value_points:
        zb = z[layout.full_rows:].numpy()
        close(h[layout.full_rows:], np.tanh(zb))
        close(dz[layout.full_rows:], seed[layout.full_rows:].numpy() * ref.stable_a1(zb))


@pytest.mark.parametrize("batch,scale", [(0, 0.4), (1, 0.01), (7, 0.4), (7, 3.0)])
def test_explicit_seed_against_independent_polynomials(batch, scale):
    rng = np.random.default_rng(61402)
    j = torch.tensor(rng.normal(size=(batch, 10, 3)) * scale, dtype=torch.float64)
    w = torch.tensor(rng.uniform(0.1, 1.0, batch) / max(batch, 1), dtype=torch.float64)
    if batch:
        w[0] = 0
    for viscosity in (0.0, 0.07, 0.2):
        loss = NSLoss(viscosity=viscosity)
        value, gradient = explicit_seed(j, w, loss)
        vr, gr = ref.residual_loss_seed(j.numpy(), w.numpy(), nu=viscosity)
        close(residuals(j, loss), ref.ns_residuals(j.numpy(), nu=viscosity))
        close(value, vr)
        close(gradient, gr)


@pytest.mark.parametrize("ni,nb,width,scale", [(5, 3, 7, 0.4), (17, 5, 32, 0.4), (3, 2, 7, 3.0), (1, 0, 7, 0.4), (0, 3, 7, 0.4), (0, 0, 7, 0.4)])
def test_all_layouts_and_seeds_against_nested_ad(ni, nb, width, scale):
    x, ws, bs, ni, pw, bw, target = reference.make_case(ni, nb, width=width, scale=scale)
    expected_value, expected_gradients = reference.nested_step(x, ws, bs, ni, pw, bw, target)
    tensors = [torch.tensor(p, dtype=torch.float64) for p in (x, pw, bw, target)]
    weights = [torch.tensor(p, dtype=torch.float64) for p in ws]
    biases = [torch.tensor(p, dtype=torch.float64) for p in bs]
    for layout in ("dense", "split", "compact"):
        for seed in ("autograd", "explicit"):
            step = PackedPINN(tensors[0], ni, *tensors[1:], layout=layout, seed=seed)
            value, gradients = step(weights, biases)
            close(value, expected_value)
            for actual, expected in zip(gradients, expected_gradients):
                close(actual, expected)
            expected_rows = (ni + nb) * 10 if layout == "dense" else ni * 10 + nb
            assert step.metadata["packed_rows_total"] == expected_rows


def test_actual_adam_updates_and_repeated_boundary_points():
    x, ws, bs, ni, pw, bw, target = reference.make_case(5, 3, width=7, seed=821)
    x[-1] = x[-2]
    original = [p.copy() for p in ws + bs]
    for layout in ("dense", "split", "compact"):
        ps = [torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in original]
        refs = [torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in original]
        step = PackedPINN(torch.tensor(x), ni, torch.tensor(pw), torch.tensor(bw), torch.tensor(target), layout=layout)
        optimizer = torch.optim.Adam(ps, lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        other = torch.optim.Adam(refs, lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        for _ in range(3):
            value, gradients = step(ps[:3], ps[3:])
            rp = [p.detach().numpy().copy() for p in refs]
            vr, gr = reference.nested_step(x, rp[:3], rp[3:], ni, pw, bw, target)
            close(value, vr)
            for parameter, gradient, reference_parameter, reference_gradient in zip(ps, gradients, refs, gr):
                parameter.grad = gradient
                reference_parameter.grad = torch.tensor(reference_gradient)
            optimizer.step()
            other.step()
            for actual, expected in zip(ps, refs):
                close(actual, expected)


def test_tensor_compiler_graph_and_weight_validation():
    j = torch.randn(7, 10, 3, dtype=torch.float64)
    w = torch.arange(7, dtype=torch.float64) / 21
    # Tests Dynamo capture only on CPU; target-device Inductor compilation is separate.
    compiled = compile_seed(backend="eager")
    for actual, expected in zip(compiled(j, w), explicit_seed(j, w)):
        close(actual, expected)
    for invalid in (torch.full((7,), -1.0, dtype=w.dtype), torch.full((7,), float("nan"), dtype=w.dtype)):
        with pytest.raises(ValueError):
            explicit_seed(j, invalid)
    with pytest.raises(ValueError):
        PackedLayout(-1, 3)
    with pytest.raises(NotImplementedError):
        PackedLayout(2, 1, 4)


def test_compact_cost_contract():
    dense = PackedLayout(2560, 0)
    compact = PackedLayout(2048, 512)
    assert dense.rows == 25600 and compact.rows == 20992
    assert 8 * dense.rows * 64 == 13107200
    assert 8 * compact.rows * 64 == 10747904
    assert dense.points == compact.points == 2560


def test_compact_keeps_one_gemm_per_affine_role():
    from torch.utils._python_dispatch import TorchDispatchMode

    class MatrixCalls(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.mm.default:
                self.shapes.append([tuple(arg.shape) for arg in args])
            return func(*args, **(kwargs or {}))

    x, ws, bs, ni, pw, bw, target = reference.make_case(5, 3, width=7)
    weights, biases = [[torch.tensor(p) for p in group] for group in (ws, bs)]
    for layout, calls, rows in (("dense", 8, 80), ("split", 16, 50), ("compact", 8, 53)):
        step = PackedPINN(torch.tensor(x), ni, torch.tensor(pw), torch.tensor(bw), torch.tensor(target), layout=layout)
        with MatrixCalls() as observed:
            step(weights, biases)
        assert len(observed.shapes) == calls
        assert observed.shapes[0][0] == (rows, 2)
        if layout == "compact":
            # Each forward, dgrad and wgrad sees the complete mixed row dimension.
            assert all(any(rows in shape for shape in operands) for operands in observed.shapes)
