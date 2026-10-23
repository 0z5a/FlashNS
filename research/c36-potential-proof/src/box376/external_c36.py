"""External derivative backends for the C36 network state (nested autograd, torch.func, collapsed jet).

The C36 engines evaluate the network on prepared geometry features F(x). The features are affine in
space, and the state needs at most one time derivative, so at each point the polynomial

    F~(d) = F0 + J d_s + v d_t + (K d_s) d_t,      d = (d_s, d_t) in R^3 x R,

with F0, J, v and K read from the prepared geometry state (value, spatial gradient, dt and the
spatial gradient of dt), has the same derivatives at d = 0 as F has at the point, to every order the
state uses. Each backend below differentiates the same function G(d) = MLP(F~(d)) at d = 0 and
returns the 36-channel raw network state in the torch_c36 layout. Gate, curl, residual and loss are
the unchanged C36 consumer code, and the parameter gradient is PyTorch autograd in every backend,
so only the coordinate-derivative method differs from the C36 engines.

- nested_ad: nested scalar-coordinate PyTorch autograd over the batch, as an ordinary PINN code
  writes it (box376.c36_coordinate_oracle.native_c36 applied to G).
- func_jacfwd: torch.func, per point jacfwd nested to third order plus the Hessian of the spatial
  Laplacian, vectorized with vmap.
- jet_collapsed: Dangel et al.'s public `jet` package (collapsed Taylor mode, pinned snapshot
  2e49005ee4ae145f375925e38f0a255f82369dda): one collapsed second-order jet along the three spatial
  directions gives the value, the spatial first derivatives and the spatial Laplacian; a first-order
  jet along time gives dt. As in the Ethier and 2D adapters, the remaining derivatives use autograd.
"""
import os
import sys

import torch
import torch_c36 as c36

from box376.c36_coordinate_oracle import native_c36

BACKENDS = ('nested_ad', 'func_jacfwd', 'jet_collapsed')


def surrogate(inputs):
    """F0 [N,4], J [N,4,3], v [N,4], K [N,4,3] from a prepared geometry state [N,4,36]."""
    return inputs[:, :, 0], inputs[:, :, 1:4], inputs[:, :, 26], inputs[:, :, 27:30]


def mlp(parameters, features):
    x = features
    for k, (weight, bias) in enumerate(parameters):
        x = x @ weight.T + bias
        if k < len(parameters) - 1:
            x = torch.tanh(x)
    return x


def axis_diagonals(inputs):
    """Per-feature coefficients of F~ in its elementwise form (requires assert_axis_aligned)."""
    F0, J, v, K = surrogate(inputs)
    pad = torch.zeros_like(F0[:, :1])
    return (F0, torch.cat((J[:, :3, :3].diagonal(dim1=-2, dim2=-1), pad), 1), v,
            torch.cat((K[:, :3, :3].diagonal(dim1=-2, dim2=-1), pad), 1))


def _features(d, F0, Jd, v, Kd):
    # d = (d_s, d_t); feature a < 3 depends on d_a and d_t, the time feature on d_t only.
    # Built on the device: a host-to-device copy is not allowed during CUDA Graph capture.
    lifted, time = torch.cat((d[..., :3], torch.zeros_like(d[..., 3:4])), -1), d[..., 3:4]
    return F0 + lifted * Jd + v * time + lifted * Kd * time


def raw_nested_ad(parameters, inputs):
    F0, J, v, K = axis_diagonals(inputs)
    d = inputs.new_zeros((len(inputs), 4), requires_grad=True)
    state = native_c36(lambda point: mlp(parameters, _features(point, F0, J, v, K)), d)
    return state


def raw_func_jacfwd(parameters, inputs):
    from torch.func import jacfwd, vmap
    F0, J, v, K = axis_diagonals(inputs)

    def g(d, f0, j, vv, k):
        return mlp(parameters, _features(d, f0, j, vv, k))

    d1, d2, d3 = jacfwd(g), jacfwd(jacfwd(g)), jacfwd(jacfwd(jacfwd(g)))

    def laplacian(d, f0, j, vv, k):
        return d2(d, f0, j, vv, k)[:, :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1)

    d4 = jacfwd(jacfwd(laplacian))
    zero = inputs.new_zeros((len(inputs), 4))
    args = (zero, F0, J, v, K)
    value = vmap(g)(*args)
    first, second, third, fourth = vmap(d1)(*args), vmap(d2)(*args), vmap(d3)(*args), vmap(d4)(*args)
    return c36.pack(value, first[..., :3], second[..., :3, :3], third[..., :3, :3, :3], fourth[..., :3, :3],
                    first[..., 3], second[..., :3, 3], third[..., :3, :3, 3])


def _jet_module():
    root = os.environ.get('FLASHNS_TORCHJET')
    if not root:
        raise RuntimeError('FLASHNS_TORCHJET must point at the pinned torch-jet snapshot')
    if root not in sys.path:
        sys.path.insert(0, root)
    from jet import jet
    return jet


def _grad(z, x):
    return torch.autograd.grad(z.sum(), x, create_graph=True, retain_graph=True)[0]


def assert_axis_aligned(inputs):
    """The jet surrogate needs diagonal spatial feature gradients and a time feature without spatial gradient."""
    _, J, _, K = surrogate(inputs)
    eye3 = torch.eye(3, dtype=inputs.dtype, device=inputs.device)
    for M in (J, K):
        if (M[:, :3, :3] * (1 - eye3)).abs().max() != 0 or M[:, 3, :].abs().max() != 0:
            raise ValueError('geometry state is not axis-aligned; the jet surrogate does not apply')


def raw_jet_collapsed(parameters, inputs):
    jet = _jet_module()
    F0, J, v, K = surrogate(inputs)
    # jet has no rule for slicing or einsum. Each spatial feature depends on one coordinate and the
    # time feature on time only (assert_axis_aligned, checked once per geometry), so F~ is
    # elementwise in (d_s, d_t).
    eye3 = torch.eye(3, dtype=inputs.dtype, device=inputs.device)
    Jd, Kd = J[:, :3, :3].diagonal(dim1=-2, dim2=-1), K[:, :3, :3].diagonal(dim1=-2, dim2=-1)
    Jd, Kd = torch.cat((Jd, torch.zeros_like(Jd[:, :1])), 1), torch.cat((Kd, torch.zeros_like(Kd[:, :1])), 1)
    lift = torch.cat((eye3, torch.zeros_like(eye3[:, :1])), 1)  # [3, 4]
    n = len(inputs)
    ds = inputs.new_zeros((n, 3), requires_grad=True)
    dt = inputs.new_zeros((n, 1), requires_grad=True)

    def G(spatial, time):
        lifted = spatial @ lift
        return mlp(parameters, F0 + lifted * Jd + v * time + lifted * Kd * time)

    # Built per call: the jets bind this call's parameters and geometry.
    collapsed, standard = jet(G, (ds, dt), collapsed=True), jet(G, (ds, dt))
    directions = eye3[:, None, :].expand(3, n, 3)
    value, first, laplacian = collapsed((ds, directions, torch.zeros_like(ds)),
                                        (dt, dt.new_zeros((3, n, 1)), torch.zeros_like(dt)))
    _, time = standard((ds, torch.zeros_like(ds)), (dt, torch.ones_like(dt)))
    gradient = first.permute(1, 2, 0)
    hessian = torch.stack([torch.stack([_grad(first[i][:, c], ds) for i in range(3)], 1) for c in range(4)], 1)
    # One autograd call per component and unique pair (i <= j); symmetry fills the rest.
    unique = {(i, j): torch.stack([_grad(hessian[:, c, i, j], ds) for c in range(4)], 1) for i, j in c36.PAIRS}
    third = torch.stack([torch.stack([unique[(min(i, j), max(i, j))] for j in range(3)], 2) for i in range(3)], 2)
    lap_gradient = torch.stack([_grad(laplacian[:, c], ds) for c in range(4)], 1)
    fourth = torch.stack([torch.stack([_grad(lap_gradient[:, c, i], ds) for i in range(3)], 1) for c in range(4)], 1)
    time_gradient = torch.stack([_grad(time[:, c], ds) for c in range(4)], 1)
    time_hessian = torch.stack([torch.stack([_grad(time_gradient[:, c, i], ds) for i in range(3)], 1) for c in range(4)], 1)
    return c36.pack(value, gradient, hessian, third, fourth, time, time_gradient, time_hessian)


RAW = {'nested_ad': raw_nested_ad, 'func_jacfwd': raw_func_jacfwd, 'jet_collapsed': raw_jet_collapsed}


class PreparedPotentialExternalObjective:
    """The C36 objective and complete parameter gradient with an external coordinate-derivative backend.

    Same frozen data, geometry, consumer and family weights as the C36 engines
    (box376.potential_ad_objective.PreparedPotentialADObjective, Taylor branch); only the network
    state comes from RAW[backend], and the parameter gradient is autograd.
    """

    def __init__(self, data, calibration, backend, *, geometry_chunk=256):
        from box376.potential_ad_objective import PreparedPotentialADObjective
        if backend not in RAW:
            raise ValueError(f'unknown backend {backend}')
        self.backend = backend
        self._base = PreparedPotentialADObjective(data, calibration, geometry_chunk=geometry_chunk)
        for inputs, _ in self._base._geometry.values():
            assert_axis_aligned(inputs)

    def __call__(self, parameters, chunk):
        from box376.data import FAMILIES, flatten
        base = self._base
        if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
            raise ValueError('Positive integer chunk required')
        base._consumer.guard(base._data)
        theta = flatten(parameters).detach().clone().requires_grad_(True)
        offset, leaves = 0, []
        for weight0, _ in parameters:
            count = weight0.numel()
            weight = theta[offset:offset + count].view(weight0.shape)
            offset += count
            bias = theta[offset:offset + weight0.shape[0]]
            offset += weight0.shape[0]
            leaves.append((weight, bias))
        alpha, origin, scale = base._model_constants
        numerators = theta.new_zeros(len(FAMILIES))
        gradient = torch.zeros_like(theta)
        for index, name in enumerate(FAMILIES):
            family = base._data['families'][name]
            inputs, gate = base._geometry[name]
            factor = base._consumer.coefficients[index] / base._consumer.denominators[index]
            for begin in range(0, len(inputs), chunk):
                sl = slice(begin, begin + chunk)
                weight, bias = leaves[0]
                physical = [(weight * alpha, bias - (weight * alpha) @ origin)] + leaves[1:]
                raw = RAW[self.backend](physical, inputs[sl])
                state = c36.curl_pressure(c36.multiply(gate[sl], raw * scale))
                weights = family['weights'][sl]
                if name == 'interior':
                    numerator = base.independent.independent_interior(
                        state, family['force'][sl], family['force_gradient'][sl],
                        weights, base._data['nu'], base._data['lambda_g'])
                else:
                    components = slice(3, 4) if name == 'gauge' else slice(0, 3)
                    inverse = base._consumer.pressure_inverse_square if name == 'gauge' else base._consumer.velocity_inverse_square
                    difference = state[:, components, 0] - family['target'][sl, components]
                    numerator = (weights * difference.square().sum(-1) * inverse).sum()
                gradient += torch.autograd.grad(numerator * factor, theta)[0]
                numerators[index] += numerator.detach()
        loss = (numerators * base._consumer.coefficients / base._consumer.denominators).sum()
        return loss.detach(), gradient.detach(), numerators


def build_compiled_external(data, calibration, backend, theta, topology, chunk, *, compile=True):
    """Zero-argument compiled loss/gradient over a persistent parameter leaf (the Ethier pattern).

    The complete evaluation, including the coordinate derivatives, the consumer and the autograd
    parameter gradient, is traced once with make_fx and compiled with torch.compile (default mode).
    The trace reads `theta` in place, so optimizer updates of that leaf are seen on every call; the
    caller may capture the compiled callable in a CUDA Graph.
    """
    from torch.fx.experimental.proxy_tensor import make_fx
    from box376.data import FAMILIES, views
    objective = PreparedPotentialExternalObjective(data, calibration, backend)
    base = objective._base
    alpha, origin, scale = base._model_constants

    def run():
        with torch.enable_grad():
            leaves = views(theta, topology)
            numerators, gradient = [], torch.zeros_like(theta)
            for index, name in enumerate(FAMILIES):
                family = base._data['families'][name]
                inputs, gate = base._geometry[name]
                factor = base._consumer.coefficients[index] / base._consumer.denominators[index]
                total = theta.new_zeros(())
                for begin in range(0, len(inputs), chunk):
                    # One gradient per chunk, as in the eager wrapper, so memory follows the chunk size.
                    weight, bias = leaves[0]
                    physical = [(weight * alpha, bias - (weight * alpha) @ origin)] + list(leaves[1:])
                    sl = slice(begin, begin + chunk)
                    raw = RAW[backend](physical, inputs[sl])
                    state = c36.curl_pressure(c36.multiply(gate[sl], raw * scale))
                    weights = family['weights'][sl]
                    if name == 'interior':
                        numerator = base.independent.independent_interior(
                            state, family['force'][sl], family['force_gradient'][sl], weights, base._data['nu'], base._data['lambda_g'])
                    else:
                        components = slice(3, 4) if name == 'gauge' else slice(0, 3)
                        inverse = base._consumer.pressure_inverse_square if name == 'gauge' else base._consumer.velocity_inverse_square
                        difference = state[:, components, 0] - family['target'][sl, components]
                        numerator = (weights * difference.square().sum(-1) * inverse).sum()
                    gradient = gradient + torch.autograd.grad(numerator * factor, theta)[0]
                    total = total + numerator.detach()
                numerators.append(total)
            numerators = torch.stack(numerators)
            loss = (numerators * base._consumer.coefficients / base._consumer.denominators).sum()
        return loss, gradient, numerators

    traced = make_fx(run)()
    return torch.compile(traced, fullgraph=True, dynamic=False) if compile else traced



def build_chunked_traced_external(data, calibration, backend, theta, topology, chunk):
    """Zero-argument loss/gradient whose memory follows the chunk size.

    build_compiled_external traces the whole evaluation as one graph, whose generated code keeps
    every intermediate until it returns, so its memory does not fall with the chunk size. Here one
    function per family and chunk shape is traced with make_fx, with the chunk's data as explicit
    arguments and the parameter leaf read in place; the evaluation calls these pieces chunk by chunk
    and accumulates. Each piece's intermediates are released when it returns. The returned callable
    owns everything the pieces read, so it may be captured in a CUDA Graph.
    """
    from torch.fx.experimental.proxy_tensor import make_fx
    from box376.data import FAMILIES, views
    objective = PreparedPotentialExternalObjective(data, calibration, backend)
    base = objective._base
    alpha, origin, scale = base._model_constants

    def piece(name, index):
        factor = base._consumer.coefficients[index] / base._consumer.denominators[index]

        def run(inputs, gate, weights, *extra):
            with torch.enable_grad():
                leaves = views(theta, topology)
                weight, bias = leaves[0]
                physical = [(weight * alpha, bias - (weight * alpha) @ origin)] + list(leaves[1:])
                state = c36.curl_pressure(c36.multiply(gate, RAW[backend](physical, inputs) * scale))
                if name == 'interior':
                    numerator = base.independent.independent_interior(state, extra[0], extra[1], weights, base._data['nu'], base._data['lambda_g'])
                else:
                    components = slice(3, 4) if name == 'gauge' else slice(0, 3)
                    inverse = base._consumer.pressure_inverse_square if name == 'gauge' else base._consumer.velocity_inverse_square
                    difference = state[:, components, 0] - extra[0][:, components]
                    numerator = (weights * difference.square().sum(-1) * inverse).sum()
                gradient = torch.autograd.grad(numerator * factor, theta)[0]
            return numerator.detach(), gradient.detach()
        return run

    traced, calls = {}, []
    for index, name in enumerate(FAMILIES):
        family = base._data['families'][name]
        inputs, gate = base._geometry[name]
        for begin in range(0, len(inputs), chunk):
            sl = slice(begin, begin + chunk)
            extra = (family['force'][sl], family['force_gradient'][sl]) if name == 'interior' else (family['target'][sl],)
            args = (inputs[sl], gate[sl], family['weights'][sl]) + extra
            key = (name, args[0].shape[0])
            if key not in traced:
                traced[key] = make_fx(piece(name, index))(*args)
            calls.append((index, traced[key], args))
    coefficients, denominators = base._consumer.coefficients, base._consumer.denominators

    class Evaluation:
        owners = (objective, traced, calls)

        def __call__(self):
            gradient = torch.zeros_like(theta)
            numerators = [theta.new_zeros(()) for _ in FAMILIES]
            for index, function, args in calls:
                numerator, contribution = function(*args)
                numerators[index] = numerators[index] + numerator
                gradient = gradient + contribution
            numerators = torch.stack(numerators)
            return (numerators * coefficients / denominators).sum(), gradient, numerators

    return Evaluation()
