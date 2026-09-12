"""Dense, split and compact MLP schedules for one identical FP64 objective.

Data and global weights are validated/prepared once. Parameter values are read on
each evaluation; CUDA Graph users must keep the parameter storage address fixed.
"""

from contextlib import nullcontext

import torch

from flashns.jet_packed import PackedLayout, TensorActivation
from flashns.ns_seed import NSLoss, compile_seed, explicit_seed, per_point_loss, validate_weights


class PackedPINN:
    def __init__(self, x, interior_count, pde_weights, boundary_weights, target, *,
                 layout="compact", seed="explicit", activation_factory=TensorActivation,
                 cuda_seed=None, residual_reference=None, loss=NSLoss(), compiled_options=None,
                 trace=False, tail_dgrad_vjp=None, coordinate_affine=None,
                 first_affine_activation=None, coordinate_wgrad=None, first_activation_wgrad=None):
        if x.ndim != 2 or x.shape[1] != 2 or x.dtype != torch.float64:
            raise ValueError("the fixed NS residual requires FP64 2-D coordinates")
        if type(interior_count) is not int or not 0 <= interior_count <= len(x):
            raise ValueError("invalid interior split")
        if layout not in ("dense", "split", "compact") or seed not in ("autograd", "explicit", "compiled", "cuda"):
            raise ValueError("unsupported layout or seed mode")
        self.ni, self.nb = interior_count, len(x) - interior_count
        validate_weights(pde_weights, (self.ni,), device=x.device)
        validate_weights(boundary_weights, (self.nb, 3), device=x.device)
        if target.shape != (self.nb, 3) or target.dtype != torch.float64 or target.device != x.device:
            raise ValueError("boundary target shape/dtype/device mismatch")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(target).all()):
            raise ValueError("finite coordinates and targets required")
        self.pde_weights = pde_weights.detach().contiguous()
        self.boundary_weights = boundary_weights.detach().contiguous()
        self.target = target.detach().contiguous()
        self.mode, self.seed_mode, self.loss = layout, seed, loss
        self.residual_reference = residual_reference or (lambda j: per_point_loss(j, loss))
        self.trace = trace
        if tail_dgrad_vjp is not None and not callable(tail_dgrad_vjp):
            raise ValueError("tail_dgrad_vjp must be callable or None")
        self.tail_dgrad_vjp = tail_dgrad_vjp
        if coordinate_affine is not None and not callable(coordinate_affine):
            raise ValueError("coordinate_affine must be callable or None")
        self.coordinate_affine = coordinate_affine
        if first_affine_activation is not None and not callable(first_affine_activation):
            raise ValueError("first_affine_activation must be callable or None")
        self.first_affine_activation = first_affine_activation
        if coordinate_wgrad is not None and not callable(coordinate_wgrad):
            raise ValueError("coordinate_wgrad must be callable or None")
        self.coordinate_wgrad = coordinate_wgrad
        if first_activation_wgrad is not None and not callable(first_activation_wgrad):
            raise ValueError("first_activation_wgrad must be callable or None")
        self.first_activation_wgrad = first_activation_wgrad
        if layout == "dense":
            self.layouts = [PackedLayout(len(x), 0)]
            coordinates = [x]
        elif layout == "compact":
            self.layouts = [PackedLayout(self.ni, self.nb)]
            coordinates = [x]
        else:
            self.layouts = [PackedLayout(self.ni, 0), PackedLayout(0, self.nb)]
            coordinates = [x[:self.ni], x[self.ni:]]
        self.inputs = [spec.coordinates(points).detach() for spec, points in zip(self.layouts, coordinates)]
        self.coordinate_inputs = ([points.detach().clone(memory_format=torch.contiguous_format)
                                   for points in coordinates] if
                                  coordinate_affine is not None or first_affine_activation is not None
                                  or coordinate_wgrad is not None or first_activation_wgrad is not None else None)
        self.zero_rows = [spec.zero_rows(x.device) for spec in self.layouts]
        self.ops = [activation_factory(spec) for spec in self.layouts]
        self.seed_function = (compile_seed(loss, **(compiled_options or {})) if seed == "compiled"
                              else lambda j, w: explicit_seed(j, w, loss, validate=False))
        self.cuda_seed = cuda_seed
        if seed == "cuda" and cuda_seed is None:
            raise ValueError("CUDA seed implementation required")
        self.metadata = {
            "layout": layout, "seed": seed, "full_order": 3, "dimension": 2,
            "interior_count": self.ni, "boundary_count": self.nb,
            "segment_rows": [spec.rows for spec in self.layouts],
            "packed_rows_total": sum(spec.rows for spec in self.layouts),
            "precision": "FP64 stable_aux_a1", "weights": "already globally normalized",
            "supports_hvp_or_double_backward": False,
            "tail_dgrad_vjp_enabled": tail_dgrad_vjp is not None,
            "coordinate_affine_enabled": coordinate_affine is not None,
            "first_affine_activation_enabled": first_affine_activation is not None,
            "coordinate_wgrad_enabled": coordinate_wgrad is not None,
            "first_activation_wgrad_enabled": first_activation_wgrad is not None,
        }

    def region(self, name):
        return torch.profiler.record_function("flashns_v8::" + name) if self.trace else nullcontext()

    def __call__(self, weights, biases):
        if not weights or len(weights) != len(biases) or weights[-1].shape[0] != 3:
            raise ValueError("matching affine layers with three final outputs required")
        cin = 2
        for weight, bias in zip(weights, biases):
            if weight.ndim != 2 or weight.shape[1] != cin or bias.shape != (weight.shape[0],):
                raise ValueError("incompatible affine layer")
            if any(t.dtype != torch.float64 or t.device != self.inputs[0].device for t in (weight, bias)):
                raise ValueError("all parameters must be FP64 on the data device")
            cin = weight.shape[0]
        checkpoints, auxiliaries = [], []
        with torch.no_grad():
            for segment, (initial, zero, ops) in enumerate(zip(self.inputs, self.zero_rows, self.ops)):
                h, saved, aux = initial, [initial], []
                for layer, (weight, bias) in enumerate(zip(weights, biases)):
                    if layer == 0 and layer + 1 < len(weights) and self.first_affine_activation is not None:
                        with self.region(f"first_affine_activation/{segment}"):
                            h, a1 = self.first_affine_activation(self.layouts[segment],
                                self.coordinate_inputs[segment], weight, bias)
                        aux.append(a1)
                        saved.append(h)
                        continue
                    with self.region(f"forward_affine/{segment}/{layer}"):
                        if layer == 0 and self.coordinate_affine is not None:
                            z = self.coordinate_affine(self.layouts[segment],
                                                       self.coordinate_inputs[segment], weight, bias)
                        else:
                            z = h @ weight.T
                            z[zero] += bias
                    if layer + 1 < len(weights):
                        with self.region(f"activation/{segment}/{layer}"):
                            h, a1 = ops.forward(z)
                        aux.append(a1)
                    else:
                        h = z
                    saved.append(h)
                checkpoints.append(saved)
                auxiliaries.append(aux)
            interior = checkpoints[0][-1][:self.ni * 10].view(self.ni, 10, 3)
            bsegment = 1 if self.mode == "split" else 0
            bzero = self.zero_rows[bsegment][self.ni:] if self.mode != "split" else self.zero_rows[1]
            boundary = checkpoints[bsegment][-1][bzero]

        with self.region("residual_loss_seed"):
            if self.seed_mode == "autograd":
                with torch.enable_grad():
                    ji = interior.detach().requires_grad_()
                    bv = boundary.detach().requires_grad_()
                    total = ((self.pde_weights * self.residual_reference(ji)).sum()
                             + self.loss.boundary_weight * (self.boundary_weights * (bv-self.target).square()).sum())
                    di, db = torch.autograd.grad(total, (ji, bv))
                total = total.detach()
            elif self.seed_mode == "cuda":
                total, di, db = self.cuda_seed(interior, boundary, self.pde_weights,
                                               self.boundary_weights, self.target, self.loss)
            else:
                with torch.no_grad():
                    lp, di = self.seed_function(interior, self.pde_weights)
                    error = boundary - self.target
                    total = lp + self.loss.boundary_weight * (self.boundary_weights * error.square()).sum()
                    db = (2 * self.loss.boundary_weight) * self.boundary_weights * error

        with torch.no_grad():
            seeds = [torch.zeros_like(saved[-1]) for saved in checkpoints]
            seeds[0][:self.ni * 10] = di.reshape(self.ni * 10, 3)
            seeds[bsegment][bzero] += db
            dws, dbs = [None] * len(weights), [None] * len(weights)
            for segment, (saved, auxiliary, zero, ops, d) in enumerate(zip(checkpoints, auxiliaries, self.zero_rows, self.ops, seeds)):
                for layer in reversed(range(len(weights))):
                    with self.region(f"wgrad_bias/{segment}/{layer}"):
                        if layer == 0 and self.coordinate_wgrad is not None:
                            dw, bias_gradient = self.coordinate_wgrad(
                                self.layouts[segment], self.coordinate_inputs[segment], d)
                        else:
                            dw = d.T @ saved[layer]
                            bias_gradient = d[zero].sum(0)
                        dws[layer] = dw if segment == 0 else dws[layer] + dw
                        dbs[layer] = bias_gradient if segment == 0 else dbs[layer] + bias_gradient
                    if layer:
                        if layer == 1 and self.first_activation_wgrad is not None:
                            with self.region(f"dgrad/{segment}/{layer}"):
                                dh = d @ weights[layer]
                            with self.region(f"first_activation_wgrad/{segment}"):
                                dw0, db0 = self.first_activation_wgrad(self.layouts[segment],
                                    self.coordinate_inputs[segment], saved[layer], auxiliary[layer-1], dh)
                            dws[0] = dw0 if segment == 0 else dws[0] + dw0
                            dbs[0] = db0 if segment == 0 else dbs[0] + db0
                            break
                        if (self.tail_dgrad_vjp is not None and layer == len(weights)-1
                                and weights[layer].shape[0] == 3):
                            with self.region(f"tail_dgrad_vjp/{segment}/{layer}"):
                                d = self.tail_dgrad_vjp(self.layouts[segment], d, weights[layer],
                                                       saved[layer], auxiliary[layer-1])
                        else:
                            with self.region(f"dgrad/{segment}/{layer}"):
                                dh = d @ weights[layer]
                            with self.region(f"activation_vjp/{segment}/{layer}"):
                                d = ops.vjp(saved[layer], auxiliary[layer-1], dh)
        return total, dws + dbs
