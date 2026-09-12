"""Current-stream bindings for the v8 packed activation and residual kernels."""

import ctypes
import hashlib
import json
from pathlib import Path

import torch


class NativeCUDA:
    def __init__(self, build_directory):
        directory = Path(build_directory)
        self.metadata = json.loads((directory / "build.json").read_text())
        if not self.metadata.get("compiled"):
            raise RuntimeError("a successful target-device CUDA build is required")
        library = directory / self.metadata["library"]
        if hashlib.sha256(library.read_bytes()).hexdigest() != self.metadata["library_sha256"]:
            raise RuntimeError("native library hash changed")
        root = Path(__file__).resolve().parents[2]
        for name, expected in self.metadata["source_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"native source changed since this build: {name}")
        if not torch.cuda.is_available():
            raise RuntimeError("v8 CUDA execution requires a CUDA device")
        self.library = ctypes.CDLL(str(library))
        self.activation = self.library.flashns_v8_activation
        self.activation.argtypes = [ctypes.c_int]*2 + [ctypes.c_void_p]*5 + [ctypes.c_int64]*2 + [ctypes.c_int, ctypes.c_void_p]
        self.activation.restype = ctypes.c_int
        self.loss_seed = self.library.flashns_v8_loss_seed
        self.loss_seed.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int64]*3 + [ctypes.c_double]*3 + [ctypes.c_void_p]
        self.loss_seed.restype = ctypes.c_int
        self.tail3 = self.library.flashns_v8_tail3
        self.tail3.argtypes = [ctypes.c_int]*2 + [ctypes.c_void_p]*5 + [ctypes.c_int64]*2 + [ctypes.c_int, ctypes.c_void_p]
        self.tail3.restype = ctypes.c_int
        self.coordinate_launch = self.library.flashns_v8_coordinate_affine
        self.coordinate_launch.argtypes = ([ctypes.c_int] + [ctypes.c_void_p]*4
                                           + [ctypes.c_int64]*2 + [ctypes.c_int, ctypes.c_void_p])
        self.coordinate_launch.restype = ctypes.c_int
        self.first_fused_launch = self.library.flashns_v8_coordinate_affine_activation
        self.first_fused_launch.argtypes = ([ctypes.c_int] + [ctypes.c_void_p]*5
                                            + [ctypes.c_int64]*2 + [ctypes.c_int, ctypes.c_void_p])
        self.first_fused_launch.restype = ctypes.c_int
        self.coordinate_wgrad_launch = self.library.flashns_v8_coordinate_wgrad
        self.coordinate_wgrad_launch.argtypes = ([ctypes.c_int] + [ctypes.c_void_p]*5
                                                + [ctypes.c_int64]*2 + [ctypes.c_int, ctypes.c_void_p])
        self.coordinate_wgrad_launch.restype = ctypes.c_int
        self.coordinate_activation_wgrad_launch = self.library.flashns_v8_coordinate_activation_wgrad
        self.coordinate_activation_wgrad_launch.argtypes = ([ctypes.c_int]+[ctypes.c_void_p]*7
                                                           + [ctypes.c_int64]*2+[ctypes.c_int, ctypes.c_void_p])
        self.coordinate_activation_wgrad_launch.restype = ctypes.c_int

    @staticmethod
    def check(tensors):
        device = tensors[0].device
        for tensor in tensors:
            if not tensor.is_cuda or tensor.dtype != torch.float64 or tensor.device != device:
                raise ValueError("same-device CUDA FP64 tensors required")

    @staticmethod
    def launch(function, arguments, tensors):
        with torch.cuda.device(tensors[0].device):
            stream = torch.cuda.current_stream()
            status = function(*arguments, stream.cuda_stream)
            for tensor in tensors:
                tensor.record_stream(stream)
        if status:
            raise RuntimeError(f"v8 CUDA launch returned {status}")

    def activation_factory(self, layout):
        owner = self

        class Activation:
            def forward(self, value):
                owner.check((value,))
                layout.check(value)
                output = torch.empty_like(value)
                aux = value.new_empty((layout.points, value.shape[1]))
                owner.launch(owner.activation, [layout.dimension, 0, value.data_ptr(), None, None,
                    output.data_ptr(), aux.data_ptr(), layout.full_points, layout.value_points, value.shape[1]],
                    (value, output, aux))
                return output, aux

            def vjp(self, hidden, aux, seed):
                owner.check((hidden, aux, seed))
                layout.check(hidden)
                layout.check(seed)
                if aux.shape != (layout.points, hidden.shape[1]) or not aux.is_contiguous() or seed.shape != hidden.shape:
                    raise ValueError("auxiliary/adjoint shape mismatch")
                output = torch.empty_like(hidden)
                owner.launch(owner.activation, [layout.dimension, 1, hidden.data_ptr(), aux.data_ptr(), seed.data_ptr(),
                    output.data_ptr(), None, layout.full_points, layout.value_points, hidden.shape[1]],
                    (hidden, aux, seed, output))
                return output

        return Activation()

    def coordinate_affine(self, layout, coordinates, weight, bias):
        """Generate the first affine jet for unencoded physical coordinates."""
        return self._coordinate_affine(layout, coordinates, weight, bias, activation=False)

    def coordinate_affine_activation(self, layout, coordinates, weight, bias):
        """Opt-in first affine + stable activation, returning packed H and a1."""
        return self._coordinate_affine(layout, coordinates, weight, bias, activation=True)

    def _coordinate_affine(self, layout, coordinates, weight, bias, *, activation):
        tensors = (coordinates, weight, bias)
        self.check(tensors)
        if not all(t.is_contiguous() for t in tensors):
            raise ValueError("contiguous coordinate/weight/bias tensors required")
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            raise NotImplementedError("manual parameter VJP; double backward/HVP unsupported")
        if coordinates.shape != (layout.points, layout.dimension) or weight.ndim != 2:
            raise ValueError("coordinates must match the physical-coordinate layout")
        channels = weight.shape[0]
        if channels < 1 or weight.shape[1] != layout.dimension or bias.shape != (channels,):
            raise ValueError("expected W[C,dimension], bias[C]")
        if channels > 2**31-1 or layout.full_points > 2**63-1 or layout.value_points > 2**63-1:
            raise ValueError("coordinate affine ABI dimension limit exceeded")
        output = coordinates.new_empty((layout.rows, channels))
        if activation:
            auxiliary = coordinates.new_empty((layout.points, channels))
            self.launch(self.first_fused_launch, [layout.dimension,
                        *[t.data_ptr() for t in tensors], output.data_ptr(), auxiliary.data_ptr(),
                        layout.full_points, layout.value_points, channels], (*tensors, output, auxiliary))
            return output, auxiliary
        self.launch(self.coordinate_launch, [layout.dimension,
                    *[t.data_ptr() for t in tensors], output.data_ptr(),
                    layout.full_points, layout.value_points, channels], (*tensors, output))
        return output

    def coordinate_wgrad(self, layout, coordinates, derivative):
        """G1: contract complete materialized bar_Z with physical input jets."""
        tensors = (coordinates, derivative)
        self.check(tensors)
        if not all(t.is_contiguous() for t in tensors):
            raise ValueError("contiguous coordinate/derivative tensors required")
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            raise NotImplementedError("manual parameter VJP; double backward/HVP unsupported")
        layout.check(derivative)
        channels = derivative.shape[1]
        if coordinates.shape != (layout.points, layout.dimension) or channels < 1:
            raise ValueError("expected coordinates[points,dimension], bar_Z[rows,C]")
        if channels > 2**31-1 or layout.full_points > 2**63-1 or layout.value_points > 2**63-1:
            raise ValueError("coordinate wgrad ABI dimension limit exceeded")
        partials = derivative.new_empty(((layout.points+7)//8, channels, layout.dimension+1))
        weight_gradient = derivative.new_empty((channels, layout.dimension))
        bias_gradient = derivative.new_empty((channels,))
        self.launch(self.coordinate_wgrad_launch,
                    [layout.dimension, derivative.data_ptr(), coordinates.data_ptr(), partials.data_ptr(),
                     weight_gradient.data_ptr(), bias_gradient.data_ptr(),
                     layout.full_points, layout.value_points, channels],
                    (*tensors, partials, weight_gradient, bias_gradient))
        return weight_gradient, bias_gradient

    def coordinate_activation_wgrad(self, layout, coordinates, hidden, aux, seed):
        """G2: complete stable activation VJP feeds the same coordinate reducer."""
        tensors = (coordinates, hidden, aux, seed)
        self.check(tensors)
        if not all(t.is_contiguous() for t in tensors):
            raise ValueError("contiguous coordinate/H/a1/bar_H tensors required")
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            raise NotImplementedError("manual parameter VJP; double backward/HVP unsupported")
        layout.check(hidden)
        layout.check(seed)
        channels = hidden.shape[1]
        if (coordinates.shape != (layout.points, layout.dimension) or channels < 1
                or seed.shape != hidden.shape or aux.shape != (layout.points, channels)):
            raise ValueError("expected x[points,dimension], H/bar_H[rows,C], a1[points,C]")
        if channels > 2**31-1 or layout.full_points > 2**63-1 or layout.value_points > 2**63-1:
            raise ValueError("coordinate activation-wgrad ABI dimension limit exceeded")
        partials = hidden.new_empty(((layout.points+7)//8, channels, layout.dimension+1))
        dw, db = hidden.new_empty((channels, layout.dimension)), hidden.new_empty((channels,))
        self.launch(self.coordinate_activation_wgrad_launch,
                    [layout.dimension, hidden.data_ptr(), aux.data_ptr(), seed.data_ptr(),
                     coordinates.data_ptr(), partials.data_ptr(), dw.data_ptr(), db.data_ptr(),
                     layout.full_points, layout.value_points, channels], (*tensors, partials, dw, db))
        return dw, db

    def tail_dgrad_vjp(self, layout, d, weight, hidden, aux, *, backend="F3"):
        """Opt-in Cout=3 packed dgrad/VJP with matched U3 and F3 arithmetic."""
        if backend not in ("U3", "F3"):
            raise ValueError("packed tail backend must be U3 or F3")
        tensors = (d, weight, hidden, aux)
        self.check(tensors)
        if not all(t.is_contiguous() for t in tensors):
            raise ValueError("contiguous packed tail tensors required")
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            raise NotImplementedError("manual parameter VJP; double backward/HVP unsupported")
        layout.check(d)
        layout.check(hidden)
        channels = hidden.shape[1]
        if d.shape != (layout.rows, 3) or weight.shape != (3, channels) or aux.shape != (layout.points, channels):
            raise ValueError("expected D[rows,3], W[3,C], H[rows,C], aux[points,C]")
        if channels > 2**31-1 or layout.full_points > 2**63-1 or layout.value_points > 2**63-1:
            raise ValueError("packed tail ABI dimension limit exceeded")
        output = torch.empty_like(hidden)
        self.launch(self.tail3, [layout.dimension, int(backend == "F3"),
                    *[t.data_ptr() for t in tensors], output.data_ptr(),
                    layout.full_points, layout.value_points, channels], (*tensors, output))
        if backend == "U3":
            return self.activation_factory(layout).vjp(hidden, aux, output)
        return output

    def seed(self, jets, boundary, pde_weights, boundary_weights, target, loss):
        self.check((jets, boundary, pde_weights, boundary_weights, target))
        ni, nb = len(jets), len(boundary)
        if jets.shape != (ni, 10, 3) or boundary.shape != (nb, 3) or boundary.stride(-1) != 1:
            raise ValueError("2-D Q10 interior and contiguous-channel boundary fields required")
        if nb > 1 and boundary.stride(0) < 3:
            raise ValueError("boundary rows must not overlap or broadcast")
        if pde_weights.shape != (ni,) or boundary_weights.shape != (nb, 3) or target.shape != (nb, 3):
            raise ValueError("objective shape mismatch")
        if not all(t.is_contiguous() for t in (jets, pde_weights, boundary_weights, target)):
            raise ValueError("contiguous jet/weight/target storage required")
        di, db = torch.empty_like(jets), boundary.new_empty((nb, 3))
        partial = jets.new_empty(((ni+nb+127)//128,))
        tensors = (jets, boundary, pde_weights, boundary_weights, target, di, db, partial)
        self.launch(self.loss_seed, [t.data_ptr() for t in tensors] +
                    [ni, nb, max(3, boundary.stride(0)), loss.viscosity, loss.gradient_weight, loss.boundary_weight], tensors)
        return partial.sum(), di, db
