"""Compile the shared first-affine math on HOST and test independent derivatives.

Uses only the Python standard library and a C++17 compiler. This does not compile
CUDA launch code or validate GPU execution, synchronization, or libdevice.
"""

import argparse
import ctypes
import hashlib
import itertools
import json
import math
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WRAPPER = r'''
#include "coordinate_jet.cuh"
template<int D>
void run(bool full, bool fused, const double* x, const double* w, double b,
         const double* seed, double* h, double* aux, double* gradient) {
  constexpr int Q=D==2 ? 10 : 20;
  double z[Q], dz[Q];
  if (fused) flashns_coordinate::affine_activation<D>(x,w,b,full,h,aux);
  else {
    flashns_coordinate::affine_jet<D>(x,w,b,full,z);
    if (full) {
      if constexpr (D==2) flashns_stable::tanh_fwd_2d3(z,h,aux);
      else flashns_stable::tanh_fwd_3d3(z,h,aux);
    } else {
      const double r=::exp(-::fabs(z[0])), v=(2.0*r)/(1.0+r*r);
      h[0]=::tanh(z[0]); *aux=v*v;
    }
  }
  if (full) {
    if constexpr (D==2) flashns_stable::tanh_vjp_2d3(h,seed,*aux,dz);
    else flashns_stable::tanh_vjp_3d3(h,seed,*aux,dz);
  } else dz[0]=seed[0]*(*aux);
  gradient[D]=dz[0];
  for (int axis=0;axis<D;++axis)
    gradient[axis]=dz[0]*x[axis]+(full ? dz[D-axis] : 0.0);
}
extern "C" void first_host(int d, int full, int fused, const double* x, const double* w,
    double b, const double* seed, double* h, double* aux, double* gradient) {
  if (d==2) run<2>(full,fused,x,w,b,seed,h,aux,gradient);
  else run<3>(full,fused,x,w,b,seed,h,aux,gradient);
}
'''


def indices(dimension):
    return [a for degree in range(4) for a in itertools.product(range(degree+1), repeat=dimension)
            if sum(a) == degree]


def same_value(actual, expected):
    if math.isnan(expected):
        return math.isnan(actual)
    return struct.pack("d", actual) == struct.pack("d", expected)


def close(actual, expected, *, strict=False):
    tolerance = max(2e-11*abs(expected), 64*math.ulp(expected)) if strict and expected != 0 else 2e-12+2e-11*abs(expected)
    if not math.isfinite(actual) or abs(actual-expected) > tolerance:
        raise AssertionError(f"actual={actual!r}, reference={expected!r}, tolerance={tolerance!r}")
    return abs(actual-expected)/tolerance if tolerance else 0.0


def analytic(x, weight, bias, alpha):
    center = x[0]*weight[0]
    for axis in range(1, len(x)):
        center += x[axis]*weight[axis]
    center += bias
    t, r = math.tanh(center), math.exp(-abs(center))
    a1 = ((2*r)/(1+r*r))**2
    derivatives = (t, a1, -2*t*a1, (6*t*t-2)*a1, 8*t*(2-3*t*t)*a1)
    degree = sum(alpha)
    coefficient = math.prod(weight[axis]**power/math.factorial(power) for axis, power in enumerate(alpha))
    value, db = derivatives[degree]*coefficient, derivatives[degree+1]*coefficient
    dw = []
    for axis, power in enumerate(alpha):
        direct = 0.0
        if power:
            direct = derivatives[degree]*math.prod(
                weight[j]**(n-int(j==axis))/math.factorial(n-int(j==axis))
                for j, n in enumerate(alpha))
        dw.append(x[axis]*db+direct)
    return value, a1, dw+[db]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("a C++17 compiler is required")
    args.output.mkdir(parents=True, exist_ok=False)
    wrapper = args.output/"first_fused_host.cpp"
    wrapper.write_text(WRAPPER)
    library = args.output/"first_fused_host.so"
    command = [compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
               "-dynamiclib" if sys.platform == "darwin" else "-shared", "-fPIC", f"-I{HERE}",
               str(wrapper), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    (args.output/"build.log").write_text(result.stdout+result.stderr)
    report = {"passed": False, "completed": False, "scope": "HOST shared scalar core only; no GPU execution",
              "command": command, "compile_exit_code": result.returncode,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (HERE/"coordinate_jet.cuh", HERE.parent/"cuda_jet_h100/stable_jet.cuh",
                                          Path(__file__))}, "checks": []}
    output = args.output/"validation.json"
    output.write_text(json.dumps(report, indent=2)+"\n")
    result.check_returncode()
    function = ctypes.CDLL(str(library)).first_host
    ptr = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [ctypes.c_int]*3+[ptr, ptr, ctypes.c_double]+[ptr]*4
    function.restype = None

    def run(d, full, fused, x, w, b, seed):
        xa, wa, sa = [(ctypes.c_double*len(values))(*values) for values in (x, w, seed)]
        h, aux, gradient = (ctypes.c_double*20)(*[917.]*20), ctypes.c_double(), (ctypes.c_double*(d+1))()
        function(d, full, fused, xa, wa, b, sa, h, ctypes.byref(aux), gradient)
        q = len(indices(d)) if full else 1
        assert all(value == 917. for value in list(h)[q:]), "HOST core wrote beyond its declared jet"
        return list(h)[:q], aux.value, list(gradient)

    rng = random.Random(92642)
    worst, cases, seeds = 0.0, 0, 0
    for d, full, center in itertools.product((2, 3), (False, True), (-350., -20., -1., 0., 1., 20., 100., 350.)):
        x = [rng.uniform(-0.2, 0.2) for _ in range(d)]
        w = [rng.uniform(0.05, 0.5) for _ in range(d)]
        value = x[0]*w[0]
        for axis in range(1, d):
            value += x[axis]*w[axis]
        b = center-value
        basis = indices(d) if full else [(0,)*d]
        for active, alpha in enumerate(basis):
            seed = [float(j == active) for j in range(len(basis))]
            control, candidate = run(d, full, False, x, w, b, seed), run(d, full, True, x, w, b, seed)
            for observed, expected in zip((*candidate[0], candidate[1], *candidate[2]),
                                          (*control[0], control[1], *control[2])):
                assert same_value(observed, expected), "shared HOST control and fused core differ"
            for j, coefficient in enumerate(basis):
                h, aux, gradient = analytic(x, w, b, coefficient)
                worst = max(worst, close(candidate[0][j], h, strict=True), close(candidate[1], aux, strict=True))
                if j == active:
                    for axis, (actual, expected) in enumerate(zip(candidate[2], gradient)):
                        worst = max(worst, close(actual, expected, strict=axis==d))
            if abs(center) >= 20:
                assert candidate[1] > 0, "stable a1 was lost in a saturated tanh tail"
            seeds += 1
        cases += 1
    report["checks"].append({"name": "independent_analytic_jet_and_parameter_VJP_through_fourth_derivative",
                             "cases": cases, "basis_seeds": seeds, "max_error_over_tolerance": worst,
                             "matched_control_bitwise_equal": True})
    cases = 0
    for d, full, special in itertools.product((2, 3), (False, True), (float("inf"), -float("inf"), float("nan"))):
        for location in ("weight", "bias", "coordinate"):
            x, w, b = [1.]*d, [0.5]*d, 0.1
            if location == "weight": w[0] = special
            elif location == "bias": b = special
            else: x[0] = special
            seed = [1.]*(len(indices(d)) if full else 1)
            a, c = run(d, full, False, x, w, b, seed), run(d, full, True, x, w, b, seed)
            assert all(same_value(v, r) for v, r in zip((*c[0], c[1], *c[2]), (*a[0], a[1], *a[2])))
            cases += 1
    report["checks"].append({"name": "matched_nonfinite_coordinate_weight_bias_propagation", "cases": cases})
    report.update(passed=True, completed=True, library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
