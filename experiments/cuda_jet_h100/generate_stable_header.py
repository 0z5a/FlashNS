"""Adapt the preserved full-jet generator to explicit stable a1 state."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cuda_jet/input"))
from generate_jet_header import make


def generate():
    text = make()
    text = text.replace("namespace flashns", "namespace flashns_stable")
    text = text.replace(
        "// Host primitives verified. CUDA backend NOT compiled/tested here.",
        "// Stable auxiliary-state revision 2; validation scope lives in target reports.",
    )
    text = text.replace(
        "// saturation-tail robustness requires separate high-precision tests.",
        "// Explicit a1 checkpoint preserves derivatives lost by rounded H[0].",
    )
    assert text.count("const double a1 = 1.0 - t*t;") == 2
    text = text.replace(
        "const double a1 = 1.0 - t*t;",
        "const double r = ::exp(-::fabs(z[0]));\n  const double v = (2.0*r)/(1.0+r*r);\n  const double a1 = v*v;\n  *aux = a1;",
    )
    text = text.replace(
        "const double* z, double* h)", "const double* z, double* h, double* aux)"
    )
    text = text.replace(
        "const double* h, const double* bh, double* bz)",
        "const double* h, const double* bh, double a1, double* bz)",
    )
    assert text.count("const double g0 = 1.0 - h[0]*h[0];") == 2
    text = text.replace("const double g0 = 1.0 - h[0]*h[0];", "const double g0 = a1;")
    return text


if __name__ == "__main__":
    path = HERE / "stable_jet.cuh"
    path.write_text(generate())
    print(path)
