"""Small CUDA compiler qualification before the full canonical experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from torch_c36_packed import activate_packed
from torch_c36_packed_vjp import activate_packed_vjp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.device_count() == 1
    torch.manual_seed(795913)
    state = .25 * torch.randn(7, 4, 36, device="cuda")
    state[..., 0] = torch.tensor([-40., -20., -1e-12, 0., 1e-12, 20., 40.], device="cuda")[:, None]
    cotangent = 1e8 * torch.randn_like(state)
    expected_forward = activate_packed(state).detach()
    differentiable = state.detach().requires_grad_(True)
    expected_vjp, = torch.autograd.grad(activate_packed(differentiable), differentiable, cotangent)
    forward = torch.compile(activate_packed, fullgraph=True, dynamic=False)
    transpose = torch.compile(activate_packed_vjp, fullgraph=True, dynamic=False)
    actual_forward, actual_vjp = forward(state).detach(), transpose(state, cotangent).detach()
    arrays, checks = {}, []
    for name, actual, expected in [("forward", actual_forward, expected_forward),
                                  ("vjp", actual_vjp, expected_vjp)]:
        a, b = actual.cpu().numpy(), expected.cpu().numpy()
        assert np.isfinite(a).all() and np.isfinite(b).all()
        arrays[name + "_actual"], arrays[name + "_reference"] = a, b
        error = float(np.max(np.abs(a - b) / (1e-11 + 1e-9 * np.abs(b))))
        checks.append({"name": name, "maximum_error_over_original_budget": error})
    raw = args.output / "raw.npz"
    np.savez_compressed(raw, **arrays)
    report = {"state": "PASS" if all(x["maximum_error_over_original_budget"] <= 1 for x in checks) else "FAIL",
              "torch": torch.__version__, "points": 7, "width": 4, "channels": 36,
              "checks": checks, "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
              "atol": 1e-11, "rtol": 1e-9, "scope": "Small compiler/activation check; no full objective or performance claim"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["state"] == "PASS", report
    print(json.dumps(report))


if __name__ == "__main__":
    main()
