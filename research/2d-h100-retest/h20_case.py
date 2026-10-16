"""Execute unchanged H20 science with frozen inputs and H100 provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=["adapter", "representation"], required=True)
    args, legacy = parser.parse_known_args()
    root = args.root.resolve()
    source = root / "source"
    sys.path[:0] = [str(root), str(source / "src"), str(source / "experiments/pinn_solver")]
    os.environ.update(FLASHNS_SOURCE=str(source), FLASHNS_BASELINES_ROOT=str(root))
    expect = json.loads((root / "H20-expectations.json").read_text())
    frozen = json.loads((root / "frozen-H20-inputs.json").read_text())
    assert sha(root / "frozen-H20-inputs.npz") == frozen["raw_sha256"]
    for name, expected in expect["original_source_manifest"].items():
        assert sha(root / name) == expected, name
    with np.load(root / "frozen-H20-inputs.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    import problem
    original_problem, original_parameters = problem.Problem, problem.parameters

    class FrozenProblem(original_problem):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            if self.interior_count == 2048:
                for name in ["training_points_sha256", "pde_weights_sha256", "boundary_weights_sha256",
                             "validation_points_sha256", "validation_boundary_sha256"]:
                    assert self.meta[name] == frozen["meta"][name], name
                device = self.x.device
                for name in ["x", "pde_weights", "boundary_weights", "target", "validation", "validation_boundary"]:
                    setattr(self, name, torch.from_numpy(arrays[name].copy()).to(device))

    def frozen_parameters(seed):
        key = "theta_" + str(seed)
        if key not in arrays:
            return original_parameters(seed)
        result = torch.from_numpy(arrays[key].copy())
        assert problem.digest(result) == frozen["theta_hashes"][str(seed)]
        return result

    problem.Problem, problem.parameters = FrozenProblem, frozen_parameters
    output = Path(legacy[legacy.index("--output") + 1])
    raw, checks = {}, []
    code = 1
    try:
        if args.kind == "adapter":
            import run_baseline as runner

            def validate(engine, data):
                from flashns.pinn_packed import PackedPINN
                n = data.interior_count
                reference = PackedPINN(data.x, n, data.pde_weights[:n], data.boundary_weights[n:], data.target[n:])
                saved = engine.flat.detach().clone()
                rows = []
                try:
                    for trial in range(3):
                        theta = engine.flat.detach().cpu().numpy().copy()
                        expected, gradients = reference(*problem.views(engine.flat))
                        gradient = torch.cat([g.flatten() for g in gradients])
                        actual, observed = engine.replay()
                        for kind, a, b in [("loss", actual, expected), ("gradient", observed, gradient)]:
                            a, b = a.detach().cpu().numpy().copy(), b.detach().cpu().numpy().copy()
                            name = "trial" + str(trial) + "_" + kind
                            raw[name + "_actual"], raw[name + "_reference"] = a, b
                            assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
                            assert a.size == (4547 if kind == "gradient" else 1)
                            assert np.isfinite(a).all() and np.isfinite(b).all()
                            error = float(np.max(np.abs(a - b) / (2e-12 + 2e-11 * np.abs(b))))
                            checks.append({"name": name, "maximum_error_over_original_budget": error})
                            assert error <= 1, (name, error)
                        raw["trial" + str(trial) + "_theta"] = theta
                        rows.append({"trial": trial, "loss_error": float((actual - expected).abs()),
                            "gradient_max_abs_error": float((observed - gradient).abs().max())})
                        with torch.no_grad():
                            engine.flat.add_(observed, alpha=-1e-4)
                finally:
                    with torch.no_grad():
                        engine.flat.copy_(saved)
                return rows

            runner.validate = validate
            if legacy[legacy.index("--backend") + 1] == "cueq_polynomial":
                # Same initialization-only portability repair used on H20.
                sys.path[:0] = [str(root / "snapshots/cueq/cuequivariance"),
                    str(root / "snapshots/cueq/cuequivariance_torch")]
                import backends
                from scientific_backends import CuEqNative
                class LibraryOnlyCuEq(CuEqNative):
                    def __init__(self):
                        self.forward_mode, self.method = "cueq", "uniform_1d"
                        self.polynomials, self.descriptors = {}, []
                        self.build = {"route": "cuEquivariance SegmentedPolynomial only",
                            "unused_flashns_native_library_initialization_omitted": True,
                            "coordinate_and_parameter_math_unchanged": True}
                backends.CuEqNative = LibraryOnlyCuEq
        else:
            sys.path.insert(0, str(source / "experiments/contracted_jets"))
            import bench_gpu as runner
            def evidence():
                return {"pid": os.getpid(), "executable": sys.executable, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                    "capability": torch.cuda.get_device_capability(0),
                    "git_base": expect["source_commit"], "git_base_scope": "Original frozen H20 source; no recreated Git history",
                    "sources": {name: digest for name, digest in expect["original_source_manifest"].items() if name.startswith("source/")}}
            runner.evidence = evidence
        sys.argv = [str(root / ("run_baseline.py" if args.kind == "adapter" else "source/experiments/contracted_jets/bench_gpu.py")), *legacy]
        result = runner.main()
        code = int(result or 0)
    finally:
        if raw:
            path = output.with_suffix(".proof.npz")
            np.savez_compressed(path, **raw)
            output.with_suffix(".proof.json").write_text(json.dumps({"state": "PASS" if code == 0 else "FAILED_OR_INCOMPLETE",
                "checks": checks, "raw_sha256": sha(path), "arrays": len(raw),
                "atol": 2e-12, "rtol": 2e-11, "full_parameters": 4547}, indent=2) + "\n")
        output.with_suffix(".provenance.json").write_text(json.dumps({"kind": args.kind,
            "source_commit": expect["source_commit"], "source_manifest_sha256": sha(root / "H20-expectations.json"),
            "frozen_inputs_sha256": sha(root / "frozen-H20-inputs.npz"),
            "theta_policy": "Stored original H20 parameter bytes for six formal seeds; separate declared development seed",
            "wrapper_sha256": sha(__file__), "scientific_source_modified": False,
            "GPU_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "reported_worker_exit": code}, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
