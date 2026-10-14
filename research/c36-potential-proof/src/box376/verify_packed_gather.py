"""Qualify packed gathers against the preserved complete native H100 reference."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from box376.data import FAMILIES, flatten, initialize, views
from box376.potential_ad_objective import PreparedPotentialADObjective
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.potential_run import capture, cpu, load_problem, sha, timed_call, write_json
from box376.verify_potential_gradient import compare

REFERENCE_SHA256 = "172174ea49a82a21e0616571ae303560d275317d8b5319c22ea49dbd628d8b66"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "protocol", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert sha(args.reference) == REFERENCE_SHA256
    with np.load(args.reference, allow_pickle=False) as saved:
        reference = {name: saved[name].copy() for name in saved.files}
    protocol = json.loads(args.protocol.read_text())
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.device_count() == 1
    data, calibration = load_problem(args, protocol)
    for name in FAMILIES:
        for key in ("points", "weights", "target") + (("force", "force_gradient") if name == "interior" else ()):
            assert np.array_equal(cpu(data["families"][name][key]), reference["canonical_" + name + "_" + key])
    explicit = PreparedPotentialNormalizedObjective(data, calibration)
    ad = PreparedPotentialADObjective(data, calibration)
    challenge = copy.deepcopy(data)
    for index, name in enumerate(FAMILIES):
        family = challenge["families"][name]
        for key in ("points", "weights", "target") + (("force", "force_gradient") if name == "interior" else ()):
            family[key] = torch.from_numpy(reference["diagnostic_" + name + "_" + key]).to("cuda")
        family["W"] = float(reference["diagnostic_denominators"][index])
    diagnostic = PreparedPotentialNormalizedObjective(challenge, calibration)
    arrays, checks, timings = {}, [], []
    record = {"state": "RUNNING", "reference_sha256": REFERENCE_SHA256, "protocol_sha256": sha(args.protocol),
              "points": 4999, "parameters_per_seed": 50692, "seeds": protocol["seeds"],
              "atol": 1e-11, "rtol": 1e-9, "tolerance_relaxed": False, "checks": checks,
              "diagnostic_timings": timings, "training_converged": False, "physical_acceptance_evaluated": False}

    def check(prefix, actual, expected):
        for name, a, b in zip(("loss", "gradient", "numerators"), actual, expected):
            a, b = cpu(a), np.asarray(b)
            checks.append({"name": prefix + "_" + name, **compare(a, b)})
            arrays[prefix + "_actual_" + name], arrays[prefix + "_reference_" + name] = a, b.copy()
        record["phase"] = prefix
        write_json(args.output / "progress.json", record)
        print(json.dumps({"phase": prefix, "checks": len(checks), "unix": time.time()}), flush=True)

    for seed in protocol["seeds"]:
        theta = flatten(initialize(tuple(protocol["topology"]), seed)).to("cuda")
        assert np.array_equal(cpu(theta), reference[f"seed{seed}_theta"])
        parameters = views(theta, tuple(protocol["topology"]))
        original = theta.clone()
        native = tuple(reference[f"seed{seed}_subset_native_" + name] for name in ("loss", "gradient", "numerators"))
        for chunk in (128, 64):
            check(f"seed{seed}-explicit{chunk}-native", explicit(parameters, chunk), native)
        check(f"seed{seed}-parameterAD-native", ad(parameters, 128), native)
        expected = tuple(reference[f"seed{seed}_diagnostic_native_" + name] for name in ("loss", "gradient", "numerators"))
        check(f"seed{seed}-nonzero-value-families", diagnostic(parameters, 7), expected)
        for name, engine in (("explicit", explicit), ("parameterAD", ad)):
            replay, seconds = capture(lambda: engine(parameters, 128))
            record.setdefault("capture_seconds", {})[f"seed{seed}-{name}"] = seconds
            check(f"seed{seed}-{name}-graph-native", replay(), native)
            theta.add_(.001)
            check(f"seed{seed}-{name}-graph-changed", replay(), tuple(cpu(v) for v in ad(parameters, 128)))
            theta.copy_(original)
            for mode, function in (("eager", lambda: engine(parameters, 128)), ("graph", replay)):
                durations = [timed_call(function)[1] for _ in range(3)]
                timings.append({"seed": seed, "engine": name, "mode": mode, "evaluation_seconds": durations,
                                "scope": "diagnostic only; three samples, no paired performance claim"})
    np.savez_compressed(args.output / "raw-proof.npz", **arrays)
    record.update(state="PASS", raw_sha256=sha(args.output / "raw-proof.npz"),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(args.output / "proof.json", record)
    print(json.dumps({"state": "PASS", "checks": len(checks)}))


if __name__ == "__main__":
    main()
