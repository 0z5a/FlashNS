"""Check complete loss, all parameter gradients, and actual Adam state updates."""

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from problem import Problem, parameters, snapshot_sources, symbolic_check
from backends import Adam, BACKENDS, Engine, nested_step
from common import comparison
from scientific_backends import versions
from solve import protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hopper-build", type=Path, required=True)
    parser.add_argument("--backends", default=",".join(BACKENDS))
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite preflight")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    configuration = protocol()
    problem = Problem(
        **{
            key: configuration[key]
            for key in ("interior_count", "edge_count", "validation_count", "data_seed")
        }
    )
    initial = parameters(361901)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": snapshot_sources(args.output),
        "problem": problem.meta,
        "versions": versions(),
        "symbolic": symbolic_check(),
        "backends": {},
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for backend in args.backends.split(","):
        print({"validating": backend}, flush=True)
        engine = Engine(problem, initial, backend, args.hopper_build)
        adam = Adam(engine.flat, engine.gradient)
        reference = initial.numpy().copy()
        first_m, second_m = np.zeros_like(reference), np.zeros_like(reference)
        rows = []
        for iteration in range(1, 4):
            expected_loss, expected_gradient = nested_step(
                problem, torch.from_numpy(reference).to(problem.x.device)
            )
            actual_loss, actual_gradient = engine.replay()
            row = {
                "iteration": iteration,
                "loss": comparison(actual_loss, expected_loss),
                "full_parameter_gradient": comparison(
                    actual_gradient, expected_gradient
                ),
            }
            gradient = expected_gradient.cpu().numpy()
            first_m = 0.9 * first_m + 0.1 * gradient
            second_m = 0.999 * second_m + 0.001 * gradient * gradient
            reference -= (
                0.001
                * (first_m / (1 - 0.9**iteration))
                / (np.sqrt(second_m / (1 - 0.999**iteration)) + 1e-8)
            )
            adam.step()
            row["parameter_update_vs_numpy"] = comparison(
                engine.flat.detach(), torch.from_numpy(reference).to(problem.x.device)
            )
            rows.append(row)
        report["backends"][backend] = {
            "passed": True,
            "checks": rows,
            "setup": engine.finish_metadata(),
        }
        save()
        print({"passed": backend}, flush=True)
        del adam, engine
        gc.collect()
        torch.cuda.empty_cache()
    report["passed"] = True
    report["finished_utc"] = datetime.now(UTC).isoformat()
    save()


if __name__ == "__main__":
    main()
