"""Complete optimization to one fixed, independent acceptance rule."""

import argparse
import gc
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch

from problem import (
    Problem,
    digest,
    evaluate,
    independent_metrics,
    parameters,
    snapshot_sources,
    symbolic_check,
)
from backends import Adam, BACKENDS, Engine
from gpu import Native

THRESHOLDS = {
    "pde_rms": 0.02,
    "boundary_uv_rms": 0.01,
    "outflow_p_rms": 0.01,
    "relative_l2_u": 0.01,
    "relative_l2_v": 0.02,
    "relative_l2_p": 0.02,
}


def protocol():
    return {
        "schema_version": 1,
        "problem": "Kovasznay steady Navier-Stokes, viscosity 0.07",
        "interior_count": 2048,
        "edge_count": 128,
        "validation_count": 8192,
        "data_seed": 642701,
        "thresholds": THRESHOLDS,
        "acceptance": "all six metrics pass on fixed jet validation and an additional independent scalar-AD point set; finite values required",
        "adam_steps": 4000,
        "adam_learning_rate": 0.001,
        "adam_decay_every": 1000,
        "adam_decay_factor": 0.5,
        "adam_check_every": 100,
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1e-8,
        "lbfgs_max_blocks": 200,
        "lbfgs_max_iter_per_block": 20,
        "lbfgs_max_eval_per_block": 25,
        "lbfgs_history_size": 50,
        "lbfgs_line_search": "strong_wolfe",
        "lbfgs_lr": 1.0,
        "lbfgs_tolerance_grad": 1e-12,
        "lbfgs_tolerance_change": 1e-15,
        "stop_check": "before training, every 100 Adam updates, then after each block of at most 20 L-BFGS iterations",
        "formal_initialization_seeds": [75101, 75102, 75103],
        "pilot_initialization_seed": 361901,
        "cache_policy": "mandatory preflight populates dependency/compiler caches; each solve gets new parameters, optimizer state and CUDA graphs; setup is reported and included in total",
        "precision": "FP64 throughout; TF32 disabled",
        "timing": "optimization wall includes loss/gradient, optimizer, line searches, synchronization, all stop checks and final independent acceptance; per-run setup and common data preparation are separately reported and included in total",
    }


def accepted(metrics, thresholds):
    return all(
        math.isfinite(metrics[name]) and metrics[name] <= threshold
        for name, threshold in thresholds.items()
    )


def solve(
    problem,
    initial,
    backend,
    configuration,
    hopper_build,
    *,
    emit=None,
    hopper_configuration=None,
):
    started = perf_counter()
    engine = Engine(problem, initial, backend, hopper_build, hopper_configuration)
    evaluator = Native()
    adam = Adam(engine.flat, engine.gradient, configuration["adam_learning_rate"])
    torch.cuda.synchronize()
    setup_seconds = perf_counter() - started
    rows = []
    optimization_start = perf_counter()
    last_check = optimization_start
    converged = False
    adam_updates = 0
    lbfgs_iterations = 0
    independent = None

    def check(phase):
        nonlocal converged, independent, last_check
        metrics = evaluate(problem, engine.flat, evaluator)
        convergence_candidate = accepted(metrics, configuration["thresholds"])
        if convergence_candidate:
            independent = independent_metrics(problem, engine.flat)
            converged = accepted(independent, configuration["thresholds"])
        row = {
            "phase": phase,
            "adam_updates": adam_updates,
            "lbfgs_iterations": lbfgs_iterations,
            "loss_gradient_evaluations": engine.closure_calls,
            "optimization_wall_seconds": perf_counter() - optimization_start,
            "metrics": metrics,
            "independent_metrics": independent if convergence_candidate else None,
            "converged": converged,
        }
        rows.append(row)
        if emit:
            emit(row)
        if perf_counter() - last_check >= 30 or converged or phase == "initial":
            print(
                {
                    "backend": backend,
                    "phase": phase,
                    "adam": adam_updates,
                    "lbfgs": lbfgs_iterations,
                    "metrics": metrics,
                    "converged": converged,
                },
                flush=True,
            )
            last_check = perf_counter()
        return converged

    check("initial")
    for update in range(1, configuration["adam_steps"] + 1):
        if converged:
            break
        if (update - 1) % configuration["adam_decay_every"] == 0:
            learning_rate = configuration["adam_learning_rate"] * configuration[
                "adam_decay_factor"
            ] ** ((update - 1) // configuration["adam_decay_every"])
            adam.lr.fill_(learning_rate)
        engine.replay()
        adam.step()
        adam_updates += 1
        if update % configuration["adam_check_every"] == 0:
            check("adam")
    optimizer = torch.optim.LBFGS(
        [engine.flat],
        lr=configuration["lbfgs_lr"],
        max_iter=configuration["lbfgs_max_iter_per_block"],
        max_eval=configuration["lbfgs_max_eval_per_block"],
        tolerance_grad=configuration["lbfgs_tolerance_grad"],
        tolerance_change=configuration["lbfgs_tolerance_change"],
        history_size=configuration["lbfgs_history_size"],
        line_search_fn=configuration["lbfgs_line_search"],
    )
    for block in range(configuration["lbfgs_max_blocks"]):
        if converged:
            break
        optimizer.step(engine.closure)
        lbfgs_iterations = optimizer.state[engine.flat].get("n_iter", 0)
        check("lbfgs")
        if not all(math.isfinite(value) for value in rows[-1]["metrics"].values()):
            break
    torch.cuda.synchronize()
    optimization_seconds = perf_counter() - optimization_start
    state = engine.flat.detach().cpu().clone()
    output = {
        "backend": backend,
        "converged": converged,
        "setup_seconds": setup_seconds,
        "optimization_wall_seconds": optimization_seconds,
        "setup_plus_optimization_seconds": setup_seconds + optimization_seconds,
        "adam_updates": adam_updates,
        "lbfgs_iterations": lbfgs_iterations,
        "loss_gradient_evaluations": engine.closure_calls,
        "metrics": rows[-1]["metrics"],
        "independent_metrics": independent,
        "parameter_initial_sha256": digest(initial),
        "parameter_final_sha256": digest(state),
        "backend_setup": engine.finish_metadata(),
        "history": rows,
    }
    del optimizer, adam
    engine = evaluator = None
    gc.collect()
    torch.cuda.empty_cache()
    return output, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=BACKENDS, default="B1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--hopper-build", type=Path)
    parser.add_argument("--hopper-configuration")
    parser.add_argument("--seed", type=int, default=361901)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or (not args.pilot and args.protocol is None):
        raise RuntimeError("require a new output and a frozen protocol for formal runs")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    configuration = (
        json.loads(args.protocol.read_text()) if args.protocol else protocol()
    )
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "pilot": args.pilot,
        "seed": args.seed,
        "protocol": configuration,
        "source_hashes": snapshot_sources(args.output),
        "symbolic_exact_solution_check": symbolic_check(),
        "completed": False,
        "progress": [],
    }
    preparation = perf_counter()
    problem = Problem(
        **{
            key: configuration[key]
            for key in ("interior_count", "edge_count", "validation_count", "data_seed")
        }
    )
    initial = parameters(args.seed)
    torch.cuda.synchronize()
    report["common_preparation_seconds"] = perf_counter() - preparation
    report["problem"] = problem.meta
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def emit(row):
        report["progress"].append(row)
        save()

    save()
    result, state = solve(
        problem,
        initial,
        args.backend,
        configuration,
        args.hopper_build,
        emit=emit,
        hopper_configuration=args.hopper_configuration,
    )
    report["result"] = result
    report["result"]["total_wall_seconds"] = (
        report["common_preparation_seconds"] + result["setup_plus_optimization_seconds"]
    )
    checkpoint = args.output.with_suffix(".pt")
    torch.save(
        {
            "flat_parameters": state,
            "problem": problem.meta,
            "protocol": configuration,
            "seed": args.seed,
            "backend": args.backend,
        },
        checkpoint,
    )
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report["completed"] = True
    report["finished_utc"] = datetime.now(UTC).isoformat()
    save()
    print(
        {
            key: result[key]
            for key in (
                "backend",
                "converged",
                "adam_updates",
                "lbfgs_iterations",
                "loss_gradient_evaluations",
                "optimization_wall_seconds",
                "total_wall_seconds",
                "metrics",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
