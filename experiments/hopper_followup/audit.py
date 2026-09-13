"""Audit local source/binary evidence and every solver acceptance/checkpoint."""

import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/hopper_followup/audit.json"
ARTIFACTS = [
    ROOT / "artifacts/openai_ns_formal",
    ROOT / "artifacts/hopper_followup",
    *[
        ROOT / "experiments" / folder / "artifacts"
        for folder in ("cuda_jet_hopper", "openai_ns_formal", "pinn_solver")
    ],
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources(value, at=""):
    if isinstance(value, dict):
        for key, child in value.items():
            location = at + "/" + key
            if key in (
                "source_hashes",
                "source_hashes_before",
                "source_hashes_after",
                "numerical_validation_source_hashes",
            ) and isinstance(child, dict):
                for name, digest in child.items():
                    if isinstance(digest, str) and len(digest) == 64:
                        yield location + "/" + name, digest
            elif key in (
                "script_sha256",
                "suite_source_sha256",
                "extension_source_sha256",
                "binary_sha256",
                "executable_sha256",
            ) and isinstance(child, str):
                yield location, child
            else:
                yield from sources(child, location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from sources(child, at + f"/{index}")


def local_path(value):
    marker = "/root/flashns-hopper-20260909/"
    return (
        ROOT / value.split(marker, 1)[-1] if value.startswith(marker) else Path(value)
    )


def main():
    import torch

    corpus = defaultdict(list)
    roots = [
        *ARTIFACTS,
        ROOT / "sources/openai_ns_formal",
        ROOT / "src",
        *[
            ROOT / "experiments" / folder
            for folder in (
                "cuda_jet_hopper",
                "openai_ns_formal",
                "pinn_solver",
                "hopper_followup",
            )
        ],
        ROOT / "experiments/cuda_jet/input",
    ]
    paths = set()
    paths.update(
        path
        for path in (ROOT / "experiments/cuda_jet_h100").glob("*")
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".txt")
    )
    for directory in roots:
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and not any(
                    part in (".git", "__pycache__", ".venv") for part in path.parts
                )
                and path != OUTPUT
                and not path.name.endswith((".tar.gz", ".zip"))
            ):
                paths.add(path)
    for path in sorted(paths):
        corpus[sha(path)].append(str(path.relative_to(ROOT)))
    records, unresolved = [], []
    for directory in ARTIFACTS:
        for path in sorted(directory.rglob("*.json")):
            if (
                path == OUTPUT
                or "executed_sources" in path.parts
                or path.name == "manifest.json"
            ):
                continue
            value = json.loads(path.read_text())
            entries = []
            for field, digest in sources(value):
                item = {
                    "field": field,
                    "sha256": digest,
                    "resolved": corpus.get(digest, []),
                }
                entries.append(item)
                if not item["resolved"]:
                    unresolved.append({"report": str(path.relative_to(ROOT)), **item})
            if entries:
                records.append(
                    {
                        "report": str(path.relative_to(ROOT)),
                        "sha256": sha(path),
                        "references": entries,
                    }
                )

    formal_path = ROOT / "experiments/pinn_solver/artifacts/formal1/suite.json"
    extended_path = ROOT / "experiments/pinn_solver/artifacts/extension1/suite.json"
    formal, extended = [
        json.loads(path.read_text()) for path in (formal_path, extended_path)
    ]
    protocols = [
        json.loads((path.parent / "protocol.json").read_text())
        for path in (formal_path, extended_path)
    ]
    first, second = protocols
    differences = {
        key: [first.get(key), second.get(key)]
        for key in first.keys() | second.keys()
        if first.get(key) != second.get(key)
    }
    if differences != {"lbfgs_max_blocks": [200, 500]}:
        raise AssertionError(differences)
    assert (
        formal["completed"]
        and formal["all_runs_completed"]
        and len(formal["runs"]) == 24
    )
    assert (
        extended["completed"]
        and extended["all_runs_completed"]
        and len(extended["runs"]) == 24
    )
    assert extended["original_suite_sha256"] == sha(formal_path)
    for report, path in ((formal, formal_path), (extended, extended_path)):
        assert report["protocol_sha256"] == sha(path.parent / "protocol.json")
    solver_checks = []
    initial_hashes, point_hashes = defaultdict(set), defaultdict(set)
    for old, new in zip(formal["runs"], extended["runs"]):
        assert (old["backend"], old["seed"]) == (new["backend"], new["seed"])
        old_path, new_path = (
            formal_path.parent / old["output"],
            local_path(new["output"]),
        )
        assert sha(old_path) == old["output_sha256"] == new["original_output_sha256"]
        assert sha(new_path) == new["output_sha256"]
        before, after = [json.loads(path.read_text()) for path in (old_path, new_path)]
        assert before["problem"] == after["problem"]
        for value, path in ((before, old_path), (after, new_path)):
            assert value["completed"]
            assert sha(path.with_suffix(".pt")) == value["checkpoint_sha256"]
            checkpoint = torch.load(
                path.with_suffix(".pt"), map_location="cpu", weights_only=True
            )
            tensor = checkpoint["flat_parameters"]
            assert tensor.dtype == torch.float64 and tensor.numel() == 4547
            assert bool(torch.isfinite(tensor).all())
            assert (
                hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                == value["result"]["parameter_final_sha256"]
            )
        result = after["result"]
        thresholds = second["thresholds"]
        acceptance = all(
            metrics is not None
            and all(
                math.isfinite(metrics[key]) and metrics[key] <= limit
                for key, limit in thresholds.items()
            )
            for metrics in (result["metrics"], result["independent_metrics"])
        )
        assert result["converged"] == acceptance
        assert (
            before["result"]["parameter_initial_sha256"]
            == result["parameter_initial_sha256"]
        )
        initial_hashes[old["seed"]].add(result["parameter_initial_sha256"])
        for key in (
            "training_points_sha256",
            "pde_weights_sha256",
            "boundary_weights_sha256",
            "validation_points_sha256",
            "validation_boundary_sha256",
        ):
            point_hashes[key].add(after["problem"][key])
        old_history, new_history = before["result"]["history"], result["history"]
        prefix_exact = True
        prefix_metric_max_abs = 0.0
        first_difference = None
        evaluation_counts_equal = True
        for index, (a, b) in enumerate(zip(old_history, new_history)):
            for key in (
                "phase",
                "adam_updates",
                "lbfgs_iterations",
                "converged",
            ):
                assert a[key] == b[key], (old["backend"], old["seed"], key)
            evaluation_counts_equal &= (
                a["loss_gradient_evaluations"] == b["loss_gradient_evaluations"]
            )
            for key in thresholds:
                error = abs(a["metrics"][key] - b["metrics"][key])
                prefix_metric_max_abs = max(prefix_metric_max_abs, error)
                prefix_exact &= error == 0
                if error and first_difference is None:
                    first_difference = {
                        "check": index,
                        "phase": a["phase"],
                        "adam_updates": a["adam_updates"],
                        "lbfgs_iterations": a["lbfgs_iterations"],
                        "metric": key,
                        "difference": error,
                    }
        assert len(new_history) >= len(old_history)
        solver_checks.append(
            {
                "seed": old["seed"],
                "backend": old["backend"],
                "original_converged": old["result"]["converged"],
                "final_converged": acceptance,
                "checkpoint_hashes_verified": True,
                "same_problem_and_initialization": True,
                "reused_converged_run": new["reused_converged_run"],
                "overlapping_check_count": len(old_history),
                "overlapping_metrics_exact": prefix_exact,
                "overlapping_metrics_max_abs": prefix_metric_max_abs,
                "overlapping_evaluation_counts_equal": evaluation_counts_equal,
                "first_overlapping_metric_difference": first_difference,
            }
        )
    assert all(
        len(values) == 1
        for values in (*initial_hashes.values(), *point_hashes.values())
    )
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "Recorded source bytes, own compiled binaries and formal tracked-file hashes resolve to local current files or immutable executed snapshots; solver checkpoints, protocols, input identity, overlapping capped/extended trajectories and final acceptance checked. Third-party runtime wheels, Mathlib cache and CUDA system libraries are version/hash recorded but are not bundled in full.",
        "source_reports": records,
        "unresolved": unresolved,
        "solver_protocol_only_difference": differences,
        "solver_runs": solver_checks,
        "solver_original_converged": sum(
            row["original_converged"] for row in solver_checks
        ),
        "solver_final_converged": sum(row["final_converged"] for row in solver_checks),
        "all_initializations_paired": True,
        "all_data_hashes_identical": True,
        "all_source_references_resolved": not unresolved,
        "all_extended_prefixes_bitwise_equal": all(
            row["overlapping_metrics_exact"]
            and row["overlapping_evaluation_counts_equal"]
            for row in solver_checks
        ),
        "bitwise_replay_note": "Bitwise equality is reported separately from requested acceptance. The PhysicsNeMo rerun has a 3.47e-18 metric difference at Adam update 100 and later different L-BFGS trajectory/evaluation counts despite unchanged source, data and initialization; six other fresh extension runs have identical overlapping metrics.",
        "passed": not unresolved
        and all(row["final_converged"] for row in solver_checks),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            "reports": len(records),
            "unresolved": len(unresolved),
            "solver_final_converged": result["solver_final_converged"],
            "passed": result["passed"],
        }
    )
    for item in unresolved[:20]:
        print(item["report"], item["field"], item["sha256"])
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
