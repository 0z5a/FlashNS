"""Resolve recorded measurement source hashes against local files and snapshots."""

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def source_maps(value, path=""):
    if isinstance(value, dict):
        for name, child in value.items():
            location = f"{path}/{name}"
            if name in ("source_hashes", "input_and_code_hashes") and isinstance(
                child, dict
            ):
                yield location, child
            elif name == "script_sha256" and isinstance(child, str):
                yield location, {"entry_point_script": child}
            elif (
                name == "sources"
                and isinstance(child, dict)
                and all(
                    isinstance(item, dict) and isinstance(item.get("sha256"), str)
                    for item in child.values()
                )
            ):
                yield location, {key: item["sha256"] for key, item in child.items()}
            else:
                yield from source_maps(child, location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from source_maps(child, f"{path}/{index}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "artifact_audit.json")
    args = parser.parse_args()
    corpus = defaultdict(list)
    paths = [
        *ROOT.glob("src/flashns/*.py"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("tests/*.py"),
        *ROOT.glob("cases/*.json"),
        *ROOT.glob("adapters/*.json"),
        *ROOT.glob("experiments/cuda_jet/input/*"),
        *HERE.glob("*.py"),
        *HERE.glob("*.cu"),
        *HERE.glob("*.cuh"),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "sources/registry.json",
    ]
    registry = json.loads((ROOT / "sources/registry.json").read_text())
    paths.extend(ROOT / item["path"] for item in registry["artifacts"].values())
    artifact_dirs = [*HERE.glob("artifacts*"), ROOT / "artifacts/h100nvl"]
    for artifact_dir in artifact_dirs:
        if artifact_dir.is_dir():
            paths.extend(artifact_dir.rglob("*"))
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        relative = str(path.relative_to(ROOT))
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        corpus[digest].append(relative)
        if path.name.endswith("sources.tar.gz"):
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    if member.isfile():
                        data = archive.extractfile(member).read()
                        corpus[hashlib.sha256(data).hexdigest()].append(
                            relative + ":" + member.name
                        )
    reports, missing = [], []
    measured_names = (
        "validation_gpu0",
        "sanitizers",
        "profile_costs",
        "local_benchmark_gpu0",
        "graph_experiment",
        "baseline_plus_smoke",
        "baseline_plus",
        "scientific_validation_v2",
        "scientific_benchmark",
        "mathdx_validation",
        "mathdx_benchmark",
        "mathdx_sanitizers",
        "euler_ab_smoke",
        "euler_ab_4m",
        "euler_precision",
    )
    for artifact_dir in sorted(artifact_dirs):
        if not artifact_dir.is_dir():
            continue
        for name in measured_names:
            path = artifact_dir / f"{name}.json"
            if not path.exists():
                continue
            value = json.loads(path.read_text())
            entries = []
            for location, hashes in source_maps(value):
                for source, digest in hashes.items():
                    row = {
                        "field": location,
                        "source": source,
                        "sha256": digest,
                        "resolved_locations": corpus.get(digest, []),
                    }
                    entries.append(row)
                    if not row["resolved_locations"]:
                        missing.append({"report": str(path.relative_to(ROOT)), **row})
            reports.append(
                {
                    "report": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_references": entries,
                    "all_source_references_resolved": all(
                        r["resolved_locations"] for r in entries
                    ),
                }
            )
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "recorded measurement source bytes and declared pinned source artifacts; does not claim all runtime wheels or remote-only failed-run artifacts were downloaded; canonical Euler point hash is checked by the measurement script",
        "reports": reports,
        "unresolved": missing,
        "all_measured_source_references_resolved": bool(reports) and not missing,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            "reports": len(reports),
            "unresolved_source_references": len(missing),
            "output": str(args.output),
        }
    )
    return 0 if result["all_measured_source_references_resolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
