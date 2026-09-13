"""Check saved report completeness and byte provenance without rerunning CUDA."""

import hashlib
import json
import tarfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = HERE / "artifacts"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    available = defaultdict(list)
    paths = [
        *HERE.glob("*.py"),
        *HERE.glob("*.cu"),
        *HERE.glob("*.cuh"),
        *ROOT.joinpath("src").rglob("*.py"),
        *ROOT.joinpath("scripts").glob("*.py"),
        *HERE.parent.joinpath("cuda_jet/input").glob("*.py"),
        *ART.glob("*.py"),
        *ART.glob("*.so"),
        ART / "sanitizer_smoke",
    ]
    for path in paths:
        if path.is_file():
            available[digest(path.read_bytes())].append(str(path.relative_to(ROOT)))
    for path in ART.glob("measured_sources*.tar.gz"):
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile():
                    available[digest(archive.extractfile(member).read())].append(
                        f"{path.name}:{member.name}"
                    )
    checked, missing = 0, []

    def verify_sources(value, location):
        nonlocal checked
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "source_hashes":
                    for source, expected in child.items():
                        checked += 1
                        if expected not in available:
                            missing.append(
                                {
                                    "report": location,
                                    "source": source,
                                    "sha256": expected,
                                }
                            )
                elif key == "script_sha256":
                    checked += 1
                    if child not in available:
                        missing.append({"report": location, "sha256": child})
                else:
                    verify_sources(child, location)
        elif isinstance(value, list):
            for child in value:
                verify_sources(child, location)

    report_paths = [
        ART / "environment.json",
        ART / "build.json",
        ART / "sanitizers.json",
        ART / "local_benchmark_gpu0.json",
        ART / "independent_throughput.json",
        ART / "profile_costs.json",
        ART / "resources.json",
        ART / "environment_final.json",
        *[ART / f"validation_gpu{gpu}.json" for gpu in range(4)],
        *[ART / f"controlled_weak_w{world}/rank0.json" for world in (1, 2, 4)],
        *[
            ART / f"scaling_w{world}/rank{rank}.json"
            for world in (1, 2, 4)
            for rank in range(world)
        ],
    ]
    files, states = {}, {}
    for path in report_paths:
        value = json.loads(path.read_text())
        name = str(path.relative_to(ART))
        verify_sources(value, name)
        files[name] = digest(path.read_bytes())
        if "failure" in value:
            raise AssertionError(f"saved failure: {name}")
        if path.name.startswith("validation_gpu"):
            states[name] = (
                value["validation"]["passed"]
                and value["high_precision_tails"]["passed"]
            )
        elif path.name == "sanitizers.json":
            states[name] = value["all_passed"]
        elif "passed" in value:
            states[name] = value["passed"]
    build = json.loads((ART / "build.json").read_text())
    for row in build["libraries"].values():
        assert row["returncode"] == 0
        assert digest((ART / row["path"]).read_bytes()) == row["sha256"]
    formal = ROOT / "sources/openai_ns_formal"
    registry = json.loads((formal / "registry.json").read_text())
    for name, row in registry["files"].items():
        assert digest((formal / name).read_bytes()) == row["sha256"]
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "saved report completeness and source/library byte provenance; numerical checks are not rerun by this audit",
        "source_hash_references_checked": checked,
        "unresolved_source_hashes": missing,
        "required_reports_sha256": files,
        "saved_check_states": states,
        "implementation_commit": None,
        "implementation_identity": "recorded source hashes and measured_sources archives",
        "passed": not missing and all(states.values()),
    }
    (ART / "artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            "reports": len(files),
            "source_references": checked,
            "unresolved": missing,
            "passed": result["passed"],
        }
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
