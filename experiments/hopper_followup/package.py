"""Package the completed follow-up and verify every archive member."""

import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "artifacts/flashns-hopper-formal-solver-20260909.tar.xz"
MANIFEST = ROOT / "artifacts/hopper_followup/manifest.json"


def main():
    if ARCHIVE.exists():
        raise RuntimeError("refusing to overwrite final archive")
    audit = json.loads((ROOT / "artifacts/hopper_followup/audit.json").read_text())
    assert audit["passed"]
    files = {
        ROOT / name for name in ("README.md", "pyproject.toml", "uv.lock", ".gitignore")
    }
    directories = [
        "src",
        "tests",
        "scripts",
        "cases",
        "adapters",
        "experiments/cuda_jet_hopper",
        "experiments/openai_ns_formal",
        "experiments/pinn_solver",
        "experiments/hopper_followup",
        "artifacts/openai_ns_formal",
        "artifacts/hopper_followup",
        "sources/openai_ns_formal/repository",
        "experiments/cuda_jet/input",
    ]
    for name in directories:
        files.update((ROOT / name).rglob("*"))
    for name in (
        "cuda-hopper-results.md",
        "openai-ns-formal-results.md",
        "pinn-solver-results.md",
        "h100-next-execution.md",
        "cuda-h100-results.md",
    ):
        files.add(ROOT / "docs" / name)
    files.update(
        path
        for path in (ROOT / "experiments/cuda_jet_h100").glob("*")
        if path.suffix in (".py", ".cu", ".cuh", ".txt", ".md")
    )
    files.add(ROOT / "sources/openai_ns_formal/registry.json")
    files.add(ROOT / "sources/registry.json")
    files = {
        path
        for path in files
        if path.is_file()
        and not any(
            part in (".git", "__pycache__", ".pytest_cache") for part in path.parts
        )
        and path.name != ".DS_Store"
        and path != MANIFEST
        and not path.name.startswith("remote-evidence.tar.")
    }
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "This three-part H100 NVL follow-up: current source, exact executed snapshots, generated numerical data/results, all new own CUDA binaries, checkpoints, original failed attempts, formal build/check logs, pinned Apache-2.0 Lean source, and standalone figures. Runtime wheels, compiler/Mathlib caches, full CUTLASS checkout, transport archives, prior-stage large datasets and the official PDF are excluded; retrieve external dependencies/PDF by the recorded URLs, commits and hashes.",
        "files": {},
    }
    for path in sorted(files):
        data = path.read_bytes()
        manifest["files"][str(path.relative_to(ROOT))] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(ARCHIVE, "x:xz", preset=9) as archive:
        for path in sorted(files | {MANIFEST}):
            data = path.read_bytes()
            info = tarfile.TarInfo(str(path.relative_to(ROOT)))
            info.size, info.mode = (
                len(data),
                0o755 if path.stat().st_mode & 0o111 else 0o644,
            )
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(ARCHIVE, "r:xz") as archive:
        stored = json.load(archive.extractfile(str(MANIFEST.relative_to(ROOT))))
        assert stored == manifest
        assert len(archive.getmembers()) == len(manifest["files"]) + 1
        for name, record in manifest["files"].items():
            data = archive.extractfile(name).read()
            assert (
                len(data) == record["bytes"]
                and hashlib.sha256(data).hexdigest() == record["sha256"]
            )
    with ARCHIVE.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256").write_text(
        digest + "  " + ARCHIVE.name + "\n"
    )
    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "sha256": digest,
                "bytes": ARCHIVE.stat().st_size,
                "members": len(files) + 1,
                "all_members_verified": True,
            }
        )
    )


if __name__ == "__main__":
    main()
