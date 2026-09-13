"""Collect this experiment's outputs and executed source without runtime caches."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite evidence archive")
    selected = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.is_symlink()
            or any(
                part in (".git", "__pycache__", ".libdeps", ".venv")
                for part in relative.parts
            )
        ):
            continue
        if relative.parts[0] == "third_party":
            if path.name.lower().startswith("license") and len(relative.parts) == 3:
                selected[
                    "artifacts/hopper_followup/executed_sources/" + str(relative)
                ] = path
            continue
        if len(relative.parts) == 1 and path.suffix == ".log":
            selected["artifacts/hopper_followup/logs/" + path.name] = path
        elif relative.parts[:3] == ("experiments", "cuda_jet_h100", "artifacts"):
            selected[
                "artifacts/hopper_followup/native_baseline/"
                + str(Path(*relative.parts[3:]))
            ] = path
        elif "artifacts" in relative.parts and relative.parts[0] == "experiments":
            selected[str(relative)] = path
        elif relative.parts[0] in ("experiments", "src") and path.suffix in (
            ".py",
            ".cu",
            ".cuh",
            ".txt",
            ".toml",
        ):
            selected["artifacts/hopper_followup/executed_sources/" + str(relative)] = (
                path
            )
        elif relative == Path(
            "sources/openai_ns_formal/repository/NavierStokes/TangentProjection.lean"
        ):
            selected["artifacts/hopper_followup/executed_sources/" + str(relative)] = (
                path
            )
    cutlass = root / "third_party/cutlass"
    revision = subprocess.check_output(
        ["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cutlass_commit": revision,
        "scope": "All new GPU experiment artifacts, checkpoints, failed/development logs, executed Python/CUDA sources and rebuilt native baseline; excludes installed runtime wheels, compiler caches and vendored CUTLASS checkout except revision/license.",
        "files": {},
    }
    with tarfile.open(args.output, "x:gz", compresslevel=6) as archive:
        for name, path in selected.items():
            content = path.read_bytes()
            manifest["files"][name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "remote_relative_path": str(path.relative_to(root)),
            }
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(content), int(path.stat().st_mtime)
            info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            archive.addfile(info, io.BytesIO(content))
        content = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("artifacts/hopper_followup/remote-manifest.json")
        info.size, info.mode = len(content), 0o644
        archive.addfile(info, io.BytesIO(content))
    with args.output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(
        json.dumps(
            {
                "files": len(selected),
                "archive": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": digest,
            }
        )
    )


if __name__ == "__main__":
    main()
