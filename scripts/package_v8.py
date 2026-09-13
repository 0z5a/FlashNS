"""Build a source/local-evidence transfer ZIP for v8, without environments or binaries."""

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def payload_paths():
    selected = set()
    for name in ("src", "tests", "sources/flashns_v8_reference"):
        for path in (ROOT / name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("."):
                selected.add(path)
    selected.update((ROOT / "sources/vendor/eulerRepo").glob("*.py"))
    for name in ("experiments/pinn_v8", "experiments/pinn_solver", "experiments/cuda_jet_h100", "experiments/cuda_jet_hopper", "experiments/cuda_jet/input"):
        for path in (ROOT / name).glob("*"):
            if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".md"):
                selected.add(path)
    for path in (ROOT / "experiments/pinn_v8/artifacts").rglob("*"):
        if path.is_file() and path.suffix in (".json", ".log", ".xml", ".py", ".cu", ".cuh") and "__pycache__" not in path.parts:
            selected.add(path)
    for name in ("pyproject.toml", "uv.lock", "sources/registry.json", "docs/FlashNS_v8_Local_Implementation_ZH.md", "scripts/run_v8_gpu.sh", "scripts/package_v8.py"):
        selected.add(ROOT / name)
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite a transfer package")
    files = payload_paths()
    manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "v8 local source and CPU/HOST evidence; target GPU build and validation pending",
                "manifest_self_excluded": True, "files": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            name = "flashns/" + path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            manifest["files"].append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            archive.writestr(name, data)
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(args.output) as archive:
        if set(archive.namelist()) != {entry["path"] for entry in manifest["files"]} | {"MANIFEST.json"}:
            raise RuntimeError("archive file set differs from manifest")
        for entry in manifest["files"]:
            data = archive.read(entry["path"])
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError("archive verification failed: " + entry["path"])
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{sha}  {args.output.name}\n")
    print(json.dumps({"file": str(args.output.resolve()), "bytes": args.output.stat().st_size,
                      "payload_files": len(files), "sha256": sha, "all_payload_hashes_verified": True}, indent=2))


if __name__ == "__main__":
    main()
