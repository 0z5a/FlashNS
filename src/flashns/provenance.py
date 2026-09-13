"""Pinned public sources, local hashes, and immutable-input checks."""

import hashlib
import json
import platform
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_registry(root):
    return json.loads((Path(root) / "sources/registry.json").read_text())


def verify_sources(root, *, names=None):
    root = Path(root)
    result = {}
    for key, item in source_registry(root)["artifacts"].items():
        if names is not None and key not in names:
            continue
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing pinned source {path}; run flashns fetch-sources"
            )
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(
                f"Source hash mismatch for {path}; refusing unpinned input"
            )
        result[key] = {"sha256": digest, "bytes": path.stat().st_size}
    return result


def fetch_sources(root):
    """Download pinned artifacts. A changed upstream file is never accepted."""
    root = Path(root)
    for key, item in source_registry(root)["artifacts"].items():
        if "url" not in item:
            continue
        target = root / item["path"]
        if target.exists():
            if sha256(target) != item["sha256"]:
                raise ValueError(f"Existing {target} does not match pinned source")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(
            item["url"], headers={"User-Agent": "FlashNS-repro/0.1"}
        )
        with (
            urllib.request.urlopen(request, timeout=180) as response,
            partial.open("wb") as out,
        ):
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > item["bytes"]:
                    raise ValueError(f"Downloaded {key} exceeds pinned length")
                out.write(block)
        if sha256(partial) != item["sha256"]:
            raise ValueError(
                f"Downloaded {key} differs from pinned artifact; kept as .part"
            )
        partial.replace(target)
    registry = source_registry(root)["artifacts"]
    for key, item in registry.items():
        if "archive" not in item:
            continue
        target = root / item["path"]
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("source destination leaves project directory")
        if target.exists():
            if sha256(target) != item["sha256"]:
                raise ValueError(f"Existing extracted source changed: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        with zipfile.ZipFile(root / registry[item["archive"]]["path"]) as archive:
            info = archive.getinfo(item["member"])
            if info.file_size != item["bytes"]:
                raise ValueError(f"Archive member size changed: {key}")
            with archive.open(info) as source, partial.open("wb") as out:
                while block := source.read(1024 * 1024):
                    out.write(block)
        if sha256(partial) != item["sha256"]:
            raise ValueError(f"Extracted source hash mismatch: {key}")
        partial.replace(target)
    return verify_sources(root)


def run_metadata(root):
    root = Path(root)
    paths = [
        *sorted((root / "src/flashns").glob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    paths += sorted((root / "cases").glob("*.json"))
    paths += sorted((root / "scripts").glob("*.py"))
    paths += sorted((root / "adapters").glob("*.json"))
    paths += [root / "sources/registry.json"]
    # AppleDouble entries from macOS archives are metadata, not source inputs.
    files = {
        str(p.relative_to(root)): sha256(p)
        for p in paths
        if p.is_file() and not p.name.startswith("._")
    }
    deps = {name: version(name) for name in ["numpy", "scipy", "sympy", "mpmath"]}
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": deps,
        "input_and_code_hashes": files,
        "code_fingerprint": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)
