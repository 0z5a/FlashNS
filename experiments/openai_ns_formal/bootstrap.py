"""Install the pinned Lean toolchain into a project-owned, unprivileged prefix."""

import hashlib
import json
import os
import subprocess
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def main():
    root = Path(os.environ["TASK_FORMAL"])
    if os.geteuid() == 0:
        raise RuntimeError("formal tools must run as an unprivileged user")
    downloads = root / "downloads"
    request = urllib.request.Request(
        "https://api.github.com/repos/leanprover/elan/releases/tags/v4.2.4",
        headers={"User-Agent": "FlashNS-reproduction"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        release_bytes = response.read()
    (downloads / "elan-v4.2.4-release.json").write_bytes(release_bytes)
    release = json.loads(release_bytes)
    asset = next(
        a
        for a in release["assets"]
        if a["name"] == "elan-x86_64-unknown-linux-gnu.tar.gz"
    )
    package = downloads / asset["name"]
    request = urllib.request.Request(
        asset["browser_download_url"],
        headers={"User-Agent": "FlashNS-reproduction"},
    )
    with (
        urllib.request.urlopen(request, timeout=180) as response,
        package.open("wb") as destination,
    ):
        while block := response.read(1024 * 1024):
            destination.write(block)
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    upstream_digest = asset.get("digest")
    if upstream_digest is not None and upstream_digest != "sha256:" + digest:
        raise RuntimeError("elan release asset hash mismatch")
    installer = root / "tools/elan-installer"
    installer.mkdir(exist_ok=True)
    with tarfile.open(package) as archive:
        archive.extractall(installer, filter="data")
    provenance = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "uid": os.geteuid(),
        "elan_version": "4.2.4",
        "elan_asset_url": asset["browser_download_url"],
        "elan_asset_sha256": digest,
        "github_asset_digest": upstream_digest,
        "toolchain": "leanprover/lean4:v4.34.0-rc2",
        "completed": False,
    }
    output = root / "logs/bootstrap.json"
    output.write_text(json.dumps(provenance, indent=2) + "\n")
    subprocess.run(
        [
            str(installer / "elan-init"),
            "-y",
            "--no-modify-path",
            "--default-toolchain",
            "none",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(root / "elan/bin/elan"),
            "toolchain",
            "install",
            provenance["toolchain"],
        ],
        check=True,
    )
    for arguments in (["--version"], ["--help"]):
        subprocess.run(
            [str(root / "elan/bin/lake"), *arguments],
            cwd=root / "repository",
            check=True,
        )
    provenance["completed"] = True
    output.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
