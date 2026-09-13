"""Install isolated, hashed Comparator dependencies as an unprivileged user."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def fetch(url, destination):
    with urllib.request.urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() == 0:
        raise RuntimeError("run as an unprivileged user")
    root = args.root.resolve()
    output = root / "logs/comparator-tools.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite a dependency run")
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "uid": os.geteuid(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "completed": False,
        "commands": [],
        "downloads": [],
        "repositories": {},
    }

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    environment = dict(os.environ)
    environment.update(
        PATH=f"{root}/tools/go/bin:{root}/tools/cargo/bin:/usr/local/bin:/usr/bin:/bin",
        CARGO_HOME=str(root / "tools/cargo"),
        RUSTUP_HOME=str(root / "tools/rustup"),
        GOCACHE=str(root / "cache/go-build"),
        GOMODCACHE=str(root / "cache/go-mod"),
        GOPATH=str(root / "tools/go-path"),
        GOMAXPROCS="8",
        CARGO_BUILD_JOBS="8",
    )

    def run(command, cwd=root):
        report["commands"].append({"command": list(map(str, command)), "cwd": str(cwd)})
        save()
        subprocess.run(command, cwd=cwd, env=environment, check=True)

    def download(url, name, expected):
        destination = root / "downloads" / name
        actual = fetch(url, destination)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {name}")
        report["downloads"].append({"url": url, "sha256": actual, "file": name})
        save()
        return destination

    save()
    go_manifest = root / "downloads/go-releases.json"
    fetch("https://go.dev/dl/?mode=json", go_manifest)
    releases = json.loads(go_manifest.read_text())
    release = next(r for r in releases if r["stable"])
    asset = next(
        f
        for f in release["files"]
        if f["os"] == "linux" and f["arch"] == "amd64" and f["kind"] == "archive"
    )
    archive = download(
        "https://go.dev/dl/" + asset["filename"], asset["filename"], asset["sha256"]
    )
    with tarfile.open(archive) as source:
        source.extractall(root / "tools", filter="data")
    rust_url = (
        "https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init"
    )
    digest_file = root / "downloads/rustup-init.sha256"
    fetch(rust_url + ".sha256", digest_file)
    installer = download(rust_url, "rustup-init", digest_file.read_text().split()[0])
    installer.chmod(0o700)
    run(
        [
            str(installer),
            "-y",
            "--no-modify-path",
            "--profile",
            "minimal",
            "--default-toolchain",
            "stable",
        ]
    )
    for name, url, branch in (
        ("landrun", "https://github.com/Zouuup/landrun.git", "main"),
        ("nanoda", "https://github.com/ammkrn/nanoda_lib.git", "master"),
    ):
        revision = subprocess.check_output(
            ["git", "ls-remote", url, f"refs/heads/{branch}"],
            env=environment,
            text=True,
        ).split()[0]
        repository = root / "tools" / name
        run(["git", "clone", "--no-checkout", url, str(repository)])
        run(["git", "checkout", "--detach", revision], repository)
        report["repositories"][name] = {"url": url, "revision": revision}
        save()
    run(["go", "build", "-o", "landrun", "cmd/landrun/main.go"], root / "tools/landrun")
    run(["cargo", "build", "--release", "--locked"], root / "tools/nanoda")
    report["binaries"] = {
        name: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for name, relative in (
            ("landrun", "tools/landrun/landrun"),
            ("nanoda_bin", "tools/nanoda/target/release/nanoda_bin"),
        )
    }
    report["versions"] = {
        name: subprocess.check_output(command, env=environment, text=True).strip()
        for name, command in (
            ("go", ["go", "version"]),
            ("rustc", ["rustc", "--version"]),
            ("cargo", ["cargo", "--version"]),
        )
    }
    report["completed"] = True
    report["finished_utc"] = datetime.now(UTC).isoformat()
    save()
    print(
        json.dumps(
            {
                "completed": True,
                "repositories": report["repositories"],
                "versions": report["versions"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
