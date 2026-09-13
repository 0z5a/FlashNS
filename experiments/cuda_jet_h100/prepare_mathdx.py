"""Download isolated, hash-checked NVIDIA SDK/compiler components for A/B.

This only writes inside third_party/mathdx_h100; it does not replace the machine
CUDA toolkit or driver. The MathDx archive is versioned and its observed SHA is
recorded; CUDA component hashes are checked against NVIDIA's release manifest.
"""

import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / "third_party/mathdx_h100"
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
MANIFEST = BASE + "redistrib_13.0.2.json"
SDK = "https://developer.nvidia.com/downloads/compute/cublasdx/redist/cublasdx/cuda13/nvidia-mathdx-26.06.1-cuda13.tar.gz"


def download(url, path, expected=None):
    if not path.exists():
        with (
            urllib.request.urlopen(url, timeout=90) as response,
            path.with_suffix(".part").open("wb") as file,
        ):
            shutil.copyfileobj(response, file)
        path.with_suffix(".part").rename(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and sha != expected:
        raise RuntimeError(f"hash mismatch: {path.name}")
    return {"url": url, "file": path.name, "sha256": sha, "bytes": path.stat().st_size}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archives = DEST / "archives"
    archives.mkdir(exist_ok=True)
    manifest_file = archives / "redistrib_13.0.2.json"
    records = [download(MANIFEST, manifest_file)]
    manifest = json.loads(manifest_file.read_text())
    toolkit = DEST / "cuda-13.0.2"
    toolkit.mkdir(exist_ok=True)
    for name in (
        "cuda_nvcc",
        "libnvvm",
        "cuda_crt",
        "cuda_cudart",
        "cuda_cccl",
        "cuda_culibos",
    ):
        data = manifest[name]["linux-x86_64"]
        path = archives / Path(data["relative_path"]).name
        records.append(
            {
                "component": name,
                "version": manifest[name]["version"],
                **download(BASE + data["relative_path"], path, data["sha256"]),
            }
        )
        component = DEST / "components" / name
        component.mkdir(exist_ok=True, parents=True)
        with tarfile.open(path) as tar:
            tar.extractall(component, filter="data")
        roots = list(component.iterdir())
        if len(roots) != 1:
            raise RuntimeError(f"unexpected archive root: {name}")
        shutil.copytree(roots[0], toolkit, dirs_exist_ok=True, symlinks=True)
        print({"downloaded": name, "version": manifest[name]["version"]}, flush=True)
    sdk_archive = archives / "nvidia-mathdx-26.06.1-cuda13.tar.gz"
    records.append(
        {"component": "MathDx", "version": "26.06.1", **download(SDK, sdk_archive)}
    )
    sdk_root = DEST / "sdk"
    sdk_root.mkdir(exist_ok=True)
    with tarfile.open(sdk_archive) as tar:
        tar.extractall(sdk_root, filter="data")
    report = {
        "cuda_manifest": MANIFEST,
        "cuda_version": manifest["release_label"],
        "toolkit": str(toolkit),
        "sdk_root": str(sdk_root),
        "archives": records,
        "system_toolkit_or_driver_modified": False,
    }
    (HERE / "artifacts/mathdx_dependencies.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
