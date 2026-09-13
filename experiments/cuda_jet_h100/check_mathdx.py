"""Check the exact MathDx cubin with a standalone driver caller and Sanitizers."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from check_sanitizers import run

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


def main():
    build = json.loads((ART / "build_mathdx.json").read_text())
    cubin = ART / build["path"]
    if hashlib.sha256(cubin.read_bytes()).hexdigest() != build["sha256"]:
        raise RuntimeError("MathDx cubin hash changed")
    executable = ART / "mathdx_smoke"
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "tested_build": build,
        "build": run(
            "mathdx_smoke_build",
            [
                "/usr/local/cuda/bin/nvcc",
                "-O3",
                "-std=c++17",
                str(HERE / "mathdx_smoke.cu"),
                "-L/usr/local/cuda/lib64/stubs",
                "-lcuda",
                "-o",
                str(executable),
            ],
        ),
    }
    if report["build"]["returncode"] != 0:
        raise RuntimeError("standalone driver caller compilation failed")
    report["source_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), HERE / "mathdx_smoke.cu", executable)
    }
    report["standalone"] = run("mathdx_standalone", [str(executable), str(cubin)])
    report["tools"] = {}
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        report["tools"][tool] = run(
            f"mathdx_{tool}",
            [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                str(executable),
                str(cubin),
            ],
        )
        (ART / "mathdx_sanitizers.json").write_text(json.dumps(report, indent=2) + "\n")
    report["all_passed"] = all(
        row["returncode"] == 0
        for row in [report["standalone"], *report["tools"].values()]
    )
    (ART / "mathdx_sanitizers.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
