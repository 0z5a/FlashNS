"""Standalone checks of the exact libraries used in validation and timing."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


def run(label, command, timeout=240):
    start = perf_counter()
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
        output, code = result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        output, code = f"TIMEOUT {timeout}s\n{error.stdout!r}\n{error.stderr!r}", None
    path = ART / f"{label}.log"
    path.write_text(output)
    row = {
        "command": command,
        "returncode": code,
        "seconds": perf_counter() - start,
        "log": path.name,
    }
    print({"tool": label, "returncode": code, "seconds": row["seconds"]}, flush=True)
    return row


def main():
    build = json.loads((ART / "build.json").read_text())
    for name, row in build["libraries"].items():
        assert (
            hashlib.sha256((ART / row["path"]).read_bytes()).hexdigest()
            == row["sha256"]
        ), name
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "tested_build": build,
        "tools": {},
    }
    executable = str(ART / "sanitizer_smoke")
    report["build"] = run(
        "smoke_build",
        [
            "nvcc",
            "-O3",
            "-std=c++17",
            "-arch=sm_86",
            "-lineinfo",
            str(HERE / "sanitizer_smoke.cu"),
            "-ldl",
            "-o",
            executable,
        ],
    )
    if report["build"]["returncode"] != 0:
        raise RuntimeError("smoke compile failed")
    report["source_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), HERE / "sanitizer_smoke.cu", Path(executable))
    }
    report["standalone_by_gpu"] = {
        gpu: run(f"smoke_gpu{gpu}", [executable, str(ART), str(gpu)])
        for gpu in range(4)
    }
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        report["tools"][tool] = run(
            tool,
            [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                executable,
                str(ART),
                "0",
            ],
        )
        (ART / "sanitizers.json").write_text(json.dumps(report, indent=2) + "\n")
    report["all_passed"] = all(
        row["returncode"] == 0
        for row in [*report["standalone_by_gpu"].values(), *report["tools"].values()]
    )
    (ART / "sanitizers.json").write_text(json.dumps(report, indent=2) + "\n")
    profile = run(
        "ncu",
        [
            "ncu",
            "--set",
            "basic",
            "--kernel-name",
            "regex:dgrad",
            "--launch-count",
            "1",
            executable,
            str(ART),
            "0",
        ],
    )
    (ART / "hardware_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
