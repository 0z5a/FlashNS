"""Compile a standalone smoke and preserve separate userspace debug results."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


def run(label, command, timeout=180):
    start = perf_counter()
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
        output = result.stdout + result.stderr
        code = result.returncode
    except subprocess.TimeoutExpired as error:
        output = f"TIMEOUT after {timeout}s\n{error.stdout!r}\n{error.stderr!r}"
        code = None
    (ART / f"{label}.log").write_text(output)
    row = {
        "command": command,
        "returncode": code,
        "seconds": perf_counter() - start,
        "log": f"{label}.log",
    }
    print({"tool": label, "returncode": code, "seconds": row["seconds"]}, flush=True)
    return row


def main():
    ART.mkdir(exist_ok=True)
    command = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-arch=sm_86",
        "-lineinfo",
        str(HERE / "sanitizer_smoke.cu"),
        str(ART / "libdgrad_jet.so"),
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        str(ART),
        "-o",
        str(ART / "sanitizer_smoke"),
    ]
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "build": run("smoke_build", command),
    }
    if report["build"]["returncode"] != 0:
        raise RuntimeError("standalone smoke build failed; see smoke_build.log")
    report["source_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            HERE / "sanitizer_smoke.cu",
            HERE / "dgrad_jet_vjp.cu",
            ART / "libdgrad_jet.so",
            ART / "sanitizer_smoke",
        ]
    }
    report["standalone"] = run("smoke", [str(ART / "sanitizer_smoke")])
    report["tools"] = {}
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        report["tools"][tool] = run(
            tool,
            [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                str(ART / "sanitizer_smoke"),
            ],
        )
        (ART / "sanitizers.json").write_text(json.dumps(report, indent=2) + "\n")
    report["all_passed"] = report["standalone"]["returncode"] == 0 and all(
        r["returncode"] == 0 for r in report["tools"].values()
    )
    (ART / "sanitizers.json").write_text(json.dumps(report, indent=2) + "\n")
    # Counter access can be disabled by the host; never change driver permissions.
    profile = run(
        "ncu",
        [
            "ncu",
            "--set",
            "basic",
            "--launch-count",
            "1",
            str(ART / "sanitizer_smoke"),
            "short",
        ],
    )
    (ART / "hardware_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
