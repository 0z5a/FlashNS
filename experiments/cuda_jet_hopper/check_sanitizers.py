"""Run all four CUDA sanitizers on the already validated binary grid."""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    directory = args.build.resolve()
    output = directory / "sanitizers"
    output.mkdir(exist_ok=False)
    build = json.loads((directory / "build.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    if not validation["passed"]:
        raise RuntimeError("numerical validation must pass first")
    libraries = []
    for name, entry in build["libraries"].items():
        path = directory / entry["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            digest != entry["sha256"]
            or digest != validation["candidates"][name]["binary_sha256"]
        ):
            raise RuntimeError("binary changed since validation")
        libraries.append(str(path))
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "build_sha256": hashlib.sha256(
            (directory / "build.json").read_bytes()
        ).hexdigest(),
        "validation_sha256": hashlib.sha256(
            (directory / "validation.json").read_bytes()
        ).hexdigest(),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), HERE / "sanitizer_smoke.cu")
        },
        "tools": {},
        "all_passed": False,
    }

    def save():
        (output / "sanitizers.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, command, timeout=1800):
        started = perf_counter()
        with (output / (label + ".log")).open("w") as log:
            process = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            )
        row = {
            "command": command,
            "returncode": process.returncode,
            "seconds": perf_counter() - started,
            "log": label + ".log",
        }
        print(
            {"phase": label, **{key: row[key] for key in ("returncode", "seconds")}},
            flush=True,
        )
        return row

    save()
    executable = output / "sanitizer_smoke"
    report["build"] = run(
        "build",
        [
            "nvcc",
            "-O3",
            "-std=c++17",
            "-arch=sm_90",
            "-lineinfo",
            str(HERE / "sanitizer_smoke.cu"),
            "-ldl",
            "-o",
            str(executable),
        ],
    )
    if report["build"]["returncode"]:
        save()
        raise RuntimeError("standalone driver build failed")
    report["executable_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
    report["standalone"] = run("standalone", [str(executable), *libraries])
    save()
    if report["standalone"]["returncode"]:
        raise RuntimeError("standalone validation failed")
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        report["tools"][tool] = run(
            tool,
            [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                str(executable),
                *libraries,
            ],
        )
        save()
    report["all_passed"] = all(
        result["returncode"] == 0 for result in report["tools"].values()
    )
    # Probe hardware-counter permission once. Never change host restrictions.
    selected = directory / "libm64_n64_s2_c2_e0_w1.so"
    report["ncu"] = run(
        "ncu",
        [
            "ncu",
            "--set",
            "basic",
            "--kernel-name",
            "regex:dgrad",
            "--launch-count",
            "1",
            str(executable),
            str(selected),
        ],
        timeout=180,
    )
    ncu_text = (output / "ncu.log").read_text()
    report["ncu"]["counter_permission_denied"] = "ERR_NVGPUCTRPERM" in ncu_text
    report["sass"] = {}
    for copy in (0, 1, 2):
        identifier = f"m64_n64_s2_c{copy}_e0_w1"
        sass = subprocess.check_output(
            ["cuobjdump", "--dump-sass", str(directory / ("lib" + identifier + ".so"))],
            text=True,
        )
        (output / (identifier + ".sass")).write_text(sass)
        report["sass"][identifier] = {
            "sha256": hashlib.sha256(sass.encode()).hexdigest(),
            "dmma_occurrences": len(re.findall(r"\bDMMA\b", sass)),
            "ldgsts_occurrences": len(re.findall(r"\bLDGSTS\b", sass)),
            "tma_mnemonics": sorted(
                set(re.findall(r"\bUTMA\w+(?:\.[A-Z0-9_]+)*", sass))
            ),
        }
    report["finished_utc"] = datetime.now(UTC).isoformat()
    save()
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
