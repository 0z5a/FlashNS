"""Run all four Compute Sanitizer modes on the same v8 build and smoke cases."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tool = shutil.which("compute-sanitizer")
    if tool is None:
        raise RuntimeError("Compute Sanitizer is unavailable; no sanitizer pass claimed")
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []
    for mode in ("memcheck", "racecheck", "initcheck", "synccheck"):
        result_path = args.output / (mode + ".json")
        command = [tool, "--tool", mode, "--error-exitcode", "86", sys.executable,
                   str(Path(__file__).with_name("preflight.py")), "--build", str(args.build.resolve()), "--quick", "--output", str(result_path.resolve())]
        with (args.output / (mode + ".log")).open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        passed = process.returncode == 0 and result_path.exists() and json.loads(result_path.read_text())["passed"]
        checks.append({"tool": mode, "passed": passed, "exit_code": process.returncode, "command": command})
        print(mode, passed, flush=True)
    report = {"passed": all(row["passed"] for row in checks), "checks": checks}
    (args.output / "sanitizers.json").write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
