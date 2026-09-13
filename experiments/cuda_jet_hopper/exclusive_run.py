"""Pause only this experiment's recorded proof job during GPU timing."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.output.exists() or not command:
        raise RuntimeError("require a new output and a command")
    phase = json.loads(args.formal_report.read_text())
    report = {"timestamp_utc": datetime.now(UTC).isoformat(), "formal_report": str(args.formal_report), "formal_phase_completed_before_timing": phase.get("completed", False), "command": command, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "paused": False}
    group = None
    try:
        if not phase.get("completed", False):
            pid = phase["process_group"]
            process = Path(f"/proc/{pid}")
            if process.exists():
                if os.stat(process).st_uid != 65533 or os.getpgid(pid) != pid or not str(args.formal_report).startswith("/workspace/flashns-comparator-20260909/logs/"):
                    raise RuntimeError("refusing to signal a process outside the recorded checker job")
                report["process_group"] = pid
                report["process_command"] = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                os.killpg(pid, signal.SIGSTOP)
                group = pid
                started = perf_counter()
                sleep(0.3)
                report["paused"] = True
                report["pause_started_utc"] = datetime.now(UTC).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["returncode"] = subprocess.run(command, check=False).returncode
    finally:
        if group is not None:
            os.killpg(group, signal.SIGCONT)
            report["pause_seconds"] = perf_counter() - started
            report["resumed_utc"] = datetime.now(UTC).isoformat()
        report["finished_utc"] = datetime.now(UTC).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
