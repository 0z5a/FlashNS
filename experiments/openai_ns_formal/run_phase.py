"""Run one formal verification phase, preserving command, sources and exit status."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.output.exists() or os.geteuid() == 0:
        raise RuntimeError("require a command, a new output, and non-root execution")
    tracked = git(args.cwd, "ls-files", "-z").split("\0")
    source_hashes = {
        name: hashlib.sha256((args.cwd / name).read_bytes()).hexdigest()
        for name in tracked
        if name
    }
    dependencies = {}
    for package in sorted((args.cwd / ".lake/packages").iterdir()):
        if (package / ".git").exists():
            dependencies[package.name] = {
                "commit": git(package, "rev-parse", "HEAD").strip(),
                "tracked_diff": git(package, "diff", "HEAD", "--"),
            }
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "uid": os.geteuid(),
        "command": command,
        "cwd": str(args.cwd),
        "commit": git(args.cwd, "rev-parse", "HEAD").strip(),
        "source_hashes_before": source_hashes,
        "tracked_diff_before": git(args.cwd, "diff", "HEAD", "--"),
        "dependency_revisions": dependencies,
        "lean_num_threads": os.environ.get("LEAN_NUM_THREADS"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "completed": False,
        "exit_code": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        partial = args.output.with_suffix(".tmp")
        partial.write_text(json.dumps(report, indent=2) + "\n")
        partial.replace(args.output)

    save()
    start = perf_counter()
    with args.output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        report["pid"] = process.pid
        report["process_group"] = process.pid
        save()
        report["exit_code"] = process.wait()
    report["wall_seconds"] = perf_counter() - start
    report["source_hashes_after"] = {
        name: hashlib.sha256((args.cwd / name).read_bytes()).hexdigest()
        for name in source_hashes
    }
    report["tracked_diff_after"] = git(args.cwd, "diff", "HEAD", "--")
    report["sources_unchanged"] = source_hashes == report["source_hashes_after"]
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report["children_usage"] = {
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "maximum_child_peak_rss_kib_linux": usage.ru_maxrss,
    }
    report["finished_utc"] = datetime.now(UTC).isoformat()
    report["completed"] = True
    report["passed"] = report["exit_code"] == 0 and report["sources_unchanged"]
    save()
    print(
        {key: report[key] for key in ("passed", "exit_code", "wall_seconds")},
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
