"""Run a frozen, finite command plan under the remote process supervisor."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"completed": False, "all_commands_exited_zero": False,
             "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
             "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "started_at_utc": datetime.now(timezone.utc).isoformat(), "commands": []}

    def save():
        temporary = args.output / "state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(args.output / "state.json")

    save()
    environment = dict(os.environ, **plan.get("environment", {}))
    for index, command in enumerate(plan["commands"]):
        row = {"name": command["name"], "argv": command["argv"], "completed": False,
               "log": f"{index:02d}.log"}
        state["commands"].append(row)
        save()
        start = perf_counter()
        with (args.output / row["log"]).open("wb") as log:
            process = subprocess.run(command["argv"], cwd=plan["cwd"], env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
        row.update(completed=True, exit_code=process.returncode, wall_seconds=perf_counter()-start)
        save()
        print(command["name"], "exit", process.returncode, flush=True)
        if process.returncode:
            state["completed"] = True
            save()
            raise SystemExit(1)
    state.update(completed=True, all_commands_exited_zero=True,
                 completed_at_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == "__main__":
    main()
