"""One finite H100 lane; every attempt uses a fresh process and private GPU lock."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback


def main():
    root, index = Path(sys.argv[1]), int(sys.argv[2])
    config = json.loads((root / "config.json").read_text())
    plan = json.loads((root / "finite-plan.json").read_text())
    execution = json.loads((root / "execution-binding.json").read_text())
    config["test_deadline_unix"] = execution["test_deadline_unix"]
    binding = json.loads((root / "gpu-lanes.json").read_text())["lanes"][index]
    lane = root / ("lane" + str(index))
    lane.mkdir()
    children, rows = [], []
    report = {"state": "RUNNING", "GPU_uuid": binding["GPU_uuid"], "index": index,
        "started": time.time(), "rows": rows, "owned_children_joined": False}
    env = os.environ.copy()
    for name in ["PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "JAX_PLATFORMS"]:
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES=binding["GPU_uuid"], PYTHONDONTWRITEBYTECODE="1",
        LD_LIBRARY_PATH=config["runtime_library_path"] + ":" + env.get("LD_LIBRARY_PATH", ""),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MAX_JOBS="2",
        TORCHINDUCTOR_COMPILE_THREADS="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        TORCHINDUCTOR_CACHE_DIR=str(lane / "cache/inductor"), TRITON_CACHE_DIR=str(lane / "cache/triton"),
        JAX_COMPILATION_CACHE_DIR=str(lane / "cache/jax"), JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="-1", JAX_LOG_COMPILES="true", JAX_EXPLAIN_CACHE_MISSES="true",
        MPLCONFIGDIR=str(lane / "cache/matplotlib"), FLASHNS_CUDA_CACHE=str(root / "native-cache-cu130"),
        TORCH_CUDA_ARCH_LIST="9.0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    python = str(root / "runtime-cu130/bin/python")
    env["PATH"] = str(root / "runtime-cu130/bin") + ":" + env["PATH"]
    env.update(CUDA_HOME=config["cuda_home"], CC=config["CC"], CXX=config["CXX"])
    env["PATH"] = str(Path(config["nvcc"]).parent) + ":" + str(Path(config["CXX"]).parent) + ":" + env["PATH"]

    def save():
        temp = lane / "progress.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(lane / "progress.json")

    def terminate(signum, frame):
        raise RuntimeError("Finite lane signalled for closure: " + str(signum))
    signal.signal(signal.SIGTERM, terminate)

    def run(case):
        assert time.time() < config["test_deadline_unix"]
        name = case["name"]
        output = lane / (name + ".json")
        arguments = ["--backend", case["backend"], "--seed", str(case["seed"]), "--output", str(output)]
        if case.get("solve"):
            arguments += ["--solve"]
        if case.get("compiled"):
            arguments += ["--compiled"]
        if case["kind"] == "representation" and not case.get("solve"):
            arguments += ["--interior-count", str(case["points"]), "--samples", "7", "--replays", "100", "--warmup-seconds", "2"]
            if case.get("profile"):
                arguments += ["--profile"]
        command = [python, str(root / "h20_case.py"), "--root", str(root), "--kind", case["kind"], *arguments]
        row = {"name": name, "command": command, "case": case, "started": time.time()}
        rows.append(row)
        with (lane / (name + ".stdout")).open("x") as out, (lane / (name + ".stderr")).open("x") as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
            children.append(child)
            row["pid"] = child.pid
            save()
            try:
                rc = child.wait(timeout=min(case["timeout_seconds"], max(1, config["test_deadline_unix"] - time.time())))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                rc = 124
        row.update(exit=rc, finished=time.time(), process_wall_seconds=time.time() - row["started"], joined=True)
        result = json.loads(output.read_text()) if output.exists() else {}
        row["status"] = result.get("status", "NO_SCIENTIFIC_REPORT")
        row["converged"] = result.get("solve", {}).get("converged")
        row["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None
        save()
        return row, result

    lock = open(binding["lock_path"], "r+b")
    try:
        assert os.fstat(lock.fileno()).st_ino == binding["actual_lock_inode"]
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
        assert binding["GPU_uuid"] not in apps
        groups = plan["lanes"][str(index)]
        selection = {}
        for case in groups["qualification"]:
            row, result = run(case)
            if row["exit"] == 0 and result.get("status") == "passed":
                choice = {"compiled": bool(case.get("compiled")), "median_ms": result["median_ms"],
                    "case": case["name"], "report_sha256": row["output_sha256"]}
                backend = case["backend"]
                if backend not in selection or choice["median_ms"] < selection[backend]["median_ms"]:
                    selection[backend] = choice
        (lane / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        report["selection_sha256"] = hashlib.sha256((lane / "selection.json").read_bytes()).hexdigest()
        for original in groups["execution"]:
            case = dict(original)
            if case.get("requires_qualification", True) and case["backend"] not in selection:
                rows.append({"name": case["name"], "case": case, "status": "NOT_RUN", "reason": "No GPU-qualified variant"})
                save()
                continue
            if case["kind"] == "adapter":
                case["compiled"] = selection[case["backend"]]["compiled"]
            run(case)
        report["state"] = "FINITE_CASES_EXECUTED"
    except BaseException:
        report.update(state="FAILED_OR_INCOMPLETE", failure=traceback.format_exc())
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        report.update(owned_children_joined=True, finished=time.time())
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        report["original_GPU_lock_released"] = True
        save()
        (lane / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(report["state"] != "FINITE_CASES_EXECUTED")


if __name__ == "__main__":
    raise SystemExit(main())
