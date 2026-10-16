"""Offline four-GPU finite batch with unconditional archival and owned cleanup."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(sys.argv[1])
    config = json.loads((root / "config.json").read_text())
    children = []
    result = {"state": "STARTING", "started": time.time(), "job": os.environ.get("SLURM_JOB_ID"),
        "node": os.uname().nodename, "actual_child_waits": []}

    def save(name, value):
        tmp = root / (name + ".tmp")
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(root / name)

    def terminate(signum, frame):
        raise RuntimeError("Batch signalled for closure: " + str(signum))
    signal.signal(signal.SIGUSR1, terminate)
    signal.signal(signal.SIGTERM, terminate)
    try:
        assert os.environ["SLURM_JOB_NAME"] == config["job_name"]
        for name, expected in config["ready_file_sha256"].items():
            assert sha(root / name) == expected, name
        for name, expected in config["runtime_critical_sha256"].items():
            assert sha(root / name) == expected, name
        assert shutil.disk_usage(root).free > 1024**3
        assert shutil.disk_usage(os.environ["NODE_LOCAL_SCRATCH"]).free > 1024**3
        assert Path(config["python_library"]).is_file()
        assert Path(config["nvcc"]).is_file()
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,utilization.gpu,driver_version", "--format=csv,noheader,nounits"], text=True)
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
        assert not apps.strip() and len(gpu.splitlines()) == 4
        locks = root / "gpu-lane-locks"
        locks.mkdir()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        lanes = []
        for line in gpu.splitlines():
            index, uuid, name, free, utilization, driver = [x.strip() for x in line.split(",")]
            assert "H100" in name and int(free) >= 50000 and int(utilization) == 0
            path = locks / (uuid + ".lock")
            with path.open("xb"):
                pass
            lanes.append({"GPU_index": int(index), "GPU_uuid": uuid, "node": result["node"],
                "boot_id": boot, "lock_path": str(path), "actual_lock_inode": path.stat().st_ino,
                "GPU_name": name, "driver": driver})
        lanes.sort(key=lambda x: x["GPU_index"])
        save("gpu-lanes.json", {"lanes": lanes, "job": result["job"], "gpu_before": gpu, "apps_before": apps})
        save("execution-binding.json", {"test_deadline_unix": result["started"] + 9000,
            "audit_and_archive_reserved_seconds": 1800, "desktop_connection_required": False})
        affinity = sorted(os.sched_getaffinity(0))
        result["allowed_CPU_affinity"] = affinity
        def child_affinity(index):
            subset = affinity[index::4]
            if subset:
                os.sched_setaffinity(0, subset)
        for index in range(4):
            command = ["python3", str(root / "h20_gpu_lane.py"), str(root), str(index)]
            with (root / ("lane" + str(index) + "-driver.stdout")).open("x") as out, (root / ("lane" + str(index) + "-driver.stderr")).open("x") as err:
                child = subprocess.Popen(command, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    start_new_session=True, preexec_fn=lambda i=index: child_affinity(i))
                children.append(child)
            save("lane" + str(index) + "-started.json", {"pid": child.pid, "command": command, "started": time.time()})
        for index, child in enumerate(children):
            rc = child.wait(timeout=max(1, result["started"] + 9300 - time.time()))
            result["actual_child_waits"].append({"lane": index, "pid": child.pid, "exit": rc, "joined": True})
            save("progress.json", result)
        assert all(row["exit"] == 0 for row in result["actual_child_waits"])
        env = os.environ.copy()
        for name in ["PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "JAX_PLATFORMS"]:
            env.pop(name, None)
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
            LD_LIBRARY_PATH=config["runtime_library_path"] + ":" + env.get("LD_LIBRARY_PATH", ""))
        with (root / "audit.stdout").open("x") as out, (root / "audit.stderr").open("x") as err:
            child = subprocess.Popen([str(root / "runtime-cu130/bin/python"), str(root / "audit_h20.py"), "--root", str(root)],
                stdout=out, stderr=err, env=env, start_new_session=True, stdin=subprocess.DEVNULL)
            children.append(child)
            rc = child.wait(timeout=600)
        result["actual_child_waits"].append({"role": "audit", "pid": child.pid, "exit": rc, "joined": True})
        assert rc == 0
        result["state"] = "EXECUTED_AND_AUDITED"
    except BaseException:
        result.update(state="FAILED_OR_INCOMPLETE", failure=traceback.format_exc())
    finally:
        # Ignore a second preemption signal while closing owned children/archive.
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        result["owned_children_joined"] = True
        result["finished_science"] = time.time()
        save("result-before-archive.json", result)
        try:
            durable = root / "durable"
            durable.mkdir()
            # Installed dependencies and source wheels are unchanged persistent
            # inputs, pinned in readiness manifests. Preserve every original
            # scientific output, source, checkpoint, log and generated cache.
            excluded = {"runtime-cu130", "H20-extra-wheels", "H20-base-wheels", "durable"}
            paths = {}
            for directory, dirs, files in os.walk(root):
                if Path(directory) == root:
                    dirs[:] = [name for name in dirs if name not in excluded]
                for filename in files:
                    p = Path(directory) / filename
                    if p.is_file() and not p.is_symlink():
                        paths[str(p.relative_to(root))] = p
            manifest = {name: sha(path) for name, path in sorted(paths.items())}
            save("archive-manifest.json", manifest)
            archive = durable / "complete-H20-H100-batch.tar.gz"
            with tarfile.open(archive, "w:gz") as tf:
                for name, path in sorted(paths.items()):
                    tf.add(path, arcname=name, recursive=False)
                tf.add(root / "archive-manifest.json", arcname="archive-manifest.json", recursive=False)
            with tarfile.open(archive) as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                assert len(members) == len(manifest) + 1
                for member in members:
                    h = hashlib.sha256()
                    with tf.extractfile(member) as stream:
                        for block in iter(lambda: stream.read(1048576), b""):
                            h.update(block)
                    expected = sha(root / "archive-manifest.json") if member.name == "archive-manifest.json" else manifest[member.name]
                    assert h.hexdigest() == expected, member.name
            result["archive"] = {"state": "PASS", "path": str(archive), "sha256": sha(archive),
                "bytes": archive.stat().st_size, "members_verified": len(members), "source_bytes_removed": False,
                "installed_dependency_inputs_preserved": True}
        except BaseException:
            result.update(state="FAILED_OR_INCOMPLETE", archive_failure=traceback.format_exc())
        result["Slurm_completion_is_scientific_pass"] = False
        result["closed_at"] = time.time()
        save("result.json", result)
    return int(result["state"] != "EXECUTED_AND_AUDITED")


if __name__ == "__main__":
    raise SystemExit(main())
