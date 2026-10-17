#!/bin/bash
# Attempt-2 compiled Ethier external-baseline entry, started by the batch script of its own allocation.
# Arguments: ROOT JOB NODE DEADLINE_UNIX LANE0..LANE3, each lane GPU_UUID:LOCK_PATH:LOCK_INODE.
# FLASHNS_ENV names the runtime environment script (Torch 2.12 runtime, CUDA, compilers).
set -euo pipefail
root=$1; job=$2; node=$3; deadline=$4; shift 4
source "${FLASHNS_ENV:?set FLASHNS_ENV to the runtime environment script}" >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1
lanes=(); for lane in "$@"; do lanes+=(--lane "$lane"); done
exec "$RT/bin/python" "$root/run_ext3.py" --root "$root" --job "$job" --node "$node" --deadline-unix "$deadline" "${lanes[@]}"
