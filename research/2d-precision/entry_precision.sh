#!/bin/bash
# Frozen 2D precision-study entry, started by the batch script of its own allocation.
# Arguments: ROOT JOB NODE DEADLINE_UNIX LANE..., each lane GPU_UUID:LOCK_PATH:LOCK_INODE.
# ROOT/tools/env.sh sets the shared runtime (read-only), CUDA, compilers and every cache under ROOT.
set -euo pipefail
root=$1; job=$2; node=$3; deadline=$4; shift 4
lanes=(); for lane in "$@"; do lanes+=(--lane "$lane"); done
source "$root/tools/env.sh"
exec "$RT/bin/python" "$root/precision/run_precision.py" --root "$root" --plan plan-2d-precision-20261016.json \
  --job "$job" --node "$node" --deadline-unix "$deadline" --python "$RT/bin/python" "${lanes[@]}"
