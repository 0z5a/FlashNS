#!/bin/bash
# Frozen C36 all-planes observation entry, started by the batch script of its own allocation.
# Arguments: ROOT CHECKPOINTS JOB NODE DEADLINE_UNIX LANE..., each lane GPU_UUID:LOCK_PATH:LOCK_INODE.
# FLASHNS_ACTIVATE names the FP64 PyTorch environment activation script.
set -euo pipefail
root=$1; checkpoints=$2; job=$3; node=$4; deadline=$5; shift 5
lanes=(); for lane in "$@"; do lanes+=(--lane "$lane"); done
source "${FLASHNS_ACTIVATE:?set FLASHNS_ACTIVATE to an environment activation script}" >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m box376.run_c36_observe --root "$root" --plan observe-plan-20261015.json \
  --checkpoints "$checkpoints" --job "$job" --node "$node" --deadline-unix "$deadline" "${lanes[@]}"
