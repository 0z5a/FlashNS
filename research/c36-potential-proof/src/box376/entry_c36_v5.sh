#!/bin/bash
# Frozen C36 v5 budget-study entry (unchanged v2 driver and trainer), started by the batch script of its own allocation.
# Arguments: ROOT REFERENCE COMPILER_RUNTIME JOB NODE DEADLINE_UNIX LANE...,
# each lane GPU_UUID:LOCK_PATH:LOCK_INODE:SEED:ENGINE. FLASHNS_ACTIVATE names the
# FP64 PyTorch environment activation script.
set -euo pipefail
root=$1; reference=$2; runtime=$3; job=$4; node=$5; deadline=$6; shift 6
lanes=(); for lane in "$@"; do lanes+=(--lane "$lane"); done
source "${FLASHNS_ACTIVATE:?set FLASHNS_ACTIVATE to an environment activation script}" >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m box376.run_c36_v2 --root "$root" --protocol protocol-v5-budget-development-20261015.json \
  --reference "$reference" --compiler-runtime "$runtime" --job "$job" --node "$node" \
  --deadline-unix "$deadline" "${lanes[@]}"
