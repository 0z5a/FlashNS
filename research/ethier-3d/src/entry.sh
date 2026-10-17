#!/bin/bash
# Frozen Ethier v4 entry for an existing allocation step (srun --overlap).
# Arguments: ROOT JOB NODE LANE0 LANE1, each lane GPU_UUID:LOCK_PATH:LOCK_INODE.
set -euo pipefail
root=$1; job=$2; node=$3; lane0=$4; lane1=$5
# Activate the FP64 PyTorch environment; the executed script sourced a site-specific activation file.
source "${FLASHNS_ACTIVATE:?set FLASHNS_ACTIVATE to an environment activation script}" >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1
exec python "$root/run_v4.py" --root "$root" --job "$job" --node "$node" --lane "$lane0" --lane "$lane1"
