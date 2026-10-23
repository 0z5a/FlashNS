#!/bin/bash
# Frozen C36 zero-gate folding GPU entry, started by the batch script of its own allocation.
# Arguments: ROOT JOB NODE DEADLINE_UNIX LANE, the lane GPU_UUID:LOCK_PATH:LOCK_INODE.
# The driver uses only the standard library; GPU children activate the recorded FP64 runtime.
set -euo pipefail
root=$1; job=$2; node=$3; deadline=$4; lane=$5
exec /usr/bin/python3 -I "$root/src/box376/run_zero_gate_gpu.py" --root "$root" \
  --protocol protocol-zero-gate-gpu-20261016.json --job "$job" --node "$node" \
  --deadline-unix "$deadline" --lane "$lane"
