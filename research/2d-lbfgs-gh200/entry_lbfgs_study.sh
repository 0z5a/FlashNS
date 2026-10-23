#!/bin/bash
# Frozen 2D L-BFGS GH200 study entry, started by the batch script of its own allocation.
# Arguments: ROOT JOB NODE DEADLINE_UNIX. ROOT holds source/ (pinned commit plus the two new
# files), lbfgs/ (this directory's files and manifest.json) and the caches; the runtime comes
# from the environment the batch script exported.
set -euo pipefail
root=$1; job=$2; node=$3; deadline=$4
exec python "$root/lbfgs/run_lbfgs_study.py" --root "$root" --plan plan-2d-lbfgs-gh200-20261023.json \
  --job "$job" --node "$node" --deadline-unix "$deadline" --python "$(command -v python)"
