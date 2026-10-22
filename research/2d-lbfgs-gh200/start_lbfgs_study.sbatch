#!/bin/bash
# Template: the submission script substitutes @ACCOUNT@, @ROOT@, @NAME@ and records the rendered hash.
#SBATCH --account=@ACCOUNT@
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --no-requeue
#SBATCH --job-name=@NAME@
#SBATCH --signal=B:TERM@300
#SBATCH --comment=flashns-2d-lbfgs-gh200-20261023
set -euo pipefail
R=@ROOT@
P=${R%/*}
exec 9>"$R/batch-once.lock"
flock -n 9 || exit 73
test ! -e "$R/lbfgs/control/attempt.json" || exit 73
mkdir -p "$R/lbfgs/control"
printf '{"job":"%s","started_utc":"%s"}\n' "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "$R/lbfgs/control/attempt.json"
exec > "$R/batch.log" 2>&1
cd "$R"
sha256sum -c lbfgs/launch-files.sha256
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
module load GPU/buildenv-gcccuda/2026.03-cu13.0
export PYTHONPATH="$P/runtime/site" PATH="$P/runtime/site/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
end=$(date -d "$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* EndTime=\([^ ]*\).*/\1/p')" +%s)
deadline=$((end-600))
node=$(hostname)
setsid timeout --signal=TERM --kill-after=60s $((deadline-$(date +%s))) \
  bash "$R/lbfgs/entry_lbfgs_study.sh" "$R" "$SLURM_JOB_ID" "$node" "$deadline" &
pid=$!
trap 'kill -TERM -- "-$pid" 2>/dev/null || true' TERM INT
wait "$pid"
