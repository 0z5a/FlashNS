#!/bin/bash
# Template: the submission script substitutes @ACCOUNT@ and @ROOT@ and records the rendered hash.
#SBATCH --account=@ACCOUNT@
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@600
#SBATCH --comment=flashns-c36-formal-tta-v2-20261021
set -euo pipefail
R=@ROOT@
P=${R%/*}
seed=$1
case "$seed" in 2013091829|1342776288|8730574|1790307490|2126574302) ;; *) exit 64 ;; esac
O="$R/seed$seed"
mkdir -p "$O/control"
exec 9>"$O/batch-once.lock"
flock -n 9 || exit 73
test ! -e "$O/control/attempt.json" || exit 73
printf '{"job":"%s","seed":%s,"started_utc":"%s"}\n' "$SLURM_JOB_ID" "$seed" "$(date -u +%FT%TZ)" > "$O/control/attempt.json"
exec > "$O/batch.log" 2>&1
owned_pid=
finish() {
  code=$?
  trap - EXIT TERM INT
  if test -n "$owned_pid"; then
    kill -TERM -- "-$owned_pid" 2>/dev/null || true
    wait "$owned_pid" 2>/dev/null || true
  fi
  PYTHONPATH="$R/packet/src" /usr/bin/python3 -m box376.finalize_c36_tta "$O" "$R/packet" "$SLURM_JOB_ID" "$code" > "$R/logs/finalize-$SLURM_JOB_ID.log" 2>&1 || code=90
  exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "$R"
sha256sum -c receipts/launch-files.sha256
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
module load GPU/buildenv-gcccuda/2026.03-cu13.0
export PYTHONPATH="$P/runtime/site:$R/packet/src" PATH="$P/runtime/site/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
scontrol show job "$SLURM_JOB_ID" -o
end=$(date -d "$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* EndTime=\([^ ]*\).*/\1/p')" +%s)
deadline=$((end-900))
remaining=$((deadline-$(date +%s)))
test "$remaining" -gt 13100
setsid timeout --signal=TERM --kill-after=120s "$remaining" \
  python -m box376.run_c36_tta_v2 --packet "$R/packet" --root "$O" --seed "$seed" --deadline-unix "$deadline" &
owned_pid=$!
wait "$owned_pid"
owned_pid=
