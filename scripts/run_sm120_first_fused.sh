#!/usr/bin/env bash
# Invoke only after this physical GPU has been allocated by its owner.
set -euo pipefail
if [[ $# != 2 || ! "$2" =~ ^GPU-[0-9a-fA-F-]{36}$ ]]; then
  echo 'Usage: FLASHNS_PYTHON=python bash scripts/run_sm120_first_fused.sh NEW_OUTPUT GPU_UUID' >&2
  exit 2
fi
flashns_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$flashns_root"
flashns_output="$1"
flashns_uuid="$2"
flashns_python="${FLASHNS_PYTHON:-python3}"
flashns_sanitizer="${FLASHNS_SANITIZER:-compute-sanitizer}"
[[ ! -e "$flashns_output" ]] || { echo 'Output already exists' >&2; exit 2; }
command -v nvcc >/dev/null
command -v "$flashns_sanitizer" >/dev/null
command -v timeout >/dev/null
command -v flock >/dev/null
exec 9>"/tmp/flashns-${flashns_uuid}.lock"
flock -n 9 || { echo 'FlashNS already owns this GPU lock' >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$flashns_uuid"
export PYTHONPATH="$flashns_root/src${FLASHNS_EXTRA_PYTHONPATH:+:$FLASHNS_EXTRA_PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Use NVML before a CUDA context exists; idle utilization alone is insufficient.
"$flashns_python" - "$flashns_uuid" <<'PY'
import csv, subprocess, sys
uuid = sys.argv[1]
known = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
if uuid not in [x.strip() for x in known]:
    raise SystemExit('GPU UUID is not present')
rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
if any(row and row[0].strip() == uuid for row in csv.reader(rows.splitlines())):
    raise SystemExit('GPU has an existing process; wait for the owner to release it')
PY
mkdir -p "$flashns_output"
flashns_output="$(cd "$flashns_output" && pwd)"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv > "$flashns_output/device.csv"
"$flashns_python" -c 'import torch; assert torch.cuda.get_device_capability() == (12, 0), "SM120 required"'
timeout 180 "$flashns_python" experiments/pinn_v8/build.py --output "$flashns_output/build" > "$flashns_output/build-command.log" 2>&1
timeout 600 "$flashns_python" experiments/pinn_v8/preflight.py --build "$flashns_output/build" --output "$flashns_output/v8-preflight.json" > "$flashns_output/v8-preflight.log" 2>&1
for flashns_tool in memcheck racecheck initcheck synccheck; do
  timeout 300 "$flashns_sanitizer" --tool "$flashns_tool" --error-exitcode 86 \
    "$flashns_python" experiments/pinn_v8/first_fused_preflight.py --build "$flashns_output/build" \
    --output "$flashns_output/${flashns_tool}.json" --kernel-only \
    > "$flashns_output/${flashns_tool}.log" 2>&1
done
"$flashns_python" - "$flashns_output" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
    report = json.loads((root / (tool + '.json')).read_text())
    log = (root / (tool + '.log')).read_text()
    footer = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
              if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors')
    assert report['completed'] and report['preflight_passed'] and footer in log, tool
PY
timeout 900 "$flashns_python" experiments/pinn_v8/first_fused_preflight.py --build "$flashns_output/build" \
  --output "$flashns_output/first-fused-preflight.json" > "$flashns_output/first-fused-preflight.log" 2>&1
timeout 900 "$flashns_python" experiments/pinn_v8/first_fused_screening.py --build "$flashns_output/build" \
  --preflight "$flashns_output/first-fused-preflight.json" --output "$flashns_output/first-fused-screening.json" \
  > "$flashns_output/first-fused-screening.log" 2>&1
echo "Completed SM120 first fusion correctness and screening: $flashns_output"
