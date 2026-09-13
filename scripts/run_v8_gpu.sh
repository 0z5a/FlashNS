#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != "--pilot" ) ]]; then
  echo 'Usage: FLASHNS_PYTHON=/path/to/python bash scripts/run_v8_gpu.sh NEW_OUTPUT_DIRECTORY [--pilot]' >&2
  exit 2
fi
flashns_v8_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$flashns_v8_root"
flashns_v8_python="${FLASHNS_PYTHON:-python3}"
flashns_v8_output="$1"
if [[ -e "$flashns_v8_output" ]]; then
  echo 'Output must be a new directory; previous evidence will not be overwritten.' >&2
  exit 2
fi
command -v nvcc >/dev/null
command -v compute-sanitizer >/dev/null
export PYTHONPATH="$flashns_v8_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$flashns_v8_python" -c 'import torch, numpy, scipy, sympy, mpmath; assert torch.cuda.is_available(), "target CUDA GPU required"'
mkdir -p -- "$flashns_v8_output"
flashns_v8_output="$(cd -- "$flashns_v8_output" && pwd)"
"$flashns_v8_python" experiments/pinn_v8/build.py --output "$flashns_v8_output/build"
"$flashns_v8_python" experiments/pinn_v8/preflight.py --build "$flashns_v8_output/build" --output "$flashns_v8_output/preflight.json"
"$flashns_v8_python" experiments/pinn_v8/check_sanitizers.py --build "$flashns_v8_output/build" --output "$flashns_v8_output/sanitizers"
"$flashns_v8_python" experiments/pinn_v8/benchmark.py --build "$flashns_v8_output/build" --preflight "$flashns_v8_output/preflight.json" --output "$flashns_v8_output/benchmark.json" --trace
if [[ $# -eq 2 ]]; then
  "$flashns_v8_python" experiments/pinn_v8/run_suite.py --build "$flashns_v8_output/build" --preflight "$flashns_v8_output/preflight.json" --phase pilot --output "$flashns_v8_output/pilot"
fi
echo "v8 reports: $flashns_v8_output"
