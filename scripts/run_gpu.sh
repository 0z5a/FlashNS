#!/usr/bin/env bash
set -euo pipefail

flashns_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$flashns_root"
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'A CUDA-capable NVIDIA host is required.' >&2; exit 1; }
mkdir -p artifacts/gpu
nvidia-smi > artifacts/gpu/nvidia-smi.txt
uv sync --locked --extra euler --no-editable
uv run --locked --extra euler --no-editable python -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot access CUDA"'
uv run --locked --extra euler --no-editable flashns fetch-sources > artifacts/gpu/sources.json
uv run --locked --extra euler --no-editable pytest -q
uv run --locked --extra euler --no-editable flashns affine --output artifacts/gpu/affine_cpu.json
uv run --locked --extra euler --no-editable flashns diagnose-euler --output artifacts/gpu/euler_precision.json
uv run --locked --extra euler --no-editable flashns euler \
  --device cuda --points-file artifacts/euler_points.npz --repeats 3 \
  --output artifacts/gpu/euler_cuda_smoke.json
uv run --locked --extra euler --no-editable flashns euler \
  --device cuda --points-per-domain 1000000 --repeats 1 --batch-size 4096 \
  --save-points artifacts/gpu/euler_points_4m.npz \
  --output artifacts/gpu/euler_cuda_4m_eager.json
uv run --locked --extra euler --no-editable flashns euler \
  --device cuda --points-file artifacts/gpu/euler_points_4m.npz \
  --torch-mode vectorized --skip-legacy --repeats 3 --batch-size 4096 \
  --output artifacts/gpu/euler_cuda_4m_vectorized.json
