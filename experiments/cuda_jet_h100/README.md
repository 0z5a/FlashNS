# H100 FP64 A/B

结果与测量边界见 [H100 报告](../../docs/cuda-h100-results.md)。这是独立 PINN jet 工作负载及原始 Euler 冻结样条的两组实验；没有把 PINN 的内核收益当作 Euler 求解收益。

`artifacts/` 保存第一张 H100 的结果（GPU UUID `234a091d…`，525 W）。第二张 H100（`75655542…`，599 W）的 MathDx 结果位于 `artifacts_h100b/`，环境及命令日志位于项目根 `artifacts/h100b/`。Euler 使用第三张 H100 NVL（`83b4118c…`，400 W），结果目录为项目根 `artifacts/h100nvl/`。跨设备的原始耗时分别报告，A/B 的分子与分母必须来自同一用例、同一次运行。

## 环境

测量基础环境为 Linux x86_64、Python 3.12、Torch 2.11.0+cu128、nvcc 12.8.93、cuBLAS 12.8.4.1。运行前读取所在机器的操作说明，并检查 GPU UUID、驱动、功耗上限和空闲状态。不要替换驱动或改变功耗配置。

从项目根执行；`/venv/main/bin/python` 是实测主机的 Python 路径：

```bash
/venv/main/bin/python -m pip install --target .deps numpy==2.5.3 scipy==1.18.1 pytest==9.1.1
git clone --depth 1 --branch v4.7.1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
PYTHONPATH=src:.deps /venv/main/bin/python scripts/preflight_multigpu.py --expect-gpus 1 --output artifacts/environment-new.json
PYTHONPATH=src:.deps /venv/main/bin/python -m pytest -q
```

构建脚本拒绝与实测 CUTLASS commit 不一致的源码。不要覆盖已经用于测量的构建清单或二进制；在新目录复现，再核对哈希。

## 原生、Graph 和 cuBLASLt

```bash
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet_h100/build.py
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet_h100/validate.py --device 0 --output experiments/cuda_jet_h100/artifacts/validation-new.json
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet_h100/check_sanitizers.py
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet_h100/benchmark.py --device 0 --repeats 15 --output experiments/cuda_jet_h100/artifacts/local-new.json
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet_h100/graph_experiment.py --repeats 15 --output experiments/cuda_jet_h100/artifacts/graph-new.json
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet_h100/build_blaslt.py
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet_h100/baseline_plus.py --repeats 15 --output experiments/cuda_jet_h100/artifacts/lt-new.json
```

Graph 保留静态输入、输出和 pool；输入/参数更新使用原地操作。GEMM 计划依赖形状、布局、对齐、设备和库版本；搜索位于计时外。计划 workspace 可复用，但并发调用需要明确的 stream 顺序。这里没有自动跨设备派发。

## PhysicsNeMo、cuEquivariance、PyTorch

```bash
/venv/main/bin/python -m pip install --target .libdeps --no-deps --require-hashes -r experiments/cuda_jet_h100/requirements-h100-libs.txt
PYTHONPATH=src:.libdeps:.deps /venv/main/bin/python experiments/cuda_jet_h100/scientific_benchmark.py --validate-only --output experiments/cuda_jet_h100/artifacts/science-validation-new.json
PYTHONPATH=src:.libdeps:.deps /venv/main/bin/python experiments/cuda_jet_h100/scientific_benchmark.py --batches 4096,16384 --repeats 15 --output experiments/cuda_jet_h100/artifacts/science-new.json
```

requirements 文件由实际下载的 wheel 名称和 SHA-256 生成，面向 CPython 3.12 / Linux x86_64。它是公共 API 实验所需的最小补充环境，依赖已有的 Torch、NumPy、SymPy 等基础包，不是完整 PhysicsNeMo 全功能安装。cuEquivariance 适配器拒绝 Naive fallback。

## MathDx / cuBLASDx

```bash
/venv/main/bin/python experiments/cuda_jet_h100/prepare_mathdx.py
/venv/main/bin/python experiments/cuda_jet_h100/build_mathdx.py
/venv/main/bin/python experiments/cuda_jet_h100/check_mathdx.py
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet_h100/mathdx_benchmark.py --validate-only --output experiments/cuda_jet_h100/artifacts/mathdx-validation-new.json
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet_h100/mathdx_benchmark.py --repeats 15 --output experiments/cuda_jet_h100/artifacts/mathdx-new.json
```

`prepare_mathdx.py` 仅写入项目 `third_party/mathdx_h100/`，下载 SDK 26.06.1 与 CUDA 13.0.2 的必要组件；CUDA 组件哈希与 NVIDIA manifest 对照。GEMM 使用 header-only cuBLASDx 0.7.1；构建必须保留 `--expt-relaxed-constexpr`。CUDA 13 生成 native cubin，由 Torch 的当前 stream 经 driver API 提交，运行时仍使用基础 Torch CUDA 12.8 环境。

适配器仅覆盖 16 个固定 GEMM 形状族，M tile 32/64；wgrad 和不支持的形状回退 Torch，并在报告中明确标出。先完成 256 个独立矩阵用例、四种 Sanitizer、Python 矩阵/完整梯度/Graph 更新验证，再运行计时。MathDx 与原生融合使用不同编译工具链，这不是单独改变库而保持编译器完全一致的消融。

## 原始 Euler：同一 4M 点 A/B

H100 NVL 的基础镜像没有 Torch，因此使用项目内 `.venv`；driver 为 570.211.01，Torch 为 2.11.0+cu128。只运行 Euler 无需构建 CUTLASS、cuBLASLt wrapper 或 MathDx。先在 Python 3.12 环境中安装：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install numpy==2.5.3 scipy==1.18.1 sympy==1.14.0 mpmath==1.3.0 pytest==9.1.1
PYTHONPATH=src .venv/bin/python -m flashns.cli fetch-sources
PYTHONPATH=src .venv/bin/python scripts/preflight_multigpu.py --expect-gpus 1 --output artifacts/euler-environment-new.json
PYTHONPATH=src .venv/bin/python -m flashns.cli diagnose-euler --points-file artifacts/euler_points.npz --output artifacts/euler-precision-new.json
PYTHONPATH=src .venv/bin/python experiments/cuda_jet_h100/euler_ab.py --points-file artifacts/euler_points.npz --repeats 1 --output artifacts/euler-smoke-new.json
PYTHONPATH=src .venv/bin/python experiments/cuda_jet_h100/euler_ab.py --points-file artifacts/gpu/euler_points_4m.npz --require-original-4m --repeats 3 --output artifacts/euler-4m-new.json
```

4M 输入必须使用上一轮保存的原始 NPZ；脚本强制核对点集规范哈希 `101bc2ad…a8110`。同 seed 重生成也可能产生末位不同的坐标，不能替代已冻结点集。5 个后端使用相同批大小 4,096，GPU 路径包含输入传输和输出返回 CPU；正式计时后逐点比较全部输出。失败路径的有效倍率为 null，不能仅凭速度字段判定后端可用。

`scripts/profile_euler.py --variant compiled` 与 `--variant upstream_compiled` 是另行记录失败的编译实验，默认数值阈值不变。首次编译成本、预热和 profiler 范围不与正式 A/B 混用。

## 文件与复现边界

- `common.py` 固定输入/参数 seed、NS loss、完整梯度与比较阈值；`gpu.py` 加载并验证原生库哈希。
- `graph_experiment.py`、`baseline_plus.py`、`scientific_benchmark.py`、`mathdx_benchmark.py` 分别记录输入与源码哈希、配对样本和显存范围。
- `artifacts/measured_h0_sources.tar.gz` 保留初次测量源码；`source_versions/` 中恢复的历史文件必须与原 JSON 的 SHA-256 完全相同，并记录恢复方法。
- 所有新性能运行使用新输出文件；不得将未完成的阶段标为通过。GPU 性能任务串行运行，编译、安装和下载先完成。
