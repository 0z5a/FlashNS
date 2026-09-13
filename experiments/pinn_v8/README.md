# FlashNS v8 本地实现与 GPU 入口

2026-09-12：已新增可选 packed U3/F3 尾层路径，完成 SM120 编译与 59 项相关 CPU 回归；GPU 执行仍在排队。详见 [SM120 推进记录](../../docs/validation.md)。以下保留 2026-09-09 的原始阶段记录。

本目录实现 v8 的显式 residual/seed、边界 value-only 路径，以及混合阶 compact MLP。2026-09-09 已完成 CPU/HOST C++ 验收；新的 CUDA 源码尚未在目标 GPU 编译或执行。

实现与验收记录见 [本地落地说明](../../docs/FlashNS_v8_Local_Implementation_ZH.md)，数学来源见 [原始 v8 RFC](../../sources/flashns_v8_reference/FlashNS_Math_Dataflow_FA4_OpenAI_RFC_v8_ZH.md)。原始参考包逐文件保存，原报告不作为本次 GPU 结果。

## 本次实际验收

- 项目 `pytest`：80 passed，0 failed，0 skipped，其中新增 v8 测试 22 项。
- 原始 v8 参考程序：23/23 组，包括旧 residual、stable jet 和 HOST header 的集成检查。
- 新共享 residual/seed C++ 核心：27/27 用例。
- 固定 2,048 内点＋512 边界点、2→64→64→3 网络：dense/split/compact × autograd/explicit，共六种路径的损失及 4,547 个参数梯度与独立嵌套 AD 对照通过。
- 实际算子调用检查：dense/compact 各 8 次 GEMM，split 16 次；compact 全部 affine 角色使用相同紧凑行维度。

原始报告：[validation.json](artifacts/local-ready-20260909/validation.json)、[pytest 日志](artifacts/local-ready-20260909/pytest.log)、[23 组参考检查](artifacts/local-ready-20260909/reference.json)。验证时间不是性能基准。

## 在本地重跑

从 `flashns/` 根目录、已有 Python ≥3.11 环境运行。需要 NumPy、SciPy、SymPy、mpmath、PyTorch、pytest 和 C++17 编译器。

```bash
PYTHONPATH=src .venv/bin/python experiments/pinn_v8/validate_local.py \
  --output experiments/pinn_v8/artifacts/my-local-validation
```

输出目录必须尚不存在。实际依赖版本写入每份报告；本次是 Python 3.12.13、Torch 2.14.0（CPU），详见 JSON。无需修改现有 GPU 环境来匹配 Mac 的 Torch 版本。

## 上 GPU 后运行

使用目标主机已有 CUDA PyTorch 环境，要求 `nvcc` 与 `compute-sanitizer` 在 PATH 中。入口不安装包、不改驱动。解压后进入包内 `flashns/`：

```bash
FLASHNS_PYTHON=/path/to/cuda-env/bin/python \
  bash scripts/run_v8_gpu.sh experiments/pinn_v8/artifacts/gpu-first
```

顺序为：生成代码一致性检查 → 按目标架构构建 → 完整 preflight → memcheck/racecheck/initcheck/synccheck → 六种布局/seed 组合的 15 轮配对 Graph 测量 → 单独的 eager 阶段 trace。任一必要检查失败即停止；每次使用新目录。

完整 preflight 包含 Q10/Q20、非整块尺寸、非默认 stream、scalar stable-aux 尾部、显式 CUDA seed、Graph replay，以及四种 seed 模式的三次真实 Adam 更新对照。它不等同于完整收敛证明。trace 无法取得时明确记录缺失原因，不更改硬件权限。

加 `--pilot` 才继续运行两个预注册 pilot 初始化种子下的完整求解：

```bash
FLASHNS_PYTHON=/path/to/cuda-env/bin/python \
  bash scripts/run_v8_gpu.sh experiments/pinn_v8/artifacts/gpu-with-pilot --pilot
```

默认六组合为 dense/split/compact × autograd/CUDA seed，activation 均为新 stable CUDA 路径。compiled seed 可用 `benchmark.py --seed-modes autograd compiled cuda` 单独测量；tensor/compiled activation 尚未纳入 GPU preflight，因此禁止直接进入正式计时。

## 冻结候选后再做正式实验

先分析 pilot，确定最强适用基线与候选。以下是命令格式示例，具体选择必须由真实 pilot 支持：

```bash
export PYTHONPATH=src
python experiments/pinn_v8/freeze_selection.py \
  --pilot experiments/pinn_v8/artifacts/gpu-with-pilot/pilot/suite.json \
  --baseline dense:cuda --candidate compact:cuda \
  --rationale '填写依据：pilot 结果、适用基线比较与选择理由' \
  --output experiments/pinn_v8/artifacts/selection.json
python experiments/pinn_v8/run_suite.py \
  --build experiments/pinn_v8/artifacts/gpu-with-pilot/build \
  --preflight experiments/pinn_v8/artifacts/gpu-with-pilot/preflight.json \
  --phase formal --selection experiments/pinn_v8/artifacts/selection.json \
  --output experiments/pinn_v8/artifacts/formal
```

请将示例 `python` 替换为同一 CUDA 环境解释器。选择文件要求双方均有完整、成功的两个 pilot 观察；冻结源码、协议和理由。正式运行使用新种子 75201–75210；`--phase regression` 使用旧种子 75101–75103。所有运行先保存打乱顺序，再为每次求解创建独立进程与参数/优化器/Graph。保留失败记录、预算耗尽、checkpoint、源码哈希和每进程墙钟时间。

协议继承原扩展实验的 500 个 L-BFGS block 上限，六项停止指标及额外独立 AD 验收全部保留。不能使用旧代码默认 200 blocks 替代。库间总体加速结论还需要在目标环境重测此前的强基线；此目录内六组合只用于新机制归因。

## 结果边界

25600→20992 行是逻辑工作量减少 18%，不是实测加速。Graph 计时报告中的显存池数字是同时驻留多个候选的合计；独立 solve 的峰值按进程记录。尚无新硬件 counter、Sanitizer 通过结论、正式收敛统计或 GPU 加速比。

M4 的 epilogue/owner/TMA 融合需由 M3 的 GPU profile 决定。O1 目前保留原包的共享传播子与源项 CPU 参考检查；真实共享几何族的传播器、Duhamel 执行路径仍未接入。它与本目录 Kovasznay PINN 是不同工作负载。
