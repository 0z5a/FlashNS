# FlashNS

FlashNS 面向需要三阶空间导数的 Navier–Stokes PINN，以 FP64 Taylor jet、显式参数梯度、内点/边界混合布局和可选 CUDA 融合降低高阶自动微分开销。矩阵乘仍复用有竞争力的 PyTorch/cuBLAS 路径。

## 相对现有库的实测提升

H100 NVL 上，同一 Kovasznay 问题、相同初始化种子和六项精度门限，FlashNS Hopper TMA 的已收敛完整运行中位耗时为 **22.206 秒**：

| 对照路径 | 完整求解中位耗时 | 对照耗时 / FlashNS 耗时 |
| --- | ---: | ---: |
| PhysicsNeMo PhysicsInformer 自动微分 | 118.790 s | **5.35×** |
| PyTorch 嵌套坐标自动微分 | 96.660 s | **4.35×** |
| 编译版 PyTorch Taylor jet | 25.941 s | **1.17×** |
| cuEquivariance 多项式 jet | 25.794 s | **1.16×** |

每个后端三个种子。首轮固定预算为 17/24 收敛；统一增加 L-BFGS 迭代上限后达到 24/24，原失败记录及累计尝试成本仍保留。表中是达到原阈值的完整运行中位数之比，包含数据准备、后端/Graph 建立、优化、线搜索、停止检查和独立验收；不同路径的优化轨迹与评估次数可能不同。详见[完整求解报告](docs/pinn-solver-results.md)。

完整参数梯度的 Graph replay 另有独立对照：16,384 点时，F 为 1.832 ms，编译版 Torch jet 为 2.467 ms，cuEquivariance 为 2.193 ms，配对倍率分别为 **1.348×、1.197×**。另一张 H100 的 65,536 点实验中，F+ 为 4.282 ms，MathDx/cuBLASDx M64 适配器为 5.588 ms，配对倍率为 **1.305×**。这些是固定工作量的梯度计算，不含完整优化过程；MathDx 对比只覆盖已实现的有限 GEMM 配置。[原始证据与比较口径](docs/library-comparison.md)。

## 提升来自哪里

- Q10/Q20 阶乘归一化 Taylor jet 和显式 VJP，减少重复构建高阶坐标 AD 的工作。
- stable tanh 辅助量，保留饱和区微小导数；高精度参考和数值门限不随性能结果放宽。
- compact 布局：内点使用完整 jet，边界只算值。既定 2048+512 点问题的逻辑行数降低 18%；该数字不等于显存峰值或完整求解提升。
- dgrad/VJP、Hopper TMA、首层坐标归约及 VJP/wgrad 融合，可分别构建、验证和消融。

主要收益来自导数表示。对已有库 GEMM jet 基线 B1，Hopper TMA 的完整求解中位数仅从 22.528 s 降至 22.206 s，约 1.014×。SM120 的 G1/G2 两组完整求解各 12/12 收敛，但均未确认稳定速度收益，普通 Torch 路径保持默认，详见 [SM120 验证](docs/validation.md)。

## 运行与本次补交

```bash
uv sync --group dev
PYTHONPATH=src uv run pytest -q
```

科学库的历史 H100 环境与哈希固定依赖见 [H100 运行说明](experiments/cuda_jet_h100/README.md)。Euler 原始输入须另行按来源登记下载，来源缺失时相关测试会跳过。

本次保留 GitHub 已发布的 G1/G2 实现，补上本地缺失的 H100/Hopper/A4000 实验、八后端完整求解、Euler/仿射波/形式化来源适配、测试及历史结果。见[差异和补交清单](docs/source-sync-20260913.md)。本次执行 CPU/HOST 回归和证据检查，没有重新运行 GPU 性能测试。
