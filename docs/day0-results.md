# FlashNS 首轮实现与实测

2026-09-08。已建立出处固定、验收明确的三个入口：Boussinesq 局部仿射波、官方 Euler 冻结样条残差、用户提供的独立 PINN CUDA jet。CPU 自动测试 31 项通过。GPU 使用用户提供的 NVIDIA RTX A5000 24 GB，Torch 2.11.0+cu128、CUDA 12.8；本地 CPU 环境为 Python 3.12、Torch 2.14.0。

## 1. 局部仿射波

实现对应 [Alpöge–Buckmaster 论文](https://cims.nyu.edu/~tristanb/boussinesq.pdf) §3.1 的局部引理。原论文与下载字节数、哈希保存在 `sources/registry.json`。实现保留方程、外力、域和精度契约，不把该局部构造称为新的标准 NS 求解器。

从一般光滑原函数独立求导的五项符号恒等式通过：相位输运、散度、涡量、温度残差增量和涡量残差增量。两条 DOP853 轨迹分别与 80 位精度参考对照，最大归一化轨迹误差约 `2.44e-12`。七组场测试中六组完成求值，大动态范围一组按预先约定因自对流消去误差过大而拒绝。测试还覆盖时间依赖背景、有限波叠加的交叉残差、流函数截断引入的误差及极小幅度。

这是局部恒等式和有限输入回归，未重放全局多层构造或 Lean 证明。GPU 主机上的整个仿射波工作流约 1.38 秒，其中 FP64 ODE 约 0.0067 秒；这条微型工作流没有显示出单独移到 GPU 的必要。

证据：`artifacts/gpu/affine_cpu.json`、`adapters/boussinesq.json`、`cases/affine.json`。

## 2. 官方 Euler 冻结样条

从作者 [公告](https://anima-ai.org/2026/09/07/stable-singularity-of-the-euler-equations-on-r3-without-forcing/) 的直接链接取得 [代码归档](https://anima-ai.org/wp-content/uploads/2026/09/euler_code.zip)，实际包含两个 Python 脚本、三个九次样条及元数据。没有训练代码、网络权重、Arb 检查器或证书。ZIP 为 284,850,126 字节，SHA-256 为 `a238bc3662ec86ee61f7b904d365255d061b07dd9f1aa12a2a728af5959265ad`。

官方 U、Omega 为 `1601×3201`，Psi 为 `4801×4801`；原始数组合计 266,547,752 字节。原脚本的四域各 100 万点规模已执行，冻结同一组 NumPy PCG64 采样点供全部后端使用。点集与上游 Torch 随机序列不同；比较域和采样测度写入 JSON。

原始自动微分与 SciPy 直接样条求导并不总能满足预先固定的 `2e-9 + 2e-9*abs(reference)` 容差。近轴的高精度检查发现更大的消去误差。为此实现局部控制系数差分，在 FP64 中先对系数求导，再收缩基函数；每个样条和批次复用同一个局部系数块。

20 个独立诊断点使用 80/100 位精度局部样条值求导作参照。在 GPU 主机的 CPU 对照中，局部系数法最大绝对误差 `3.20e-14`，全部通过；SciPy 直接求导最大误差 `7.34e-5`，3 点失败；原始 Torch 最大误差 `3.02e-4`，8 点失败。旧路径失败记录保持可见，阈值没有放宽。

400 万点 eager 实测：

| 执行项 | 秒 |
|---|---:|
| 校验源文件哈希 | 0.570 |
| JSON 读取与 CPU 样条构建 | 10.901 |
| 上游适配器加载与设备传输 | 0.130 |
| SciPy 直接样条导数 | 100.500 |
| 上游 eager Torch / GPU | 150.546 |
| 局部系数 / CPU | 47.055 |
| 局部系数 eager / GPU，含返回 CPU | 57.804 |
| 四后端、比较与加载的总工作流 | 367.784 |

该次只执行一遍完整四后端基线，不能用作重复计时分布。局部系数 GPU/CPU 在全部 400 万点上通过，三条残差的最大绝对差为 `3.55e-14`。Torch 峰值分配 377,033,216 字节，约 360 MiB；不包含 CUDA 上下文和主机 JSON 解析内存。

单次“加载源文件后仅运行局部 CPU 路径”为 58.527 秒，局部 eager GPU 为 69.405 秒。若保留独立 CPU 逐点对照，就必须另外计入其约 47 秒，不能把 GPU 内核计时写成验证端到端时间。

证据：`artifacts/gpu/euler_cuda_4m_eager.json`、`euler_precision.json` 与 `euler_points_4m.npz`。

## 3. 样条编译实验与拒绝结果

向量化基函数递推已通过小样本数值验收。4,096 点的用户态 profile 仍记录约 2,963 次 CUDA kernel launch，启动开销和大量细粒度算子值得继续处理。

随后对同一 400 万点执行三次完整局部系数 CPU/GPU 交叉检查，全部通过：

| 重复 | CPU / 秒 | 向量化 GPU / 秒，含返回 CPU |
|---|---:|---:|
| 1 | 47.145 | 28.694 |
| 2 | 47.032 | 29.269 |
| 3 | 46.897 | 28.653 |

求值中位数为 CPU 47.032 秒、GPU 28.694 秒，约 1.64 倍。源文件检查、JSON 构建、设备传输和 GPU 首遍求值分项合计 40.410 秒；首遍再加独立 CPU 对照和最终比较，约 87.801 秒。三遍完整交叉检查的总工作流为 239.652 秒。这里没有把仍然必需的 CPU 检查隐藏在 GPU 加速比之外；1.64 倍只描述求值路径。

该报告使用 `--skip-legacy`，没有重复已执行的直接样条/上游自动微分基线；它们的字段为 null 或空列表，旧基线记录在上一节。证据为 `artifacts/gpu/euler_cuda_4m_vectorized.json`。

实际运行代码与输入配置快照为 `artifacts/gpu/euler_vectorized_source_snapshot.tar.gz`，已逐项核对报告中的代码哈希。快照保留运行当时的文件，后续报告元数据过滤规则的小改动不覆盖已记录的测量版本。

直接 fullgraph 编译原始上游检查器受基函数域检查的数据依赖控制流阻断。独立局部系数实现可以编译，但本次 Torch/Inductor 组合产生错误导数：在轴上一个探针，eq1 从参考约 `3.13e-7` 变为约 `0.499786`。因此约 5 毫秒的小样本编译后耗时被拒绝，不能列作有效加速。诊断中单独的基函数、索引和双精度双曲函数一致；导数组合发生偏差，尚未定位到最终编译器变换原因。

本地 Torch 2.14 CPU 编译的独立解析多项式小例子最大差约 `7.11e-15`；它不能替代 Torch 2.11/CUDA 的实际样条验收，也不足以确认框架版本是否是偏差原因。

完整 CLI 会在执行后逐点对照 CPU 并用失败退出码阻断错误结果；profile 工具也按数值验收返回退出码。报告中的编译失败、首次类型适配失败和后续数值失败均保留。

证据：`artifacts/gpu/profile_vectorized.json`、`profile_compiled_v2.json`、`profile_upstream_compiled_v2.json`、`compile_diagnostic.json`。

## 4. 独立 CUDA jet

用户补充的 CUDA 源码在 `experiments/cuda_jet/input/` 保持原样。本轮增加编译、stream 接口、CUDA/AD 对照、库 GEMM 的完整参数反传和 benchmark。它不是官方 Euler 训练工件。

nvcc 12.8 对 sm_86 编译成功，未启用 fast math。ptxas 记录二维 forward/VJP 分别使用 44/48 个寄存器，三维为 70/80；四个 kernel 的 stack frame、spill load/store 均为零。这些是编译器报告，不是实测 DRAM 或 occupancy 计数。

12 个激活测试案例覆盖二维/三维、不同规模、奇数 channel 和非默认 stream。三个 `2→8→8→3` 网络案例对照独立 nested AD，覆盖所有三阶输出 jet、含残差梯度的 loss 及全部参数梯度，均通过固定的 FP64 容差。没有把三阶空间导数的参数反传截断在三阶激活导数。

首次 warmed、7 次同步计时中位数如下；前向和 VJP 合并计时，单位毫秒：

| jet / B / C | eager Torch | compiled Torch | 原生 CUDA |
|---|---:|---:|---:|
| 2D / 4096 / 64 | 5.390 | 2.239 | 0.663 |
| 2D / 16384 / 64 | 17.200 | 9.565 | 2.986 |
| 3D / 4096 / 64 | 12.079 | 4.909 | 1.353 |
| 3D / 16384 / 64 | 39.919 | 23.266 | 7.259 |

首次原生 CUDA 对编译版 Torch 的局部倍率约 3.2–3.6。编译版各形状的首次调用额外约 5.3–13.6 秒，另列在 JSON，不计入 warmed 中位数。

在 `2→64→64→3` 的完整一次参数梯度步骤中，B=4096 的编译版/原生版分别为 26.761/23.172 毫秒；B=16384 为 126.342/111.579 毫秒。双方都使用同一库 GEMM、同一 loss 和输出 seed，包含 forward、dgrad、wgrad、bias 归约，不含优化器。约 1.13–1.16 倍的步骤加速明显小于激活局部收益；profile 中矩阵乘法占主要设备时间。进一步交替执行顺序的配对结果单列在最终 JSON。

第二轮增加 9 对交替执行顺序的计时，结果支持首轮观察：

| B | compiled Torch 中位数 / ms | 原生 CUDA 中位数 / ms | 配对倍率中位数 | 9 对范围 |
|---|---:|---:|---:|---:|
| 4096 | 26.750 | 23.157 | 1.1552× | 1.1547–1.1558× |
| 16384 | 126.349 | 111.575 | 1.1324× | 1.1320–1.1327× |

这是同一台卡、同一进程内的短期重复结果，尚未覆盖跨机器、跨日或科学训练的波动。第二轮激活局部倍率为 3.13–3.58×，相同编译强基线和数值对照均通过。

饱和区不通过：100 位精度诊断显示，z=15 的一阶导数相对误差约 `1.66e-4`；z=20 时原生内核给出 0，而参考约 `1.70e-17`。从已舍入的 H 重建 `1-H²` 丢失信息。该路径目前只对列出的普通输入通过，不能宣称鲁棒尾部支持。

证据：`experiments/cuda_jet/artifacts/cuda_a5000_v1.json`、`cuda_a5000.json`、`nvcc_build.log`。已保存原始输入和执行代码哈希。

## 5. 复现环境与接下来值得做的事

本地代码位于 `flashns/`；现有 GPU 主机上的部署目录为 `/root/flashns-day0-20260908`。远端使用已有 `/venv/main` 的 CUDA Torch；额外 CPU 依赖安装在项目 `.deps/`，没有修改驱动或平台服务。远端复现形式为：

```bash
cd /root/flashns-day0-20260908
PYTHONPATH=src:.deps /venv/main/bin/python -m flashns.cli affine
PYTHONPATH=src:.deps /venv/main/bin/python -m flashns.cli euler \
  --device cuda --points-file artifacts/gpu/euler_points_4m.npz \
  --torch-mode vectorized --skip-legacy --repeats 3
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet/run_cuda.py --benchmark
```

按现有证据，下一阶段应先定位样条导数组合的编译偏差；独立 PINN 线先增加稳定尾部导数 checkpoint，再评估矩阵乘法边界和参数反传融合。GEMM 融合、真实训练收敛、Arb/Lean 对接和 Rubin 路径均未完成，不属于这次有限输入验收的结论。现有单卡足够继续定位问题和测试原型，暂不需要增加 GPU 数量。
