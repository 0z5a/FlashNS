# FlashNS v5：4×RTX A4000 首轮实现与实测

日期：2026-09-09。工作负载 ID：`pinn_jet_local`。代码与原始结果位于 `experiments/cuda_jet_v5/`；远端工作目录为 `/root/flashns-v5-20260909`。

这轮完成了 stable aux、N=32 和 K=3 候选、真实 native CUDA 参数梯度的 1/2/4 卡 sample-DP，以及强扩展、弱扩展和独立任务吞吐实验。固定全局 65,536 点，B1 完整梯度从单卡 **110.15 ms** 降到四卡 **32.32 ms，约 3.41×**。K=3 专用融合有明确的局部收益，但完整梯度只改善几个百分点。小 batch 跨轮波动明显，不能据此宣称稳定的弱扩展效率。

这是 v5 的 A4000 首轮交付。B1+ 库计划搜索、官方构造的 CPU 重放和局部数值适配、H100 路线仍未完成。所有计时都是独立 PINN 工作负载，未执行官方证明或科学收敛验收。

## 1. 实机与构建

| 项目 | 实测值 |
|---|---|
| GPU | 4×NVIDIA RTX A4000，每卡 16,376 MiB，sm86 |
| 驱动 / nvcc | 595.84 / 12.8.93 |
| Torch / CUDA runtime | 2.11.0+cu128 / 12.8.90 |
| cuBLAS / NCCL | 12.8.4.1 / 2.28.9 |
| Python / host C++ | 3.12.14 / GCC 13.3.0 |
| CUTLASS | v4.7.1，`cb4247394dd82148787aed73e5dc7cef33cbf862` |
| 精度与构建 | FP64，`-O3 -std=c++17 -arch=sm_86`，未启用 fast math |
| Python 补充依赖 | 隔离在 `.deps`：SciPy 1.18.1、pytest 9.1.1；NumPy 2.5.3 |

| 设备 | UUID |
|---|---|
| 0 | GPU-b0dd3c99-e896-4231-e044-feb228d8312b |
| 1 | GPU-05f77118-4e18-dea2-78fc-d40ad5397445 |
| 2 | GPU-70480540-f4f9-6c7a-23f4-790c840194f7 |
| 3 | GPU-dfbcbbb7-c51d-27d8-7da4-85959913a7ea |

拓扑中 GPU0/1/2 之间为 PIX，GPU3 到其余卡为 NODE；均在 NUMA0，CPU affinity 为 0–31。所有跨卡 P2P 能力查询通过。12 个有向卡对分别执行 64 MiB Torch `copy_` 加目标卡同步，约 **5.66–5.67 GB/s**。这些是该调用路径的实测带宽，不能仅凭能力查询推断 NCCL 实际传输路径。

原始环境：[初始清单](../experiments/cuda_jet_v5/artifacts/environment.json)、[最终依赖与主机状态](../experiments/cuda_jet_v5/artifacts/environment_final.json)、[构建命令及二进制哈希](../experiments/cuda_jet_v5/artifacts/build.json)。最终 `pip check` 通过。未修改驱动、功耗或时钟设置。

## 2. 数值修复与验收

完整 jet 仍采用 `[B,Q,C]` 和规范化 Taylor 系数 `D^alpha/alpha!`；二维 Q=10，三维 Q=20。每个隐藏层额外保存一个 FP64 `a1[B,C]`：

```text
r = exp(-abs(z0))
a1 = (2*r/(1+r*r))^2
```

前向高阶项使用稳定的 `a1`；VJP 的零阶导数系数从 aux 读取，其余系数由完整 H 的卷积恢复。这样在 FP64 `tanh(z0)` 已经舍入到 ±1 时，仍可保留可表示的导数。例如 z0=20，实测 `a1=1.6993417021166355e-17`，与高精度参考正确舍入值一致。

四张卡各自通过以下检查：

- 44 个高精度尾部配置：2 个空间维度 ×22 个中心，覆盖 0、±20、350、370、373、374、500、750 等；检查一至三阶前向系数，以及包含四阶标量导数的第三阶种子 VJP。参考用 mpmath 100 位精度，并增加微分工作精度。
- 24 个局部形状/偏移配置，比较 7 个后端与独立可微 Taylor 参考；另有两种维度的非默认 stream 检查，以及空 batch、非法 dtype/layout、未支持的 HVP 调用检查。
- 三个不同坐标尺度的 `2→7→7→3` 加权网络完整参数梯度，与独立嵌套空间 AD 比较。
- 两个隐藏偏置为 ±20 的 `2→2→3` 加权网络，逐参数检查尾部梯度。

常规比较阈值为 `2e-12 + 2e-11*abs(reference)`。高精度标量尾部使用 `1e-11*abs(correctly_rounded_reference) + 8 ulp`；尾部网络使用 `1e-10*abs(reference) + 8 ulp`。完整网络参考是独立嵌套 AD，其标量激活导数另外与高精度检查；没有把整个网络宣称为 mpmath 重算。

z0=373 的 a1 舍入到最小正 subnormal，z0=374 和 750 舍入到零。此处通过表示的是预先声明的**绝对量化误差**标准，不表示下溢后仍保持相对精度。普通范围的通过也不构成对任意输入、任意深度的误差保证。

独立 C++ smoke 加载正式计时使用的三个 `.so`，在每张卡上完成 138 项数值缓冲区检查，覆盖非默认 stream、输出 guard、alias 拒绝、尾块和重复 dgrad/tail 启动。GPU0 的 **memcheck、racecheck、initcheck、synccheck 全部通过**。本地和远端 pytest 均为 **58 passed**。

证据：[GPU0](../experiments/cuda_jet_v5/artifacts/validation_gpu0.json)、[GPU1](../experiments/cuda_jet_v5/artifacts/validation_gpu1.json)、[GPU2](../experiments/cuda_jet_v5/artifacts/validation_gpu2.json)、[GPU3](../experiments/cuda_jet_v5/artifacts/validation_gpu3.json)、[Sanitizer](../experiments/cuda_jet_v5/artifacts/sanitizers.json)。

## 3. 候选与局部归因

| 后端 | dgrad 与 VJP |
|---|---|
| B1 | 当前 Torch FP64 matmul，再执行 stable native VJP |
| U / F | 同一 N=64、M=64、K tile=16、3 stages 的 CUTLASS mainloop；分别分离/融合 VJP |
| U_n32 / F_n32 | 匹配的 N=32 变体，仍为 3 stages |
| U3 / F3 | Cout=3 时使用完全相同的三项点积，分别分离/融合 VJP；其它 Cout 回退 B1 |

**B1 尚未经过 B1+ cuBLASLt 计划搜索。** 不将“击败当前 B1”等同于击败最优库计划。U/F、U_n32/F_n32、U3/F3 的匹配比较用于区分融合收益和 GEMM/小 K 特化收益。

以下为 GPU0、B=16,384 的配对 CUDA event 中位数，单位 ms。每轮轮转并反转后端顺序，9 轮，每个样本内部 5 次；完整原始数据还包含 B=4,096。

| 维度 | Cin / Cout | B1 | U | F | U_n32 | F_n32 | U3 | F3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2D | 32 / 32 | 1.504 | 2.634 | 2.513 | 1.507 | 1.377 | — | — |
| 2D | 64 / 64 | 5.154 | 5.255 | 4.979 | 5.244 | 4.967 | — | — |
| 2D | 32 / 3 | 0.968 | 1.491 | 1.379 | 0.932 | 0.791 | 0.468 | 0.285 |
| 2D | 64 / 3 | 1.235 | 1.835 | 1.561 | 1.843 | 1.553 | 0.915 | 0.553 |
| 3D | 32 / 32 | 2.982 | 5.251 | 5.119 | 2.973 | 2.778 | — | — |
| 3D | 64 / 64 | 10.310 | 10.502 | 10.186 | 10.506 | 10.093 | — | — |
| 3D | 32 / 3 | 1.900 | 2.964 | 2.786 | 1.828 | 1.615 | 0.917 | 0.615 |
| 3D | 64 / 3 | 2.453 | 3.660 | 3.222 | 3.663 | 3.235 | 1.812 | 1.208 |

N=32 修复了 Cin=32 时 N=64 大量空列的成本，但 Cin=64 时没有明显额外收益。Cout=3 时，F3 相对 U3 的配对倍率约 **1.49–1.66×**；F3 相对 B1 约 **2.03–3.40×**，后者同时包含三项点积特化和融合，不能全部归给融合。

## 4. 单卡完整梯度与显存

完整目标是二维 `2→64→64→3` tanh 网络，包含稳态动量残差、散度和残差的一阶空间梯度；因此需要三阶空间 jet。黏性为 0.07，残差梯度权重为 0.2。性能点使用单位求积权重；协议正确性另外使用非均匀权重。具体公式和来源隔离见 [工作负载登记](../adapters/pinn_jet_local.json)。

以下 CUDA event 计时包含前向、残差种子 AD、全部参数梯度；不含 NCCL、优化器或独立数值对照。每轮内部 3 次、9 轮配对。初始化和 warmup 排除；原始结果同时记录同步 wall time。

| B | B1 ms | F ms | F_n32 ms | U3 ms | F3 ms | B1/F3 配对倍率 |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 23.317 | 21.906 | 22.763 | 23.213 | 22.722 | 1.045×，波动大 |
| 16,384 | 29.618 | 29.693 | 29.812 | 29.296 | 28.969 | 1.023× |
| 65,536 | 111.062 | 111.918 | 111.849 | 109.775 | 108.425 | 1.025× |

65,536 点的 B1 9 个样本范围为 110.426–111.455 ms，F3 为 107.711–108.743 ms。4,096 点中 B1 为 19.711–24.393 ms、F3 为 15.701–24.009 ms，不能用该行给出稳健的小 batch 结论。

| B | B1 峰值 allocated MiB | F 峰值 allocated MiB | F3 峰值 allocated MiB |
|---:|---:|---:|---:|
| 4,096 | 113.89 | 93.89 | 113.89 |
| 16,384 | 430.85 | 350.85 | 430.85 |
| 65,536 | 1,698.73 | 1,378.73 | 1,698.73 |

F 的全部隐藏层融合减少中间物化，在 65,536 点节省约 320 MiB，但这轮略慢于 B1。F3 仅特化输出尾层，完整梯度峰值仍由其它阶段决定，因此没有降低上述整体峰值。内存数值来自 Torch allocator，不含驱动上下文、库内部或其它进程；完整 reserved 和增量分配在原始结果中。

在单独的 Torch profiler 归因运行中，B1/U3 每步有 617 个 CUDA 事件，F3 为 616 个。局部移除一次 VJP 和中间张量后，其余前向、wgrad、残差与调度成本仍在。Profiler 数据不作为无插桩性能计时。

证据：[局部与完整梯度](../experiments/cuda_jet_v5/artifacts/local_benchmark_gpu0.json)、[独立 profile](../experiments/cuda_jet_v5/artifacts/profile_costs.json)。

## 5. 真实 CUDA sample-DP 协议

每个 rank 对自己的点运行上述 native 前向和参数 VJP。点范围为 `[floor(B*r/P), floor(B*(r+1)/P))`，局部损失都除以同一个全局求积权重和 Z，参数梯度使用 FP64 SUM。正则项仅在 rank0 加一次。空分片返回零局部贡献，仍进入全部 collective。

每步先检查并归约 finite 状态，再归约单个扁平参数缓冲区及 loss，最后按需执行 SGD。进入 NCCL 前显式同步 native 输出。默认网络有 4,547 个参数，梯度 collective 为 **36,376 B**，另有 4 B 的 finite MIN 和 8 B 的 loss SUM。

1/2/4 卡均通过 B1/U/F/F3 的 17 点不等长分片、3 点含空分片、非均匀权重、正则项和 3 步 SGD 检查；这些小案例的参考为独立嵌套 AD。大 batch 每步及计时后的完整 SGD 状态与单卡全局 B1 轨迹比较。受控弱扩展又验证了 25 步 SGD 后的参数一致性。

失败协议只注入过某 rank 的失败标志并检查全体一致决策；没有注入实际 GPU/NCCL 故障，也没有验证进程故障恢复。通信超时配置为 120 秒。

仅测 collective 的 max-rank wall 中位数如下；它不包含完整训练步骤里的到达等待和其它控制成本。

| FP64 payload | 1 卡 ms | 2 卡 ms | 4 卡 ms |
|---|---:|---:|---:|
| 36,376 B | 0.053 | 0.143 | 0.124 |
| 1 MiB | 0.053 | 0.313 | 0.865 |

## 6. 强扩展：固定全局点集

表中为 8 轮配对测量的 max-rank 同步 wall 中位数。计时包括本地完整梯度、打包、finite 检查、所有归约及可选 SGD；每步起始 barrier、初始化、warmup、独立验收不计入。1 卡也运行相同 distributed 控制协议，因此不能直接与上一节的本地 CUDA event 数值混算。

| 全局 B | 后端 | 操作 | 1 卡 ms | 2 卡 ms | 4 卡 ms | 1→4 倍率 |
|---:|---|---|---:|---:|---:|---:|
| 16,384 | B1 | 完整梯度 | 33.695 | 26.834 | 22.974 | 1.47× |
| 16,384 | F3 | 完整梯度 | 33.126 | 26.107 | 22.789 | 1.45× |
| 65,536 | B1 | 完整梯度 | 110.154 | 56.479 | 32.324 | 3.41× |
| 65,536 | F3 | 完整梯度 | 107.430 | 55.395 | 31.462 | 3.41× |
| 65,536 | B1 | 梯度＋SGD | 110.583 | 56.706 | 32.631 | 3.39× |
| 65,536 | F3 | 梯度＋SGD | 107.876 | 55.323 | 31.923 | 3.38× |

同一个强扩展案例在各 world size 使用完全相同的全局点集、权重和初始参数哈希。65,536 点的 B1 峰值 allocated 从单卡约 1,708 MiB 降至四卡各约 431–439 MiB；rank0 的参考及持有状态也会影响该统计。这是普通点数据并行的扩展收益，不是新的分布式算法或 CUDA 融合倍率。

原始结果：[1 卡](../experiments/cuda_jet_v5/artifacts/scaling_w1/rank0.json)、[2 卡](../experiments/cuda_jet_v5/artifacts/scaling_w2/rank0.json)、[4 卡](../experiments/cuda_jet_v5/artifacts/scaling_w4/rank0.json)。目录中另存每个 rank 的报告；rank0 汇总包含逐 rank、逐步的计算、状态同步、梯度/loss 归约、更新和内存数据。

## 7. 弱扩展与独立任务

固定每卡 4,096 点。首轮先测梯度、再测 SGD；B1 完整梯度在 1/2/4 卡分别为 12.875/23.292/21.635 ms，梯度＋SGD 为 26.245/24.080/24.033 ms。单卡的差异主要出现在本地计算阶段，不能将它解释为优化器自身成本。首轮弱扩展的参数初始化还随全局 batch 改变，因此保留原始记录，另做受控复测。

复测冻结同一组初始参数和嵌套的全局点集前缀，交错 B1/F3、梯度/SGD，5 步 warmup 后各 20 次，得到：

| 操作 | 1 卡 ms | 2 卡 ms | 4 卡 ms |
|---|---:|---:|---:|
| B1 梯度 | 27.197 | 23.882 | 21.860 |
| B1 梯度＋SGD | 27.513 | 24.952 | 21.843 |
| F3 梯度 | 26.876 | 24.690 | 21.137 |
| F3 梯度＋SGD | 27.320 | 23.738 | 21.216 |

交错后 SGD 增量较小，但同一单卡小 batch 跨轮仍出现 12.9–27.2 ms 的明显变化；复测单行本身也有宽范围，例如四卡 B1 梯度 18.57–26.32 ms。本轮未建立该变化的唯一原因，不用这些数据给出可靠的超线性或弱扩展效率结论。证据：[受控 1 卡](../experiments/cuda_jet_v5/artifacts/controlled_weak_w1/rank0.json)、[2 卡](../experiments/cuda_jet_v5/artifacts/controlled_weak_w2/rank0.json)、[4 卡](../experiments/cuda_jet_v5/artifacts/controlled_weak_w4/rank0.json)。

独立任务吞吐采用四个固定任务，每个任务 16,384 点。1/2/4 个常驻 worker 分别处理同一组四个任务，9 轮交替比较 B1/F3。下表计入父进程发令、四个完整梯度、CUDA 同步及响应 IPC，排除进程启动、输入准备、warmup 和独立检查；不执行 optimizer 或梯度归约。

| 后端 | 1 worker 完成四任务 ms | 2 workers ms | 4 workers ms | 1→4 吞吐倍率 |
|---|---:|---:|---:|---:|
| B1 | 112.234 | 57.094 | 37.197 | 3.02× |
| F3 | 109.431 | 55.745 | 33.719 | 3.25× |

这表示同时完成独立工作项的吞吐，不等于同一训练目标的加速。启动/warmup/检查耗时和任务输入哈希另外保存在 [原始吞吐报告](../experiments/cuda_jet_v5/artifacts/independent_throughput.json)。

## 8. 资源、来源与未测边界

| kernel | threads/CTA | 寄存器/线程 | shared/CTA | stack / spill |
|---|---:|---:|---:|---|
| N=64 U/F，2D/3D | 128 | 142 | 49,152 B | 0 / 0 |
| N=32 U/F，2D/3D | 64 | 138 | 36,864 B | 0 / 0 |
| U3 2D / F3 2D | 128 | 40 / 64 | 0 | 0 / 0 |
| U3 3D / F3 3D | 128 | 64 / 96 | 0 | 0 / 0 |

以上来自 ptxas 和 cuobjdump 静态检查。两个 dgrad 库的 SASS 都能看到 `DMMA.884`，但不据此推断峰值吞吐或其它 GPU 的性能。Nsight Compute 返回 `ERR_NVGPUCTRPERM`，硬件计数器和实测 occupancy 仍未取得。证据：[资源与 SASS 哈希](../experiments/cuda_jet_v5/artifacts/resources.json)、[计数器访问日志](../experiments/cuda_jet_v5/artifacts/ncu.log)。

新的 [OpenAI 官方仓库](https://github.com/openai/NavierStokesAndEuler) 独立登记为 `openai_ns_formal`，固定到 `8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538`。已保存八个入口/元数据文件及其字节哈希，工具链为 Lean 4.34.0-rc2。[来源登记](../sources/openai_ns_formal/registry.json) 和 [适配状态](../adapters/openai_ns_formal.json) 明确记录：未执行 Lake build、Comparator、官方构造的局部数值参考或 GPU 适配，`openai_proof_replayed=false`；论文文件的本地哈希仍为空。

本轮也未在 A4000 上重跑官方 Euler 冻结样条的 400 万点全链路。此前的 Euler/A5000 报告保持独立。C8/C13 压缩表示、B1+ 计划搜索、stage=2 消融、CUDA Graph、H100/TMA 和科学收敛均不在本轮已验证范围内。native 接口只支持手动的一阶参数 VJP，HVP/double backward 不支持。

源码哈希、构建哈希和历史测量源码已保存。项目目前没有独立实现 commit，登记为 null，使用源码归档和哈希标识。`artifact_audit.py` 检查了 22 份关键报告和 392 个源码哈希引用，全部匹配；它只检查工件完整性，不重新运行数值测试。[审计结果](../experiments/cuda_jet_v5/artifacts/artifact_audit.json)。

## 9. 下一步选择

1. 保留 stable aux 作为 v5 精度合同；F3 作为 Cout=3 的显式选项，F 作为显存取舍选项。当前不自动替换所有形状的 B1。
2. 在本机补 B1+ 有限库计划搜索，并针对 residual/host 调度做受控归因；新的 Graph 实验必须同时比较 B1/U/F，重新解释小 batch 波动。
3. 官方构造先完成固定版本的 CPU 构建/检查和局部可计算对象定义，再接入 GPU。H100 上单独构建、验证和建立本机基线。

实际运行入口和重放命令见 [v5 README](../experiments/cuda_jet_v5/README.md)。
