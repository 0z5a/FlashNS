# H100 NVL：Hopper 资源、TMA 与异步流水线对照

2026-09-09，在 H100 NVL 上完成了 48 个 FP64 配置的构建、验证、资源扫描与同步复制 / `cp.async` / TMA 对照。TMA 融合路径的局部收益明确，但新增的完整网络收益很小：相对 B1，完整参数梯度为 **1.023–1.094×**；相对既有融合 F，只增加约 **0.1%–0.3%**。这些是梯度步骤的结果；到统一停止标准的实际求解另见 [完整求解报告](pinn-solver-results.md)。

## 固定环境与比较对象

设备为 H100 NVL，UUID `GPU-83b4118c-a437-c4e5-e929-805a26db958f`，95,830 MiB、400 W，driver 570.211.01、nvcc 12.8.93、PyTorch 2.11.0+cu128。基线本机 CUDA 库和 cuBLASLt wrapper 均在这张卡上重新构建。编译目标 `sm_90`，全程 FP64、禁用 TF32、不启用 fast math。

目标仍是 `D @ W → stable tanh jet VJP`，二维 Q=10、三维 Q=20，阶乘归一化、cotangent、稳定 aux 与前一轮相同。B1 使用库 GEMM 加独立 VJP；original F 是既有 CUTLASS 融合。新增三种搬运路径使用完全相同的 FP64 `mma.sync.m8n8k4`、CTA 划分、shared layout 和 VJP 公式，每种均提供 U（独立 GEMM/VJP）与 F（融合）两个版本。

这里使用同步 FP64 MMA 搭配异步搬运，没有实现或声称 FP64 WGMMA。

## 有限扫描与同步约束

预先记录的 48 个配置为：M/N=32/64 × stage=1/2/3 × 三种复制，共 36 个；M=N=64 的 9 个 `__noinline__` epilogue 对照；另有 M=N=64、stage=2 的 3 个无 swizzle 对照。每个二进制含二维/三维 × U/F 四个特化，共 192 个 kernel。全部配置和 ptxas 日志保存在 [构建目录](../experiments/cuda_jet_hopper/artifacts/grid1/)。

每个 warp 计算 32×32 输出块，K tile=16。M 按 `floor(M/Q)` 分配完整样本；多余计算行不写回其他样本。输入支持 Cin/Cout=32/64，小 K 和完整网络 Cout=3 由 binding 显式回退；完整网络中的 U3/F3 尾层对三种复制路径相同。

同步复制为 16 字节 `double2`；`cp.async.cg` 使用组提交和等待；TMA 使用 2D A 描述符及拆分连续维度后的 3D B 描述符，无需额外打包 B。三种路径共享 128 字节 swizzle，stage 起点按 1024 字节对齐。

TMA barrier 在发布给异步代理之前初始化，每轮登记 A+B 的预期字节数，并等待对应 parity。CTA 在开始 MMA 前、复用流水线槽前均同步。所有搬运和同步 MMA 完成后，输入流水线的 shared union 才复用为完整 jet tile；完成 CTA 同步后才执行 VJP 并写全局输出。累加器限制在主循环作用域内，另以 noinline 对照检查局部变量生命周期的影响。

筛选在 B=16,384、C=64 上对全部配置各运行 3 轮，二维和三维均选中 `m64_n64_s2_c2_e0_w1`：M=N=64、stage=2、inline epilogue、有 swizzle。随后固定相同 M/N/stage/epilogue/layout 的同步和 `cp.async` 对照，使用新输入做 15 轮配对复测。表中结果来自复测，而非筛选最小值。

## 编译资源与验证

选中配置的 runtime resource / occupancy API 输出如下；四种维度与 U/F 特化在这些字段上相同。

| 复制路径 | 寄存器/线程 | 线程/CTA | shared/CTA | local bytes/线程 | 预测 CTA/SM | 预测 warp/SM |
|---|---:|---:|---:|---:|---:|---:|
| 同步 | 127 | 128 | 33,792 B | 0 | 4 | 16 |
| cp.async | 168 | 128 | 33,792 B | 0 | 3 | 12 |
| TMA | 116 | 128 | 33,792 B | 0 | 4 | 16 |

这些占用率来自 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 的预测。Nsight Compute 探针被宿主机以 `ERR_NVGPUCTRPERM` 拒绝，因此没有实测 occupancy、stall 或带宽计数器；未修改宿主机权限。SASS 中三个受控二进制各有 256 处 DMMA；cp.async 路径另有 112 处 LDGSTS，TMA 路径出现 `UTMALDG.2D` / `.3D`。

48 个配置均通过二维/三维、Cin/Cout=32/64、不同 batch/尾块、U/F、canary、非默认 stream、Graph 输入更新和小网络完整参数梯度检查。四种 Compute Sanitizer（memcheck、racecheck、initcheck、synccheck）对同一组已验证二进制均返回 0。另对选中三种 F 路径进行了 4 次实际 Graph 参数更新，与独立 nested AD 对照通过。初次开发验证脚本的整数边界断言修正和原始失败记录保留在 dev1 目录，不混入正式性能统计。

原始证据：[数值验证](../experiments/cuda_jet_hopper/artifacts/grid1/validation.json)、[Sanitizer 与 SASS 报告](../experiments/cuda_jet_hopper/artifacts/grid1/sanitizers/sanitizers.json)、[完整性能 JSON](../experiments/cuda_jet_hopper/artifacts/grid1/benchmark.json)。

## 局部结果

以下均为 Graph replay 的配对中位倍率，C=Cin=Cout=64；倍率大于 1 表示 TMA F 更快。完整 JSON 同时保留 C=32、eager、U/F、CUDA event、同步 wall、capture 成本和逐轮观测。

| 维度 | B | B1 / TMA F | 同步 F / TMA F | cp.async F / TMA F |
|---|---:|---:|---:|---:|
| 2D | 4,096 | 1.318× | 1.362× | 1.130× |
| 2D | 16,384 | 1.385× | 1.287× | 1.139× |
| 2D | 65,536 | 1.372× | 1.335× | 1.239× |
| 2D | 262,144 | 1.328× | 1.254× | 1.125× |
| 3D | 4,096 | 1.479× | 1.292× | 1.149× |
| 3D | 16,384 | 1.361× | 1.288× | 1.202× |
| 3D | 65,536 | 1.320× | 1.335× | 1.251× |
| 3D | 262,144 | 1.088× | 1.021× | 1.056× |

全部 16 个局部尺寸中，TMA F 相对 B1 为 1.088–1.479×；TMA U 相对 B1 均小于 1，说明单独更换搬运和主循环不能替代融合的收益。3D/C64/B262K 上 original F 为 1.125× B1，快于 TMA F，不能将 TMA 设为不分尺寸的普遍赢家。

## 完整参数梯度

网络为 2–64–64–3，包含加权 NS loss、坐标三阶导数、全部参数梯度及原生 Cout=3 尾层；不包含优化器。每个后端使用同样的 Graph 边界。

| B | B1 (ms) | original F (ms) | TMA F (ms) | 配对 B1 / TMA F |
|---|---:|---:|---:|---:|
| 4,096 | 1.3443 | 1.3161 | 1.3145 | 1.023× |
| 16,384 | 2.0311 | 1.9173 | 1.9129 | 1.062× |
| 65,536 | 4.6947 | 4.3041 | 4.2926 | 1.094× |

TMA F 对相同主循环同步 F 的完整梯度倍率只有 1.008–1.021×，对 cp.async F 为 1.005–1.010×。前向、wgrad、残差反传和尾层仍占据大部分工作；局部内核倍率不能直接当作训练或完整求解倍率。对 original F 的千分之几差异按测得的小收益报告，没有做显著性或普遍加速声明。

## 描述符、准备成本与复现

binding 以地址、形状、维度和设备为键，使用最多 16 项的 host plan 缓存，保留输入引用并记录 stream。新 CUDA Graph 内存池地址可能在首次 capture 时触发 host descriptor 编码，这部分归入 capture/setup，不属于 replay。二维复测累计 TMA plan 创建 344 次（其中 7 次发生在 capture）、host 计时 9.006 ms；该累计包含各尺寸及验证活动，不是每步固定开销。

源码入口：[hopper.cu](../experiments/cuda_jet_hopper/hopper.cu)、[binding](../experiments/cuda_jet_hopper/hopper.py)、[构建](../experiments/cuda_jet_hopper/build.py)、[验证](../experiments/cuda_jet_hopper/validate.py)、[性能](../experiments/cuda_jet_hopper/run_benchmark.py)。正式 GPU 性能阶段暂停了本任务自己的 Euler Comparator 进程组，结束后恢复；暂停记录在 [exclusive-benchmark.json](../experiments/cuda_jet_hopper/artifacts/grid1/exclusive-benchmark.json)。没有并行执行依赖下载或 CPU 编译。

```bash
python experiments/cuda_jet_hopper/build.py --output experiments/cuda_jet_hopper/artifacts/new-grid
python experiments/cuda_jet_hopper/validate.py --build experiments/cuda_jet_hopper/artifacts/new-grid --output experiments/cuda_jet_hopper/artifacts/new-grid/validation.json --exhaustive
python experiments/cuda_jet_hopper/check_sanitizers.py --build experiments/cuda_jet_hopper/artifacts/new-grid
python experiments/cuda_jet_hopper/run_benchmark.py --build experiments/cuda_jet_hopper/artifacts/new-grid --output experiments/cuda_jet_hopper/artifacts/new-grid/benchmark.json --repeats 15
```

需先按既有 [H100 运行说明](../experiments/cuda_jet_h100/README.md) 构建本机基线。脚本拒绝覆盖现有结果；不会因权限不足更改宿主机配置。
