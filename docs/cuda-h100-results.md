# H100：Graph、库基线与 FP64 jet 融合 A/B

日期：2026-09-09。**Graph、cuBLASLt、PINN 科学计算库、MathDx 与原始 Euler 4M 点 A/B 均已完成，正式结果已下载。** 实验先后使用三台设备；每张性能表的 A/B 均在同一设备上配对，不混合不同主机的原始耗时。

## 已得到的结论

在相同 `2–64–64–3` 网络、FP64、相同输入与参数、完整 loss 和参数梯度范围内，16,384 点的 Graph replay：融合 F 为 **1.832 ms**，cuEquivariance 为 **2.193 ms**，编译版 PyTorch jet 为 **2.467 ms**。配对加速比分别为 **1.197×、1.348×**。

直接用嵌套自动微分实现同一残差目标，PyTorch 和 PhysicsNeMo 分别需要 **28.001 ms、30.729 ms**，相对 F 的配对比为 **15.289×、16.761×**。这个差距同时包含导数表示和实现路径的变化，不能全部归因于 dgrad/VJP 融合。更直接的融合消融是 B1 对 F：16K 点约 **1.068×**，65K/262K 点约 **1.11×**。

小批量的主要改进来自 CUDA Graph。4,096 点 B1 从 eager **6.844 ms** 降至 replay **1.230 ms**；262,144 点则从 **19.941 ms** 降至 **19.030 ms**。cuBLASLt 有限候选搜索对完整 replay 的改善均不足 1%，目前不能宣称它带来了显著整步收益。

MathDx 在第二张 H100 上与重新测量的 B1/F/F+ 比较。65K 点 F+ 为 **4.282 ms**，MathDx 最快的 M tile 64 为 **5.588 ms**；262K 分别为 **17.187、22.308 ms**。本次有限 cuBLASDx 适配实现没有超过 F+。

Euler 在第三台 H100 NVL 上完成原始 400 万点、5 后端、3 轮比较。向量化稳定 GPU 路径为 **15.558 秒**，稳定 CPU 为 **41.918 秒**，配对加速比 **2.694×**；两条旧的直接公式路径未通过数值阈值，不计算有效加速比。

## 设备与测量范围

- 单卡 NVIDIA H100 80GB HBM3，compute capability 9.0，MIG disabled；实测功耗上限 **525 W**，未修改设备设置。
- Driver 595.71.05；PyTorch 2.11.0+cu128；CUDA 编译器 12.8.93；cuBLAS 12.8.4.1。原生融合使用 CUTLASS v4.7.1，commit `cb4247394dd82148787aed73e5dc7cef33cbf862`。
- CPU 为 Xeon Platinum 8568Y+；容器 CPU quota 约 23.04 核。PINN benchmark 使用 2 个 Torch CPU 线程，Euler 使用 4 个。跨机器 eager 时间包含不同主机的提交成本。
- 科学计算库：PhysicsNeMo 2.2.1；cuEquivariance、Torch adapter 与 CUDA ops 均为 0.11.1；NumPy 2.5.3、SciPy 1.18.1、SymPy 1.14.0。
- 新库位于隔离的 `.libdeps`，只安装公共 API 所需的依赖；没有替换基础 PyTorch。初始环境清单早于这些依赖安装，最终科学库版本与 wheel 哈希以 `scientific_benchmark.json` 为准。

上述设备信息对应第一台 H100（A，UUID `GPU-234a091d-e544-81aa-a6c8-44b77efbff23`）。MathDx 正式 A/B 使用第二台 H100（B，UUID `GPU-75655542-902a-e1a0-4829-5772a3bec9ee`），功耗上限 **599 W**、driver **610.57.04**；CPU 型号、CPU quota、Torch/CUDA/cuBLAS 版本相同。B 的 eager 主机提交成本更高，因此没有把两台设备的原始耗时合并为一组配对数据。B 上重新完成原生构建、58 项测试、数值验证和四种 Sanitizer。

Euler 的正式配对运行使用第三台 **H100 NVL**（UUID `GPU-83b4118c-a437-c4e5-e929-805a26db958f`），可见显存 95,830 MiB、功耗上限 **400 W**、driver **570.211.01**。CPU 为 Xeon Platinum **8562Y+**，容器 CPU quota 约 61.44 核；Euler 固定 4 个 Torch CPU 线程。基础镜像缺少 Torch，因此在项目 `.venv` 安装 Torch 2.11.0+cu128、NumPy 2.5.3、SciPy 1.18.1。没有在这台设备重测 PINN 或 MathDx，也没有跨设备计算加速比。

同一用例的后端顺序每轮旋转、隔轮反转；报告 15 轮的中位数。Graph/cuBLASLt 完整梯度每次计时内循环 5 次，科学库完整梯度内循环 3 次，局部算子均为 5 次。每次测量前同步，CUDA event 与同步 wall time 同时记录；倍率是各轮比值的中位数，不是两列中位数相除。

计时覆盖驻留 GPU 数据上的前向、残差 loss、loss 对 jet 的 seed、显式/自动参数反传，**不含**优化器、独立验证、编译、计划搜索和 Graph capture。它不等于到共同收敛标准的求解时间。不同实验表格是独立运行，应在同一表格内作配对比较。

## 固定科学目标与后端含义

PINN 使用二维稳态 NS 的两条动量方程和散度残差，黏性系数 `ν=0.07`；loss 还包含三条残差的一阶空间梯度平方，权重 `0.2`。网络为 `2–64–64–3`，每层使用同一 stable tanh；需要三阶空间导数和一阶参数梯度。完整 jet 为二维 Q=10，局部三维测试 Q=20。所有后端保持 FP64，没有改用低精度或删掉混合导数。

| 后端 | 实际实现 |
|---|---|
| B1 | Torch FP64 GEMM + stable 原生 jet 前向/独立 VJP |
| U | CUTLASS dgrad GEMM + 独立 stable VJP，作为相同主循环消融 |
| F | CUTLASS dgrad 与 stable jet VJP 融合，N tile 64 |
| F_n32 | 相同融合的 N tile 32 版本 |
| F3 | Cout=3 尾层特化，其余层采用 B1 |
| B1+ / F+ | 相应路径中的适用 GEMM 使用 Torch/cuBLASLt 有限计划搜索 |
| Torch nested AD | 直接通过嵌套 Torch AD 构造相同 NS 残差、空间梯度及参数梯度 |
| PhysicsNeMo AD | 公共 `PhysicsInformer` autodiff 构造 NS 残差；Torch AD 接续残差空间梯度与参数梯度 |
| cuEquivariance | 公共 `SegmentedPolynomial` 的 FP64 `uniform_1d` JIT 实现完整 jet 前向与显式 VJP；GEMM、loss、seed 与 B1 共享 |
| Torch jet eager / compiled | 相同 stable 张量 jet 前向和 VJP；compiled 用 Inductor `fullgraph=True, dynamic=True`，GEMM、loss、seed 与 B1 共享 |

PhysicsNeMo 2.x 路径以内联 SymPy PDE 定义同一方程，不假定旧版内置 NS 类仍存在。cuEquivariance 实际使用 `SegmentedPolynomialFromUniform1dJit`，适配器会拒绝 Naive fallback；这里没有使用其 autograd backward 替代显式 jet VJP。公共接口范围见 [PhysicsNeMo PhysicsInformer](https://docs.nvidia.com/physicsnemo/latest/user-guide/physics_addition.html) 和 [cuEquivariance polynomial tutorial](https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/poly.html)。

## 科学计算库完整参数梯度 A/B

以下均为 CUDA Graph replay，单位 ms。

| 后端 | 4,096 点 | 16,384 点 | 16K 配对时间 / F |
|---|---:|---:|---:|
| F | 1.210581 | 1.832256 | 1.000× |
| B1 | 1.234325 | 1.954325 | 1.068× |
| cuEquivariance | 1.331328 | 2.192768 | 1.197× |
| Torch jet compiled | 1.374528 | 2.466613 | 1.348× |
| Torch jet eager | 3.330709 | 8.404000 | 4.590× |
| Torch nested AD | 11.240907 | 28.000524 | 15.289× |
| PhysicsNeMo AD | 12.360235 | 30.729131 | 16.761× |

没有 Graph 时，16K 的 F、B1、cuEquivariance、Torch compiled jet 分别为 7.876、7.932、10.993、9.317 ms；嵌套 AD 和 PhysicsNeMo 为 39.722、44.429 ms。eager 受主机提交空隙影响，不能把这组时间当作纯内核耗时。

cuEquivariance 的局部 VJP 并不全面落后：二维 4K 的 replay 为 **0.02412 ms**，原生独立 VJP 为 **0.02639 ms**；16K 与原生接近。其前向较慢：二维 16K 为 **0.18662 ms**，原生为 **0.06212 ms**；三维 16K 分别为 **0.21889、0.11836 ms**。因此完整步骤的差距需要结合前向、反向和共享 GEMM 解读。

7 个后端均通过普通输入、加权目标、饱和尾部和 4 次输入/参数变化的 Graph replay 检查；3 个新增 jet 后端还各通过 44 个 100 位精度尾部诊断，覆盖二维/三维、次正规数和下溢。早期 Torch compile 因 Python 维数参数被符号化而失败；改为显式二维/三维入口后，`scientific_validation_v2.json` 与最终性能运行均通过。没有修改第三方库或放宽阈值。

科学库完整步骤只测试 4K/16K。7 个独立 Graph pool 同时存活时，16K 的 allocator reserved 约 **19.08 GiB**；更大批量的这套同时驻留测量会显著增加显存占用，未把 65K/262K 的原生结果冒充为所有库的大尺寸结果。

原始证据：[scientific_benchmark.json](../experiments/cuda_jet_h100/artifacts/scientific_benchmark.json)、[scientific_validation_v2.json](../experiments/cuda_jet_h100/artifacts/scientific_validation_v2.json)。

## Graph 与融合分开计量

| 点数 | B1 eager | B1 replay | F replay | F3 replay | B1/F replay 配对比 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 6.844371 | 1.230445 | 1.207238 | 1.207053 | 1.020× |
| 16,384 | 7.171213 | 1.949542 | 1.825491 | 1.873754 | 1.068× |
| 65,536 | 7.185606 | 4.761216 | 4.292999 | 4.493722 | 1.109× |
| 262,144 | 19.941408 | 19.029945 | 17.178081 | 17.951379 | 1.108× |

Graph 先在 side stream warmup，再捕获同一组静态地址；输入与参数原地更新。4 个后端各通过 4 次输入变化和 SGD 参数更新后的独立嵌套 AD 对照。capture 单独记录，没有在计时中重新捕获；管理原则遵循 [PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)。

Graph replay 的新增 allocated peak 接近零，**不代表 Graph 总显存很小**。262K 同时保留 4 个后端状态时 reserved 为 32.48 GiB；cuBLASLt 实验的 6 个后端约 44.80 GiB。测量脚本记录的是当时全部存活的缓存和 pool，不能直接解读为单一后端独占显存。

原始证据：[graph_experiment.json](../experiments/cuda_jet_h100/artifacts/graph_experiment.json)。

## cuBLASLt 强化基线 B1+

每个矩阵签名最多取 8 个 cuBLASLt heuristic 候选，workspace 上限 8 MiB，并把 Torch GEMM 放入候选集。所有候选先验证，再用 7 轮配对 Graph replay、每图 16 次 GEMM 选择；差异不足 2% 时保留 Torch。记录转置、stride、指针对齐、算法属性、workspace、版本与 GPU UUID。计划只在当前进程中缓存，不作为跨设备的通用 dispatch。

| 点数 | B1 replay | B1+ replay | F replay | F+ replay | B1+/F+ 配对比 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1.229075 | 1.226259 | 1.206227 | 1.202496 | 1.020× |
| 16,384 | 1.951424 | 1.945728 | 1.829766 | 1.823277 | 1.067× |
| 65,536 | 4.743078 | 4.728051 | 4.287782 | 4.267059 | 1.108× |
| 262,144 | 19.021594 | 18.946605 | 17.170898 | 17.122292 | 1.107× |

部分前向和 dgrad 选中 cuBLASLt，所有全局 wgrad 最终保留 Torch。选中的 Lt 计划所需 workspace 均为零；这不意味着 Torch 内部 workspace 为零。多个调用复用同一计划时必须串行执行或显式维护 stream 依赖。候选搜索是有限集合、热数据 Graph 测量，不证明已经找到全局最优算法。

实现遵循 [cuBLASLt matmul API](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasltmatmul) 的 FP64 compute、row-major 布局与对齐要求。初次构建的动态库链接参数失败，改为显式链接已安装的 `libcublasLt.so.12` 后成功；此构建失败没有混入性能数据。

原始证据：[baseline_plus.json](../experiments/cuda_jet_h100/artifacts/baseline_plus.json)、[baseline_plus_smoke.json](../experiments/cuda_jet_h100/artifacts/baseline_plus_smoke.json)。

## 原生 H0 验证与局部消融

原生 H0 的 58 项 CPU 测试、完整 jet 尾部高精度诊断、局部/完整梯度和 stream 检查通过；独立程序通过 138 个检查，memcheck、racecheck、initcheck、synccheck 全部通过。Nsight Compute 因 `ERR_NVGPUCTRPERM` 无法读取硬件计数器，所以没有声称实测 occupancy 或带宽瓶颈。

H0 局部 dgrad/VJP，B=16,384，单位 ms：

| 维数 / Cin / Cout | B1 | F | F_n32 或 F3 |
|---|---:|---:|---:|
| 2 / 32 / 32 | 0.0902 | 0.0680 | F_n32 0.0609 |
| 2 / 64 / 64 | 0.1621 | 0.1050 | F_n32 0.1347 |
| 2 / 64 / 3 | 0.1434 | 0.0788 | F3 0.0714 |
| 3 / 64 / 64 | 0.2969 | 0.1840 | F_n32 0.2506 |
| 3 / 64 / 3 | 0.2670 | 0.1374 | F3 0.1292 |

局部 Cout=3 收益较大，但在完整网络中 F3 仍慢于 F；局部最优不等于完整梯度最优。H0 eager 完整步骤没有稳定融合优势，后续 Graph 实验才分离出较小的设备收益。原始 H0 源码快照与实际构建动态库已保存在本地。

证据：[local_benchmark_gpu0.json](../experiments/cuda_jet_h100/artifacts/local_benchmark_gpu0.json)、[validation_gpu0.json](../experiments/cuda_jet_h100/artifacts/validation_gpu0.json)、[sanitizers.json](../experiments/cuda_jet_h100/artifacts/sanitizers.json)、[ncu.log](../experiments/cuda_jet_h100/artifacts/ncu.log)。

## MathDx / cuBLASDx 完整 A/B

MathDx SDK 26.06.1 / cuBLASDx 0.7.1 使用隔离的 CUDA 13.0.2 工具包，nvcc 13.0.88；CUDA redistributable 组件按官方 manifest 的 SHA-256 校验。SDK 自带 CUTLASS 4.5.2。GEMM 可用 header-only 模式，见 [cuBLASDx 安装文档](https://docs.nvidia.com/cuda/cublasdx/installation.html)。CUDA 13 仅生成 `sm_90` cubin，PyTorch 进程经 CUDA driver API 加载，没有把 CUDA 13 runtime 链入 Torch 12.8 进程。

适配器实现 M tile 32/64、C=32/64、K=2/3/32/64 的 16 个固定 FP64 GEMM，用于 forward/dgrad；全局 wgrad 保留 Torch。它是有限适配实现，不代表 MathDx 的全局调优上限。初次编译遗漏 `--expt-relaxed-constexpr`，编译器产生跨执行空间 constexpr 警告，生成代码在共享内存 tensor 首次写入时触发 trap。只补上该编译参数后，256 个独立矩阵用例和 memcheck 返回 0。

第一台实例随后断开。第二台 H100 上重新构建后，256 个独立用例、memcheck、racecheck、initcheck、synccheck 全部通过；racecheck 为 0 errors、0 warnings。Python 集成又通过 32 个随机矩阵/转置/producer-consumer stream 检查、两种 tile 的完整参数梯度、4 个饱和尾部网络以及每种 tile 各 4 次输入和 SGD 更新对照。

正式测量覆盖 16 个局部 dgrad/VJP 用例、4 个完整梯度点数，每组 15 轮配对、inner=5。以下为第二台 H100 的 Graph replay，单位 ms；每行的 B1、F、F+ 都在这台设备重新测量。

| 点数 | B1 | F | F+ | MathDx M32 | MathDx M64 | M64/F+ 配对比 |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 1.245837 | 1.220416 | 1.215968 | 1.326240 | 1.294470 | 1.064× |
| 16,384 | 1.961350 | 1.833766 | 1.834170 | 2.271430 | 2.168358 | 1.182× |
| 65,536 | 4.766202 | 4.312845 | 4.282233 | 6.007974 | 5.588179 | 1.305× |
| 262,144 | 19.114342 | 17.243321 | 17.186803 | 23.834099 | 22.307968 | 1.298× |

M64 在完整步骤中快于 M32，但仍慢于 B1/F。局部 Cout=3 时 MathDx 有时比 B1 稍快，依然没有超过融合路径。该结果只评价这里的静态共享内存 GEMM 适配器；没有搜索全部 block 大小、pipelined GEMM 或与激活共同融合的 MathDx 实现。

首次计时启动时，验证工件下载还未完成，因此该份 JSON 保留为 `mathdx_benchmark_first.json`，不用于正式性能结论。下载完成后，以相同代码和完整矩阵重新测量；主表来自 `mathdx_benchmark.json`。修正后实测 cubin SHA-256 为 `7d57e8b9a5d355a6feb34e4fcd1f5bb1a0284923b384197a5c8b961d803c022c`，验证和计时使用同一份二进制。

证据：[mathdx_benchmark.json](../experiments/cuda_jet_h100/artifacts_h100b/mathdx_benchmark.json)、[mathdx_validation.json](../experiments/cuda_jet_h100/artifacts_h100b/mathdx_validation.json)、[mathdx_sanitizers.json](../experiments/cuda_jet_h100/artifacts_h100b/mathdx_sanitizers.json)、[build_mathdx.json](../experiments/cuda_jet_h100/artifacts_h100b/build_mathdx.json)。

## Euler：原始 4M 点正式 A/B

使用上一轮四域各 100 万点的原始 NPZ，规范点集 SHA-256 为 `101bc2adbc7bb58624f0685f7a5efdeb7fa172ffe9857b43f2f6b47cc59a8110`。同 seed 在另一环境重新生成的一个域出现末位差异，因此正式 A/B 强制接受原始点集哈希。域为完整计算盒、物理半径 3 盒、轴和远场；半径 3 盒没有改称为球。

5 个后端均使用 FP64、批大小 4,096、相同官方冻结样条和相同四域输入，进行 3 轮旋转/反转顺序的配对测量。下表单位为秒，覆盖全部 400 万点残差求值；GPU 路径包含输入传输和所有输出返回 CPU。每轮结束后逐点核对全部三条残差，阈值固定为 `2e-9 + 2e-9 × abs(stable CPU reference)`。

| 后端 | 三轮中位数（秒） | 对稳定 CPU 的全量比较 | 配对时间 / 向量化 GPU |
|---|---:|---|---:|
| Torch stable vectorized GPU | 15.557548 | 通过 | 1.000× |
| Torch stable eager GPU | 32.883839 | 通过 | 2.112× |
| SciPy/NumPy stable CPU | 41.918246 | 参考实现；另有 20 点高精度核验 | 2.694× |
| SciPy direct CPU | 102.408521 | 失败 | — |
| Torch upstream GPU | 175.895502 | 失败 | — |

两条稳定 GPU 路径在所有三轮、所有点上的最大绝对差异均为 **3.553×10⁻¹⁴**，最大容差比为 **1.776×10⁻⁵**。SciPy direct 与上游 Torch 的最大容差比分别为 **522.45、1,264.06**；其耗时保留在原始结果中，有效倍率字段为 null。向量化 GPU 的三次耗时为 15.570108、15.557548、15.496224 秒；上述倍率按每轮时间比取中位数。

同一 NVL 环境另外完成 20 点的 80/100 位精度诊断，包含每域最差分歧点、固定样本和 4 个轴/近轴探针。稳定 CPU 的 20 点全部通过，最大绝对误差为 **3.587×10⁻¹⁴**，80/100 位参考也满足精度收敛阈值。直接 SciPy 和上游 Torch 分别有 3、8 个点未通过。这里的高精度核验仅覆盖列出的诊断点；400 万点逐点一致性和采样残差范数不构成连续域证明。

初始化单独记录：点集读取/哈希 0.372 秒、来源哈希 0.649 秒、JSON 解析及 SciPy 构造 8.447 秒、适配器及样条 GPU 传输 0.031 秒、5 后端小样本预热合计 1.718 秒。三次完整输出比较另耗 1.153、38.183、27.178 秒，不计入上表后端求值时间；包含初始化、15 次后端测量、比较和记录的整个 A/B 工作流为 **1,179.810 秒**。一次完整检查的总耗时需要另加相应初始化与核验成本，不能直接等同于表中的 15.558 秒。

此前两台实例断连只影响早期 Euler 预检，当前结论来自已下载的 NVL 正式 JSON。旧 H100 的 `torch.compile` 局部系数试验曾数值失败，最大容差比约 `2.499×10⁸`；上游 fullgraph 试验返回非零，其完整日志未能取回。这些编译试验不属于上表，也不用于有效性能结论。

证据：[euler_ab_4m.json](../artifacts/h100nvl/euler_ab_4m.json)、[euler_ab_smoke.json](../artifacts/h100nvl/euler_ab_smoke.json)、[euler_precision.json](../artifacts/h100nvl/euler_precision.json)、[NVL 环境](../artifacts/h100nvl/environment.json)、[实际安装版本](../artifacts/h100nvl/requirements_actual.txt)。

## 复现与边界

运行顺序与复现命令见 [H100 实验入口](../experiments/cuda_jet_h100/README.md)。[源码审计](../experiments/cuda_jet_h100/artifact_audit.json) 核对正式结果记录的源码字节及 Euler 固定来源工件；H0 保存测量时源码快照，历史文件按原 SHA-256 恢复并注明方法。新增文档、审计脚本与测量后修改不冒充测量时的源码。

第一台 H100 的原生二进制已下载；其 cuBLASLt 自有 wrapper 二进制和新库 wheel 缓存留在原实例，未能取回，构建参数、源码及已记录的哈希仍在。第二台 H100 的 native/Lt/MathDx 二进制、构建清单、验证日志和正式 MathDx JSON 已完整下载。源码审计不等于所有第三方运行时工件都已离线归档。

本地源码与结果归档（体积较大，未随 GitHub 源码发布） 包含 H100 实验源码、原始 4M 点集和已下载结果；[文件清单](../experiments/cuda_jet_h100/artifact_manifest.json) 记录每个文件的 SHA-256。官方样条缓存、第三方库与工具链按固定来源重新获取，未伪装为完整离线运行环境。

本报告对应的这轮实验没有执行多卡 H100、TMA 新流水线、任意宽度自动调优、HVP/double backward、共同收敛停止标准的求解比较、OpenAI 官方构造的 CPU/证明重放或区间证书。PINN 是独立工作负载，Euler 是官方冻结样条求值，两者结果分别解释。

2026-09-09 后续工作单独记录于 [Hopper 资源与 TMA 对照](cuda-hopper-results.md)、[官方 CPU 构建与 Comparator](openai-ns-formal-results.md) 和 [相同停止标准的完整求解](pinn-solver-results.md)。后续结果不替换本报告的原始测量值。
