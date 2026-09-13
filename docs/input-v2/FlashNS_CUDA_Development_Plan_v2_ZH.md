# FlashNS CUDA 开发方案 v2：参考 FlashAttention 1–4 的渐进式演进

**更新：2026-09-08｜文档版本：v2｜工程版本号均为建议，不表示已经发布。**

**状态：设计、依赖清单和开发辅助脚本。尚未实现或运行本文提出的融合 CUDA 后端，没有 GPU 性能结果，也没有 Rubin 实测。**

本稿继承已有 `CUDA_Design_ZH.md` 的高阶 jet＋参数 VJP 路线，新增渐进里程碑、依赖分层、代数合作者接口和可复现要求。旧稿保存在 `reference/CUDA_Design_v1_ZH.md`。不把前文的数学研究消息当成已经可移植的 GPU 工件；真正的科学工作负载仍须固定公开代码、版本和验收标准。

> **开发原则：先完成一条正确、可测、可替换的执行路径；每轮只验证一个主要性能假设。第一版不承担“通用 AD 编译器＋完整求解器＋自研高精度 GEMM＋Rubin 专用后端”。**

## 0. 本轮决策

首个目标仍是高精度坐标网络中的：

```text
空间 jet forward → PDE residual / residual gradient → loss → 参数 VJP
```

但首个自定义融合只选择：

```text
D_l @ W_l → 上一层激活的 jet VJP → D_(l-1)
```

这里不物化中间 `barH_(l-1)`；暂时保留 `D_(l-1)`，交给下一层 dgrad 和独立 wgrad 使用。原生 FP64、完整 jet 和强 packed-GEMM 基线先保留。

**13 分量代数压缩不阻塞上述 CUDA 工作。** 合作者可以在独立分支完成数学表示和对偶规则，再通过同一接口接入。异步流水线、精度仿真、Rubin、多卡和完整验证器都不是首版前置条件。

这是一条研究路线，不是性能承诺；单个融合若抵不过库 GEMM 的损失，应回退并重新选瓶颈。

## 1. 从 FA1–FA4 学“瓶颈迁移”，不是照抄四个功能包

| 公开工作 | 该代主要处理的问题 | 对 FlashNS 的启发 | 不能直接类比的部分 |
|---|---|---|---|
| FA1，2022 | 以分块和重计算减少 HBM 与片上存储之间的 I/O。[R01] | 先消除明确不必物化的微分中间量 | FlashNS 当前不是注意力的二次空间问题，不能直接宣称获得相同的渐近复杂度改善 |
| FA2，2023 | 改善 block/warp 划分，减少非矩阵工作及共享内存通信。[R02] | 前向、dgrad、wgrad 分别选 tile；测量 jet fragment 重排 | “算子更少”不等于更高利用率，也不是所有维度都应按同一种方式切 |
| FA3，2024 | 针对 Hopper，用异步矩阵计算、数据搬运与 warp specialization 重叠不同工作。[R03] | 在既有正确实现上增加架构特化流水线 | FP16/BF16/FP8 attention 的具体指令和近似策略不能直接搬到 FP64 jet |
| FA4，2026 | 面向 Blackwell 的非对称硬件增长，联合调整流水线、非矩阵工作和反向执行，并采用 CuTe DSL。[R04] | 后期再处理硬件改变后暴露的新瓶颈 | 不要因目标硬件更新就默认所有模块必须重写或降精度 |

上述是对已公开技术贡献的归纳，不是关于作者预先制定了四代路线图的推断。FlashNS 不需要机械产生四篇论文；每个独立研究主张都需要新的证据。

**渐进不等于第一版只有 forward。** 首版功能范围可以窄，但必须有完整参数梯度、强基线和明确失败边界。后续版本优化同一语义，才有可比较的演进曲线。

## 2. 建议里程碑：小版本完整，后续阶段有条件启动

| 工程里程碑 | 唯一主要问题 | 交付与通过条件 | 暂不纳入 |
|---|---|---|---|
| v0.1 Reference | 当前工作到底花时间在哪里？ | 完整 FP64 jet＋参数 VJP；packed GEMM；固定 loss；误差、内存与阶段耗时 | 自研 GEMM、架构专用指令、精度改变 |
| v0.2 IO | 不物化 `barH` 是否改善完整梯度？ | 一个 dgrad＋jet-VJP 融合；保留 D 和独立 wgrad；与相同数学算法对照 | 全网络 persistent、多消费者全融合 |
| v0.3 Layout | 融合后是否被片上交换或资源占用限制？ | 有限 tile/分组搜索；实测寄存器和共享内存；首尾层特化；可插拔 C13 实验 | 通用编译器、未验证的代数自动搜索 |
| v0.4 Async | 计算、搬运、局部 jet 算术能否重叠？ | 对有资源的目标卡做能力门控后端；覆盖异步错误和回退 | 无条件复用 attention 的低精度指令 |
| v0.5 Co-design | 新硬件上的主瓶颈是否已改变？ | 阶段特定的精度/数据流比较；可选 MathDx、CuTe DSL、Rubin 后端 | 未获实机却填入推测性能 |

版本顺序是组织开发用的建议，不是必须完成的流水线。v0.3 可以先发布布局优化；C13 若未通过梯度与数值测试，保持实验状态。v0.4/v0.5 只在 profile 和硬件条件支持时启动。

**第一篇论文的候选冻结点是 v0.2–v0.3 形成完整证据之后，而不是“等 v0.5 做完”。** 但一个微小融合或单形状快一点，并不自动足以投稿。

## 3. v0.1：把语义与基线固定下来

### 3.1 首个受支持子集

建议先支持仿射层＋tanh、二维/三维坐标、最高三阶空间导数、一阶参数梯度、逐点可分的固定目标。先做小网络和宽度 64；其他宽度逐步扩展。

不默认支持 BatchNorm、注意力、非局部积分损失、任意坐标变换、double backward、参数 HVP 或完整 residual Jacobian。不支持的变换必须显式报错，或走已经验证的参考路径。[R19]

如果真实科学工作负载需要 HVP 或特定优化器，必须先满足该接口，不能换成更容易融合的优化器后称为等价加速。

### 3.2 全 jet 的系数约定

采用阶乘归一化：

\[
H_\alpha=\frac{\partial^\alpha h}{\alpha!}.
\]

二维三阶有 Q=10，三维三阶有 Q=20。进入 PDE 之前恢复原始导数；返回 seed 时使用相应线性恢复映射的转置。不能只在 forward 乘阶乘。

对线性层：

\[
Z_\alpha=H_{\mathrm{prev},\alpha}W^T+\mathbf1_{\alpha=0}b.
\]

对 tanh，可在截断多项式代数内构造 `G = 1 - H*H`，局部伴随为：

\[
\bar Z_\beta=\sum_{\alpha\ge\beta}\bar H_\alpha G_{\alpha-\beta}.
\]

这是既有微分规则的专门实现，不是本项目声称的新定理。高阶 Taylor 压缩、向下闭合多指标和参数反传已有直接近邻。[R20,R21]

三阶空间导数的参数反传一般涉及四阶标量激活导数。不能在 `phi'''` 截断，也不意味着必须保存完整四阶空间张量。

### 3.3 三类 GEMM 与 bias

令 D 为对本层预激活 jet 的梯度，HBM 布局为 `[B,Q,C]`，C 连续。

| 操作 | 表达式 | 归约维 |
|---|---|---|
| forward | `Hprev.reshape(BQ,Cin) @ W.T` | Cin |
| dgrad | `D.reshape(BQ,Cout) @ W` | Cout |
| wgrad | `D.reshape(BQ,Cout).T @ Hprev.reshape(BQ,Cin)` | BQ |
| bias | `D[:,0,:].sum(dim=0)` | B，只有零阶通道 |

转置优先作为库的视图/矩阵描述表达，不先构造整个转置 tensor。jet 维不是新增的训练 batch，不能额外除以 Q。

bias 的零阶伴随已经包含高阶 loss 经激活 VJP 汇入的贡献；不能把高阶目标对 bias 的梯度置零。

### 3.4 基线分开回答两个问题

B0：独立嵌套 AD，主要用于小规模语义交叉检查。

B1：采用同一 jet 算法的 packed 库 GEMM＋独立局部激活/VJP，是 CUDA 融合的主要性能基线。

B2：在目标算子确实受支持时，加入 Collapsed Taylor/torch-jet 等算法基线；明确覆盖范围。[R20]

不能只报告比 eager 嵌套 AD 快多少，而避开 B1。Fortran/CPU 实现可用于算法参考，GPU 对单核 CPU 的倍率不能证明 CUDA 调度新意。[R21]

## 4. v0.2：首个融合 kernel 的完整合同

### 4.1 拟定接口（本包没有提供此 kernel 实现）

```text
dgrad_jet_vjp_fp64(
    D_l:       [B,Q,Cout],
    W_l:       [Cout,Cin],
    Hprev:     [B,Q,Cin],
    JetSpec,
    workspace,
    stream
) -> Dprev:    [B,Q,Cin]
```

函数计算 `barHprev = D_l @ W_l`，随后执行上一层激活 VJP，返回 `Dprev`。首版为 tanh 特化；对其输出重建导数的数值策略单独记录。

输入必须有明确的 dtype、stride、对齐和设备约束。默认不允许输入输出 alias；只有完成活跃区间证明及测试后才开放原地重用。

### 4.2 执行分层

```text
1. CTA 载入 D_l 和 W_l 的对应 tiles。
2. 完成 Cout 归约，得到该输出 tile 的完整 barHprev。
3. 将 GEMM accumulator 的分布转换成 jet 运算所需的分布。
4. 读取 Hprev，计算 tanh 导数 jet 或其合法重建。
5. 生成 Dprev，写回 HBM。
6. 独立 wgrad 读取该层 D 和 H；通过执行依赖保证 buffer 生命周期。
```

不能在 K 归约或 split-K 合并未完成时执行非线性 VJP。普通 GEMM epilogue 的逐元素模型也不能自动处理跨 jet 系数依赖。[R06]

### 4.3 保留 D 是一个设计选择，不是遗漏

D 有两个消费者：下一步 dgrad 和本层 wgrad。消除 D 可能要求重复局部 VJP、复杂片上双计算，或写出大量 partial dW。

首版选择“一次写回，两次高效消费”。后续再比较：

| 策略 | 可能收益 | 必须计入的成本 |
|---|---|---|
| 保留 D | 两个 GEMM 都能使用成熟实现 | D 的物化和读取 |
| 两个消费者分别重算 D tile | 少一个中间状态 | 重算、额外 H 读取、打断矩阵流水线 |
| 同一 CTA 生成 dH 和 partial dW | 片上复用 D | partial 数量、最终归约、累加器存储 |

不要以“单 kernel”作为默认胜出条件。

### 4.4 数据量实例：不是加速比预测

B=65536、Q=20、C=64、FP64 时，一个完整状态是 640 MiB。消除它的一次写回和一次读取，对应 1.25 GiB 逻辑流量；实际 DRAM 流量仍取决于 cache，必须实测。

以上是张量容量计算，不是 I/O 下界，也不能直接换成实际运行时间。

## 5. v0.3：jet 所有权、tile 和寄存器

### 5.1 全局布局与片上所有权分开

初始全局布局保持 `[sample,jet,channel]`。相邻 lane 访问相邻 channel，固定 jet 分量时具有连续访问。

激活/VJP 需要同一 `(sample,channel)` 的导数依赖集合。MMA fragment 不自动满足此条件；使用 warp shuffle 或 shared-memory 重排时，应显式计算其同步、交换与 bank-conflict 成本。[R06]

不能把某个样本需要的 jet 分在无通信的 CTA 上，之后用局部数据代替全量依赖。若将 jet 依赖分组，必须保存或重算组间需要的量。

### 5.2 不要强制把 Q padding 到 32

Q=20 补到 32，会使对应矩阵行数增加 60%；Q=13 补到 32，增加约 146%。这可能抵消代数压缩。

优先比较样本与 jet 联合打包、整组 jet tile、较小 tile、尾部 predication 和独立局部算子。物理 tile 必须服从具体后端支持，不能把逻辑 M=S*Q 当成所有 MMA 都接受的形状。

### 5.3 片上空间预算先于“融合深度”

单份 jet tile 字节数：`S * Q * Ctile * sizeof(double)`。

| S | Q | Ctile | 单份状态 |
|---:|---:|---:|---:|
| 4 | 20 | 64 | 40 KiB |
| 8 | 20 | 64 | 80 KiB |
| 16 | 20 | 64 | 160 KiB |
| 16 | 13 | 64 | 104 KiB |

表中不含权重、双缓冲、伴随、累加器、barrier 和辅助系数。不能据此宣称整段网络可驻留。

小 Q 先生成静态索引与直线代码，减少动态索引数组。寄存器是否真实分配、是否 spill，以编译器和 Nsight 记录为准；“C++ 局部数组”不保证位于寄存器。[R15]

### 5.4 两个边界特化

**首层：** 原始坐标输入时，零阶为 x，一阶为单位方向，高阶为零，可直接构造首层预激活 jet。存在 Fourier encoding 或非线性坐标映射时，禁用该捷径或提供专门规则。

**尾层：** 输出场较窄，可尝试最后线性层＋点局部残差＋seed。局部可分、固定权重的 loss 不必等待全 batch 标量归约才能启动局部反传；全局归一化、非局部约束等仍须保留依赖。

二者分别做消融，不与主融合一次性打包后只给总收益。

## 6. 代数合作者接口：C13 后端可以独立推进

### 6.1 适用范围与闭合式

在限定的三维空间目标中，若仅需值、梯度、对称 Hessian 与梯度-Laplacian，可保存：

\[
S(z)=(z,g,H,q),\quad g=\nabla z,\ H=\nabla^2z,\ q=\nabla\Delta z.
\]

共 1+3+6+3=13 个分量。对逐元素激活：

\[
\nabla\Delta\phi(z)=\phi'(z)q+\phi''(z)[g\,\mathrm{tr}(H)+2Hg]+\phi'''(z)g(g^Tg).
\]

这支持所限定表示的逐层闭合，不表示所有三阶 PDE 都只需 13 个量。一般各向异性算子、额外时间导数、非线性坐标变换和不同 loss 需要重新分析。降低导数表示的已有研究也必须列为近邻。[R20,R21]

### 6.2 同一协议，不同基

建议注册：

```text
full_taylor_2d3       Q=10，阶乘归一化
full_taylor_3d3       Q=20，阶乘归一化
grad_laplacian_3d3    Q=13，明确原始导数与 Hessian 打包约定
```

每个 JetSpec 至少包含：

```text
schema_version, basis_id, basis_revision
coordinate_dimension, max_derivative_order
coefficient_order, coefficient_normalization
primal_projection, output_recovery, cotangent_pairing
forward_rule, vjp_rule, activation_aux_state
operator_scope, unsupported_operations
symbolic_certificate_hash, generated_source_hash
```

这是拟定接口，不是本包已实现的 runtime。

### 6.3 朋友与 CUDA 开发者的交接件

朋友交付精确公式、结构常数、缩放、VJP、反例以及符号测试；CUDA 端把同一规格编译成寄存器级算术，检查线程布局和完整梯度。

若 `s=R j`，则对 s 的 cotangent 映射回 full-jet 坐标要用 `R.T`。不能因为 R 不可逆就随意用伪逆生成反传。Hessian 六分量打包的非对角权重也必须固定。

C13 的首要验收不是与 C20 逐分量相等，而是：观测量、loss 和完整参数梯度一致。参考公式和 CUDA 布局各自测试，避免两端共享同一错误索引。

**先接一个手工验证过的 C13，再考虑自动商代数生成器。** 不让通用代数搜索成为首个 CUDA PR 的依赖。

## 7. v0.4–v0.5：架构与精度按能力接入

### 7.1 能力不是一个 `sm >= 某值` 条件

分别检测和记录 native FP64 矩阵路径、TMA、异步 MMA、cluster、TMEM 等所需能力与工具链。一个架构支持某些低精度矩阵指令，不代表同样调度适用于 FP64。[R06,R18]

首版在实际可用 GPU 上建立功能和本机强基线；不要从消费卡结果推断数据中心卡的 FP64 性能，也不要从 B200 推断所有 Blackwell 型号。

### 7.2 异步协议必须可检查

至少区分：buffer ready、copy completion、MMA completion、消费者可见性和 buffer reuse。普通 block barrier 不替代所有异步完成或 proxy ordering。[R18]

先使用成熟库提供的 pipeline 规则。支持多 stream 时，不得覆盖尚被 wgrad 使用的 D/H；库句柄与当前 stream、workspace 生命周期必须一致。

### 7.3 cuBLASDx 是可选后端，不是先重写一个高精度 GEMM

本次文档核对到 cuBLASDx 0.7.1；0.7.0 提供 `RequiredMantissaBits<>` 仿真接口，0.7.1 修复了部分多阶段 TMEM accumulator 问题。0.6 起最低 CUDA Toolkit 为 13.0。[R07]

仿真 pipeline 还限制 `global_k <= 33025`。wgrad 的 K=BQ 很容易超过，需要 split-K；该限制不应泛化为所有 cuBLAS GEMM 的限制。[R09]

例如 B=4096、Q=20 时 K=81920。必须计入分段、额外归约、预处理与缓冲，不能从前向 GEMM 的收益推出完整训练收益。

### 7.4 Rubin 分支

Rubin 只在真实硬件、可发布结果的工具链和对应正确性测试可用后进入实测矩阵。此前可做编译与接口适配，但不要把带宽或峰值吞吐计算填成实测值。

本次 cuDNN 文档已经列出 9.25.1 GA；9.25.0 Developer Preview 的禁止发布性能数据规定只应准确归于该预览版本，不能写成“所有 Rubin 软件都仍为预览”。cuDNN Graph 的特定 support surface 也不自动覆盖任意 FP64 jet 图。[R12,R13]

近似 tanh、低精度激活、TF32/FP8 等变化若进入实验，独立开精度路线，重新验证完整参数梯度与科学验收，不作为默认吞吐捷径。

## 8. 依赖分层：首版只装与当前问题有关的东西

完整版本、安装和兼容性见 `DEPENDENCIES_ZH.md`。

| 层级 | 依赖 | 进入阶段 | 项目中的位置 |
|---|---|---|---|
| 数学与测试 | Python、PyTorch、NumPy、SymPy、mpmath、pytest | v0.1 | CPU 参考、参数梯度和代数校验 |
| CUDA 最小开发 | 完整 Toolkit、匹配的 PyTorch CUDA wheel、兼容 C++ 编译器、Ninja | v0.1–0.2 | 库 GEMM、扩展构建和运行 |
| GEMM/布局后端 | CUTLASS C++ / CuTe C++ 或 MathDx 中 cuBLASDx | v0.2 按需选一个 | 专用 epilogue 与 tile 计算 |
| 调试测量 | Compute Sanitizer、Nsight Systems、Nsight Compute | 第一个 CUDA kernel 起 | 错误、关键路径、寄存器与访存 |
| DSL 实验 | CuTe DSL | v0.3–0.5 可选 | 快速生成与布局搜索，不与首版 C++ 路线同时重写 |
| 代数强基线 | cuEquivariance | C13/稀疏多项式实验 | 比较分段多项式和伴随计算 |
| 应用接入 | PhysicsNeMo | 核心路径稳定以后 | 真实残差目标与训练集成 |
| 更远扩展 | Warp、cuDNN Graph、cuSolver、NCCL/NVSHMEM | 有明确工作负载再加 | 不作为当前高阶 MLP 的强制依赖 |

cuBLAS 通常由选用的 CUDA/PyTorch 分发提供；cuBLASDx 是另外的 MathDx 组件。安装 PyTorch CUDA wheel 不等于安装可构建扩展的 nvcc。[R05,R08]

## 9. 验证分层：数值正确性不随版本升级而放宽

| 层次 | 测试 | 通过后能说明什么 |
|---|---|---|
| L0 代数 | 链式法则、乘法、对偶、缩放的符号或精确检查 | 表达式在规定语义下等价 |
| L1 CPU | 与独立 AD 比较局部 jet、VJP、网络梯度 | 有限样例的浮点实现一致性 |
| L2 GPU | 同一输入的 forward、loss、所有参数梯度、奇数尺寸、尾块与 stride | 目标 GPU 实现的测试覆盖 |
| L3 并发 | stream、workspace、重复执行、边界、Sanitizer | 被测试路径未发现对应错误，不是形式化证明 |
| L4 优化 | 单步完整梯度、多个优化步骤、相同停止标准 | 系统收益未破坏规定的求解过程 |
| L5 应用 | 固定科学工件、物理指标或独立验证器 | 特定应用的可信结果，不是新 NS 定理 |

必须有零输入、负值、tanh 饱和区、交叉项抵消、极小导数、较大动态范围、不同坐标尺度以及 B/C 不整除 tile 的测试。loss 正确但参数梯度错误仍然失败。

容差按量纲、导数阶数与参考质量事先固定。近零量使用绝对误差或带固定 floor 的相对误差，不在看到加速结果后放宽阈值。有限差分只作补充，不是高阶导数的唯一参考。

首次只支持一阶参数 VJP 时，double backward/HVP 显式拒绝。后续由朋友生成混合微分规则，再在 CUDA 端验证，不默认 PyTorch 会补出缺失规则。[R19]

## 10. Profiling：每一代都重新寻找瓶颈

先测不加 profiler 的端到端 wall-clock，再用工具解释热点。分开记录编译、计划搜索、预热、稳定步骤、必要验收和修正时间。

至少记录：完整梯度时间、完整求解到达同一标准的时间、dgrad/wgrad/局部 jet 占比、实际 DRAM 字节、峰值显存、registers、spill/local 访问、shared-memory 交换、barrier stall 和 launch 数。[R15,R16]

Compute Sanitizer 的 memcheck、racecheck、initcheck、synccheck 分别检查不同错误；它们不是同一个工具的替代选项。[R14]

建议基准用例先固定 Q=10/20、C=32/64/128、若干从 launch-bound 到吞吐区的 batch。再补真实目标的实际形状分布，不能只保留最有利的一格。

**每次只改一个主要变量：** 比较融合时先固定表示与精度；比较 C13 时使用各自经过合理调优的实现；比较新硬件时双方都在同一硬件上重测。若库版本升级，保留旧 baseline 的标签，并重新生成强基线。

对重叠流水线，不把各子阶段独立计时简单相加冒充关键路径。对新数学表示，明确是同一个连续目标的等价实现，而不是另一种有限差分或另一个 loss。

## 11. 建议的第一组小 PR

| PR | 内容 | 验收 | 不应顺带做 |
|---|---|---|---|
| P0 | 固定 JetSpec、完整 loss、参考测试和环境清单 | CPU 对照可复现，unsupported 明确 | CUDA 特化 |
| P1 | packed 库 GEMM＋独立 jet/VJP GPU 基线 | 完整参数梯度，阶段 profile | 深融合 |
| P2 | 一个 `dgrad_jet_vjp_fp64` | 数学不变，GPU 测试＋B1 对照 | 改 Q、改精度 |
| P3 | 接回完整训练/细化步骤 | 计入 wgrad、优化器和验收成本 | 换优化器掩盖不足 |
| P4 | 完整 jet CTA 布局与有限 tile 搜索 | 报告交换、spill 与性能反例 | 任意形状通用编译器 |
| P5 | 朋友的 C13＋对偶规则，独立开关 | 观测量、loss、参数梯度一致 | 将符号测试说成 GPU 正确 |
| P6 | 一个真实工作负载的集成与 artifact | 固定源版本、数据、终止条件 | 声称完成整个奇性证明 |

新基线和 fallback 始终保留。某一阶段失败时，归档结果与反例，不必把已完成的正确实现推倒重写。

## 12. 第一篇论文的候选边界

候选主张：**通过高阶导数感知的片上所有权与参数伴随数据流，减少不必要的微分中间态，同时保留高效权重梯度归约。**

需要的证据是：强同算法 GPU 基线、完整参数梯度、消融、资源使用分析、真实端到端任务、误差与失败边界。至少一个主要收益机制须经单独验证，而不是把所有开关一起打开。

C13 可以成为第二项贡献，但只有当完整语义与数值测试通过、且与 CUDA 联合选择确有收益时再纳入。若只是手推一个已知收缩公式，不能把它包装成通用代数编译突破。[R20,R21]

多卡、Rubin、自动商代数搜索、HVP、区间验证、设备侧优化器分别进入后续 backlog；没有这些并不自动说明第一篇不完整。反之，加入它们也不能替代第一篇所需的科学证据。

不设“必须 2× 才继续”的拍脑袋门槛。继续条件是收益稳定、可归因、覆盖目标负载，并且实现与维护成本合理。若单 kernel 赢但完整目标不赢，应修改核心论点。

## 13. 本包实际提供与尚未提供的东西

已提供：本文、依赖文档、候选依赖 profile、隔离环境安装脚本、只读环境检查脚本、检查脚本的单元测试、此前代数检查的副本。

未提供：融合 CUDA 主 kernel、生产安装包、GPU 测试通过证据、真实科学工件复现、Rubin 实测或投稿完成承诺。安装脚本不安装/修改系统驱动，不代表其候选依赖组合已经在 GPU 上验证。

请先阅读 `README_ZH.md`，再按 `DEPENDENCIES_ZH.md` 的最小 profile 开始。`artifacts/` 内标为本次环境的记录只描述本次 CPU 环境，不能拿来作为用户 GPU 的锁定环境。

## 参考来源

以下均于 2026-09-08 检索；版本状态会变化。代码与文档的 `latest` 必须在实际实验中改为精确版本、commit 或归档哈希。

- [R01] FlashAttention，2022：`https://arxiv.org/abs/2205.14135`
- [R02] FlashAttention-2，2023：`https://arxiv.org/abs/2307.08691`
- [R03] FlashAttention-3，2024：`https://arxiv.org/html/2407.08608v1`
- [R04] FlashAttention-4，2026：`https://arxiv.org/abs/2603.05451`
- [R05] PyTorch C++/CUDA extension：`https://docs.pytorch.org/docs/stable/cpp_extension.html`
- [R06] CUTLASS GEMM 与布局：`https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html`
- [R07] cuBLASDx releases：`https://docs.nvidia.com/cuda/cublasdx/release_notes.html`
- [R08] MathDx/cuBLASDx 安装：`https://docs.nvidia.com/cuda/cublasdx/installation.html`
- [R09] cuBLASDx pipelined GEMM：`https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html`
- [R10] CUTLASS DSL quick start：`https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html`
- [R11] PyTorch wheel 版本：`https://pytorch.org/get-started/previous-versions/`
- [R12] cuDNN release notes：`https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html`
- [R13] cuDNN Graph 支持范围：`https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html`
- [R14] Compute Sanitizer：`https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html`
- [R15] Nsight Compute profiling：`https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html`
- [R16] CUDA Linux 安装与工具链：`https://docs.nvidia.com/cuda/cuda-installation-guide-linux/`
- [R17] cuEquivariance 多项式：`https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance.SegmentedPolynomial.html`
- [R18] PTX ISA 与同步规则：`https://docs.nvidia.com/cuda/parallel-thread-execution/index.html`
- [R19] PyTorch 自定义变换支持：`https://docs.pytorch.org/docs/stable/notes/extending.func.html`
- [R20] Collapsing Taylor Mode AD：`https://arxiv.org/abs/2505.13644`
- [R21] 高阶混合导数与 Bell 多项式：`https://arxiv.org/abs/2609.03768`
