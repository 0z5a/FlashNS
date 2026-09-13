# FlashNS：高阶 jet + 参数 VJP 的 CUDA 执行设计

2026-09-08。研究提案；CPU 数学参考已做小样本测试；无 CUDA 编译、实机验证、性能数字或 Rubin 实测。

## 1. 新近邻改变新颖性边界

2026-09-03 的 Imoto 预印本《Computing high-order mixed derivatives in physics-informed neural networks using multi-index Bell polynomials》已包括向下闭合的多重指标集合、前向混合导数、显式参数反传和独立验证，公开实现 DNNF90 为 Fortran。不能声称首次实现这些数学机制。[R1,R2]

候选 GPU 贡献应是：**联合优化 jet 分组的输出归属、片上激活伴随量重建、GEMM 融合边界与全局梯度归约。** 是否有足够新颖性与端到端收益仍待实验和更广泛检索。

## 2. 数学接口

采用阶乘归一化系数 `H_alpha = partial^alpha h / alpha!`。二维三阶 Q=10，三维三阶 Q=20。完整 loss（包括残差梯度、边界条件、坐标变换）决定依赖集合，不能只分析 PDE 名称。

线性层：`Z_alpha = Hprev_alpha @ W.T + 1[alpha=0]*bias`。

将 jet 看作截断多项式。若 `H=tanh(Z)`，则 `G=1-H*H` 是 `tanh'(Z)` 的 jet，乘法是截断 Cauchy 卷积。激活 VJP：

`barZ_beta = sum_{alpha>=beta} barH_alpha * G_(alpha-beta)`。

这可在精确实数算术下从 H 重建 G，无需保存整个 Bell 中间图、Z 或单独的 G checkpoint；普通 FP64 有舍入、抵消和饱和风险。该重建策略是 tanh 特化，不是所有激活通用性质。三阶空间导数的参数梯度涉及四阶激活导数，不能在 phi''' 截断。

本层三种线性代数：

- forward: `Z = H @ W.T`
- dgrad: `barH = barZ @ W`
- wgrad: `dW = barZ_flat.T @ H_flat`

`flat` 合并 batch 和 jet 轴。bias 梯度只从 `barZ[:,0,:]` 沿 batch 求和，但这个零阶伴随量已经接收了高阶 loss 的贡献。

## 3. 基础数据布局与 CTA 所有权

HBM 布局先固定 `[B,Q,C]`，channel 连续。forward/dgrad 把 B*Q 视为矩阵行；wgrad 把 B*Q 视为归约维，不需要显式 transpose tensor。

普通 GEMM output fragment 并不把同一神经元的 Q 个系数自动放在一个线程。epilogue 必须将 MMA 所有权转换为 `(sample,neuron)` 所有权：先在 CTA 内汇集完整 jet，再进行多项式和 VJP。不要在缺少跨 CTA 同步时用一个 CTA 的局部寄存器代替完整 jet。

候选逻辑 tile：样本 S=4，Q=20，输出 N=32，逻辑 M=S*Q=80；由后端满足实际 MMA 原子及 tile 限制，并计入 padding。不要默认把 Q=20 补到32（会使相关矩阵工作增加60%）。同一 CTA 需要覆盖每个被处理样本的完整依赖闭包。

## 4. 最建议的融合边界

### 4.1 Forward GEMM + jet activation

K 维归约完成后，CTA 内重新分配输出，直接生成 H，只保存 H checkpoint，不把 Z 写到 HBM。小 Q 用静态展开或生成代码，避免动态表和动态数组索引造成 local memory。

### 4.2 Dgrad GEMM + 前一层的 jet VJP

从 `D_l=barZ_l` 计算 `barH_(l-1)=D_l @ W_l`，在 epilogue 内利用 `H_(l-1)` checkpoint 重建 G，立刻变成 `D_(l-1)`。

**不保存 `barH_(l-1)`，但先保存 `D_(l-1)`**：后者由下一次 dgrad 和本层 wgrad 两个消费者共享。强行消除它可能造成激活 VJP 重算或低效梯度局部归约。

### 4.3 独立 wgrad

每层保留强 GEMM 基线：`D_flat.T @ Hprev_flat`。先用库或成熟 CUTLASS 后端。只有 profile 证明 D 的数据搬运为主要瓶颈且融合带来总收益时，再研究双消费者融合。

### 4.4 尾部融合

输出宽度小，可研究“最后线性层 + 点局部残差 + 残差梯度 loss + 输出 seed + 返回隐藏层的 dgrad”。点可分、固定权重 loss 不必等全 batch 标量归约后才启动局部反传。最后一层 wgrad 仍需归约。非局部 loss / 自适应全局归一化有额外依赖。

### 4.5 首层特化

原始坐标或已知仿射坐标的输入 jet 很稀疏：零阶为 x，一阶为单位方向，高阶为0。第一线性层直接生成 `Z0=Wx+b`、`Z_ei=W[:,i]`，其余为0，无需乘稠密零 jet。非线性坐标变换/编码需先正确传播其全部 jet。

## 5. 片上存储与危险的全融合

S 个样本、Q 个系数、C 个通道、FP64 单缓冲字节数为 `S*Q*C*8`。

Q=20,C=64：S=4为40KiB，S=8为80KiB，S=16为160KiB。双缓冲、权重、MMA operands、checkpoint、barrier、activation 临时量要另外计入。因此不能只按一个 jet tile 宣称可以驻留整段网络。

B=65536,Q=20,C=64 时，一个全 batch 层状态为640MiB；去掉一次全量写回+读取，逻辑流量为1.25GiB。实际 DRAM 流量受缓存影响，需要测量。

反例：B=2^20，每 CTA处理16点，每 CTA为8个64x64层写一份 dW partial，临时数组为16GiB。换成少量 persistent CTA 会减少最终 partial 数量，但若梯度累加器放不进片上，需要反复读写 scratch；不能假定因此没有成本。

推荐第一版逐层融合+独立wgrad，而不是整个网络一个kernel。

## 6. 同步、数值与后端边界

使用 cp.async/TMA/MMA 时，将 buffer-ready、copy completion、MMA completion、跨 proxy 可见性分别处理；普通 `__syncthreads` 不自动替代所有 async 完成/排序规则。[R5,R6]

寄存器压力按实际 ptxas 与 Nsight 测量；FP64标量通常占两个32位寄存器槽。模板数组不保证驻留寄存器，动态索引和大数组可能进 local memory。[R7,R8]

不要用 tanh.approx.f32 冒充FP64激活。先不启用 fast math，并验证 tanh 饱和区、极小导数、交叉项抵消、坐标缩放、残差梯度和参数梯度。符号正确不等于数值稳定。[R9]

FP64后端必须与低精度Tensor Core后端区分。PTX列出FP64 mma.sync的特定形状；不能把WGMMA/tcgen05示例无条件换成double。[R9] Rubin高精度矩阵路径与cuBLAS仿真应作为独立精度策略进行比较，不预设当前自定义epilogue可直接嵌入所有库的内部仿真路径。[R10,R11]

## 7. 验证与基线

当前实际验证：2-D/3-D三阶activation VJP；2→8→8→3小网络的完整输出jets；二维稳态动量/散度+残差空间梯度loss；显式参数梯度与独立nested AD比较。数据见两个JSON。不代表科学收敛、鲁棒性或GPU正确性。

强基线：同样数学表达的高效GPU jet（库GEMM+独立activation/VJP），可适用的torch-jet/collapsed Taylor，以及compiled/nested AD。DNNF90是重要正确性与算法近邻，但GPU对单核CPU的倍率不能证明CUDA机制的贡献。[R1,R2,R3]

关键消融：普通layout vs完整jet CTA归属；保存G vs重建G；dgrad和VJP分离vs融合；独立wgrad vs双消费者融合；逐层checkpoint vs分段重算；一致参数梯度要求下的端到端耗时。

记录：HBM/DRAM字节、local load/store、register spill、shared memory bank conflicts、barrier stalls、GEMM吞吐、激活开销、wgrad归约成本、peak memory和得到同一验收结果的总时间。[R8]

只实现参数的一阶VJP不自动支持HVP、完整Jacobian或double backward。需要这些的优化器必须另做JVP/VJP组合与验证，不能宣称无条件drop-in。

## 原始来源（2026-09-08检索）

[R1] Imoto, 2026-09-03. https://arxiv.org/abs/2609.03768 ; https://arxiv.org/html/2609.03768v1
[R2] DNNF90 official repository. https://github.com/fimoto/DNNF90
[R3] Collapsing Taylor Mode Automatic Differentiation; torch-jet. https://arxiv.org/abs/2505.13644 ; https://github.com/f-dangel/torch-jet
[R4] CUTLASS Efficient GEMM. https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
[R5] CUDA async copies. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
[R6] CUDA advanced kernel programming. https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html
[R7] CUDA Best Practices. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
[R8] Nsight Compute Profiling Guide. https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
[R9] PTX ISA. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
[R10] Rubin architecture. https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/
[R11] cuBLAS floating-point emulation. https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/
