# FlashNS RFC v8：数学合同、按需求阶数执行与可归因的快速求解

**日期：2026-09-09｜承接交接包 v7，不是退回 FA3/v6 的实现状态。**

**主线：明确计算什么 → 证明哪些状态充分 → 消除无用工作 → 在相同科学目标下验证完整求解收益。**

本文将独立 Kovasznay PINN 和 OpenAI 论文局部波幅构造分成两个 workload。前者推进完整参数梯度与求解，后者推进论文 §7 的批量局部演化；不把二者合并为“已实现通用 NS 解析求解器”。

## 0. 执行摘要与证据分层

| 标记 | 含义 | 本版内容 |
|---|---|---|
| **H** | 上传交接包的历史运行记录 | H100 NVL、stable aux、TMA、完整 Kovasznay 求解、OpenAI 局部适配与形式化重放 |
| **P** | 本文给出的实数算术推导 | jet/VJP 等价性、保存状态充分性、边界消除、混合阶 packed 布局、逻辑成本 |
| **CPU** | 本轮在当前容器真实执行 | 20 组独立 CPU 检查；另 3 组交接源码集成检查，共 23/23 通过 |
| **Plan** | 尚需目标 GPU 实现/测量 | 显式 residual/seed CUDA、紧凑混合阶 GPU 路径、跨谐波传播复用性能 |

当前环境是 **Torch 2.10.0+cpu，无可用 CUDA**。本轮没有新的 H100/A4000 benchmark，没有重新执行 Lean/Comparator，也没有编译 CUDA。原 CUDA header 仅重新编译成 HOST C++ 检查代数表达，不等于 CUDA 指令、stream、Graph 或 barrier 验收。

本版最重要的新增候选不是更大的融合内核，而是：

> **内点保存完整三阶 jet，纯 Dirichlet/压力定值边界只保存零阶值；两类状态紧凑打包，仍共享一次 affine GEMM 和一次 wgrad GEMM。**

数学上，独立样本上的不同截断阶数可以共存；系统上，这避免“删掉无用导数，但增加多次小 GEMM”的直接两路实现陷阱。

另一个独立入口来自 OpenAI 论文式 (7.18)：同一几何轨迹上的不同谐波可共享基础二维齐次传播子，以标量阻尼恢复其余模式。本文补上适用条件、证明和有源项反例，不声称整个受迫解只需一次缩放。[W02]

---

## 1. 先更新项目真实状态

### 1.1 已经完成，不能继续列为待实现

依据交接包 [H01–H05]：stable aux 已实现；H100 NVL 的 FP64 TMA 路径已实现；八后端完整 Kovasznay 求解已执行。TMA 使用 FP64 `mma.sync`，而非 FP64 WGMMA。硬件计数器未取得；不能将资源静态信息当作已测访存或 stall。

本轮从原始 `extension1/suite.json` 重新计算统计，结果与根目录 `02_KEY_RESULTS.json` 的配对倍率完全一致。这里复核的是 JSON 算术，不是远端 GPU 复测。

| 后端 | 优化/验收中位数 s | 求解总计中位数 s | 同种子 total/TMA 的配对中位数 |
|---|---:|---:|---:|
| B1 | 21.847940 | 22.527761 | 1.014498× |
| 原融合 F | 21.867414 | 22.426633 | 1.009944× |
| Hopper TMA | 21.647115 | 22.205823 | 1.000000× |
| cuBLASLt | 21.811742 | 22.722830 | 1.023283× |
| TorchJetCompiled | 21.471771 | 25.941474 | 1.168228× |
| CuEquivariance | 22.341913 | 25.793690 | 1.117091× |
| TorchNestedAD | 95.766067 | 96.659866 | 4.352906× |
| PhysicsNeMo | 117.541131 | 118.789797 | 5.278978× |

B1/TMA 的三个种子倍率为 **1.014498、0.976816、1.074301**。不能把约 1% 的三种子中位差表述成稳健优势。TorchJetCompiled 的热阶段中位数还略低于 TMA；它的总计差距包含 setup，不能全部归给 kernel。

原预算 17/24 达标；统一把 L-BFGS 上限从 200 改到 500 blocks 后，7 个失败组合从相同初始化重跑，最终 24/24 达标。实际运行包括原失败共 **31 次**，不是从第一次运行起即 24/24。保留全部成本、失败和协议修改时间。[H01,H02]

### 1.2 正在补齐，而不是已经证明性能提升

当前 `manual_step` 仍将最终输出 jet 作为 autograd leaf，通过 loss 的框架反向生成 seed；2,560 个点仍一起执行完整 Q=10 jet。v7 的 residual/seed 和边界分路是下一轮候选。本版提供其数学合同和 CPU 参考，没有伪造 GPU 移植结果。

源码锚点：

| 文件 | 本轮检查对象 |
|---|---|
| `experiments/pinn_solver/problem.py` | PDE/边界权重、压力出流、原始 loss |
| `experiments/pinn_solver/backends.py` | 手工参数反传和最终 leaf seed 边界 |
| `experiments/cuda_jet_h100/common.py` | 三阶 jet 的 residual-gradient 目标 |
| `src/flashns/jet_stable.py` | 稳定 a1、tanh forward 与 VJP |
| `experiments/cuda_jet_h100/stable_jet.cuh` | 已有展开式；本轮仅 HOST C++ 复核 |
| `experiments/openai_ns_formal/projected_amplitude.py` | 已有局部投影波幅 RHS |

---

## 2. 冻结目标：首先证明的是离散训练目标的梯度

### 2.1 原始问题

固定物理坐标 `(x,y)`、黏度 `ν=0.07`、网络 `2→64→64→3`，输出 `(u,v,p)`。训练集为 2,048 个内点、512 个速度边界点，其中最后 128 个点位于压力出流边界。参数共 4,547 个。[H03]

记

\[
r_u=uu_x+vu_y+p_x-\nu(u_{xx}+u_{yy}),
\]
\[
r_v=uv_x+vv_y+p_y-\nu(v_{xx}+v_{yy}),\qquad d=u_x+v_y.
\]

以固定全局权重 `w_i`、`b_{j,c}` 表示目标：

\[
L_I=\frac12\sum_{i\in I} w_i\big[r_u^2+r_v^2+d^2
+\lambda_g(\|\nabla r_u\|^2+\|\nabla r_v\|^2+\|\nabla d\|^2)\big],
\quad \lambda_g=0.2,
\]
\[
L_B=5\sum_{j\in B}\sum_{c\in\{u,v,p\}}b_{j,c}(f_{\theta,c}(x_j)-y_{j,c})^2,
\qquad L=L_I+L_B.
\]

当前内点 `w_i=1/2048`；边界 u/v 各用 `1/512`；压力仅出流的 128 点用 `1/128`。u/v 项是“每点两个分量平方和再取点均值”，**不是再对两个分量平均**。压力项用于固定 pressure gauge，不可删除。

一般非均匀积分权重也允许，但在进入 kernel 前已包含各 loss 项自己的全局分母。严禁在分路后再次 `/local_batch`、`/Q` 或 `/world_size`。

### 2.2 证明范围

以下命题证明：**对固定点集、固定参数无关权重及声明的网络，在实数算术下，新表示/调度得到相同有限目标及其一阶参数梯度。**

这不等于证明 PINN 收敛到 PDE 真解、连续积分与有限采样完全等价、任意边界条件可用零阶状态，也不证明浮点程序逐位一致。

假设网络对坐标与参数足够光滑。本文 tanh MLP 联合解析；涉及三阶空间导数再对参数求导时，局部激活最高需要四阶导数。ReLU、随机 dropout、跨样本 BatchNorm、训练中变化且参与求导的采样/权重不直接落在此合同内。

---

## 3. 命题 P1：三阶阶乘归一化 jet 的计算等价性

令多重指标 `α=(α₁,…,α_d)`，`|α|=Σα_i`，`α!=∏α_i!`。注意 **α! 不是 |α|!**。

定义有限维截断代数

\[
\mathcal A_{d,3}=\mathbb R[\varepsilon_1,\ldots,\varepsilon_d]/\mathfrak m^4,
\qquad \mathfrak m=(\varepsilon_1,\ldots,\varepsilon_d),
\]
\[
J^3f(x)=\sum_{|\alpha|\le3}\frac{\partial_x^\alpha f(x)}{\alpha!}\varepsilon^\alpha,
\qquad Q=\binom{d+3}{3}.
\]

其中 `d=2/3/4` 对应 `Q=10/20/35`。Q=35 是一般四变量参考大小，不代表现有 CUDA 支持四变量非定常 NS。

### 证明

线性性由导数线性性得到。由多重指标 Leibniz 公式，

\[
\frac{\partial^\alpha(fg)}{\alpha!}
=\sum_{\beta+\gamma=\alpha}
\frac{\partial^\beta f}{\beta!}\frac{\partial^\gamma g}{\gamma!}.
\]

因此 `J³(fg)=J³f · J³g mod 𝔪⁴`；归一化后乘法就是没有额外二项式系数的 Cauchy 卷积。

对 `Z=z₀+δ`、`δ∈𝔪`，有 `δ⁴=0`，故

\[
J^3[\phi(z)]=\phi(z_0)+\phi'(z_0)\delta
+\frac{\phi''(z_0)}2\delta^2
+\frac{\phi'''(z_0)}6\delta^3.
\]

这不是丢弃所需导数的近似：在截断代数中，高于三阶的项本来为零。对网络的 affine 层与激活层依次归纳，得到所有 `|α|≤3` 的精确导数系数。

### 原始导数恢复及 seed 转置

令 `R=diag(α!)`，`D=RJ` 是原始导数坐标。在这两个向量空间的普通 Euclidean 配对下：

\[
\bar J=R^T\bar D.
\]

例如 `u_xxy=2J_(2,1)`、`u_xxx=6J_(3,0)`，对应 seed 也分别乘 2、6；不能恢复前向时乘了阶乘，反向却忘掉转置。

若空间坐标另作变换，需将其真实链式映射及转置加入 `R` 的位置；非线性坐标映射一般不再只是对角缩放。

**CPU 对应检查：**混合导数系数、seed 配对恒等式；二维/三维小 MLP 全部三阶输出 jet 与独立 nested AD，以及任意全阶 seed 的全部参数梯度。

---

## 4. 命题 P2：显式 JetVJP 与原始参数梯度一致

固定一个神经元的 jet。对输入扰动 `δZ`，由链式法则在截断代数中有

\[
\delta H=G\,\delta Z,\qquad G=J^3[\phi'(z)].
\]

逐系数展开、交换有限求和：

\[
\sum_\alpha\bar H_\alpha\delta H_\alpha
=\sum_\beta\left(\sum_{\substack{\alpha\ge\beta\\|\alpha|\le3}}
\bar H_\alpha G_{\alpha-\beta}\right)\delta Z_\beta.
\]

所以

\[
\boxed{\bar Z_\beta=\sum_{\substack{\alpha\ge\beta\\|\alpha|\le3}}
\bar H_\alpha G_{\alpha-\beta}.}
\]

`α≥β` 指逐坐标比较，不是仅比较总阶数。实现为稀疏相关，不需要构造 Q×Q 的 Jacobian。

非零结构对数为

\[
S_d=\#\{(\beta,\gamma)\in\mathbb N^{2d}:|\beta|+|\gamma|\le3\}
=\binom{2d+3}{3}.
\]

因此二维 35、三维 84、四维 165；它是这套展开的结构计数，**不是所有算法的乘法下界**。

### 4.1 为什么要四阶激活导数

取一变量 `Z=z₀+cε`。三阶系数为 `H₃=φ'''(z₀)c³/6`，对参数控制的 z₀ 求导得到 `φ''''(z₀)c³/6`。故“三阶空间导数只需三阶激活导数做完整参数反传”是错误的。

tanh 的特殊恒等式允许用已保存的 H 表示 G，但这并不消除四阶导数的数学作用；它被等价编码进了 G 的三阶系数。

### 4.2 affine 反向与权重/边界的完整性

设 `Z_{b,α,o}=Σ_c W_{o,c} H_{b,α,c}+1_{α=0}b_o`，则

\[
\bar H_{b,\alpha,c}=\sum_o\bar Z_{b,\alpha,o}W_{o,c},
\]
\[
\bar W_{o,c}=\sum_{b,\alpha}\bar Z_{b,\alpha,o}H_{b,\alpha,c},
\qquad
\bar b_o=\sum_b\bar Z_{b,0,o}.
\]

**bias 仅对零阶行归约；wgrad 对全部有效 jet 行归约。** 将 residual、边界、阶乘恢复、固定权重的转置 seed 传入，再逐层应用这些公式，链式法则即给出 `∇θL`。

未增加任何 `/Q`；点集重排对应同一个置换及其逆转置；浮点归约顺序可不同，但数学目标不变。

**CPU 对应检查：**JVP/VJP 内积、全部混合导数参数梯度、不同点数/宽度、零权重、空分组、五次相同 SGD 更新。五步一致不是科学收敛证明，也不替代真实 Adam/L-BFGS 检查。

---

## 5. 命题 P3：tanh 保存状态的充分性，与 stable aux 的必要性边界

### 5.1 实数上的充分状态

对 tanh，`φ'(z)=1−φ(z)²`。所以在实数算术下，保存完整 H 即可由

\[
G=1-H^2\pmod{\mathfrak m^4}
\]

重建一阶参数 VJP 所需的 G。这里说的是**一种充分状态**，不是证明所有实现的最小状态，也没有证明保存 H 比重算 Z 总是更快。

### 5.2 FP64 中 H-only 的信息丢失

取常量输入 jet `Z=(20,0,…,0)` 与 `(21,0,…,0)`。当前 CPU 上二者的 `tanh(z₀)` 均舍入为 1，所有高阶 H 均为零，但真实一阶导数不同且非零。

因此任何只看到这些相同 FP64 H 的确定性重建规则都无法同时恢复两个真实斜率。这是 **H-only 表示在此情形不充分的构造性反例**，不是对任意激活的统一结论。

现有实现保存

\[
a_1=\left(\frac{2r}{1+r^2}\right)^2,\qquad r=e^{-|z_0|}.
\]

再使用

\[
H=\tanh(z_0)+a_1\delta-\tanh(z_0)a_1\delta^2
+a_1(\tanh^2(z_0)-1/3)\delta^3,
\]
\[
G_0=a_1,\qquad G_{\alpha\ne0}=-(H^2)_\alpha.
\]

保存 `H[B,Q,C]+a1[B,C]` 是当前稳定策略的一组充分 checkpoint。边界零阶分支也必须保存/重算同一稳定 a1，不能退回 `1−H0²`。

### 5.3 不可扩大声明

这证明的是：在实数规则下等价，且 aux 修复了指定的常数项饱和信息丢失。它不证明：

- FP64 计算结果等于“每条机器舍入指令组成的离散程序”的经典导数；数值 AD 通常是在浮点值处评估实数导数规则。
- 非零高阶 H 已经下溢或强消去时还能由 a1 恢复全部信息。
- 所有参数范围具有统一相对误差界，或 H+a1 是全局信息论最小状态。
- native 接口已经支持 HVP/double backward；高阶参数求导还需要额外的接口与状态导数合同。

### 5.4 检查零点的参考实现

CPU 高阶 AD 参考中的 `exp(-abs(z))` 若直接走 abs 在零点的选定次梯度，可能破坏后续高阶导数。交接源码使用 `where(z>=0,-z,z)` 的解析单侧分支；其组成的 a1 在实数上是光滑偶函数。测试必须覆盖 z=0 的二至四阶关系，不仅检查函数值和一阶导数。[H04]

---

## 6. FP64 误差：给出局部界，不把实数证明冒充稳定性定理

令 `u=2^-53`，`γ_n=nu/(1−nu)`。在无溢出、普通相对舍入模型适用且 `nu<1` 的累加中，固定 β，定义 `A_β={α:α≥β,|α|≤3}`。记保存态及 seed 的误差分别为 `ΔG,ΔbarH`，可以写出

\[
|\widehat{\bar Z}_\beta-\bar Z_\beta|
\le \sum_{\alpha\in A_\beta}
\big(|\bar H_\alpha||\Delta G_{\alpha-\beta}|
+|G_{\alpha-\beta}||\Delta\bar H_\alpha|
+|\Delta\bar H_\alpha||\Delta G_{\alpha-\beta}|\big)
+\gamma_{c|A_\beta|}\sum_{\alpha\in A_\beta}|\widehat{\bar H}_\alpha\widehat G_{\alpha-\beta}|
+\eta_{\rm underflow}.
\]

常数 c 取决于实际乘加/FMA 顺序，不能在未检查指令时写死。`η_underflow` 表示使用绝对误差模型处理的次正规/下溢项。

对于 `G_γ=−Σ H_βH_η`，同样由乘积扰动和求和界给出 ΔG；但 γ=0 应直接使用稳定 a1 的误差，不再经过两个接近 1 的数相减。

**强消去时绝对误差界仍可能有用，相对误差界可能任意差。** `exp/tanh` 的库误差须另行进入 ΔH/Δa1，不能假设它们均正确舍入，也不能只套一条 γ_n 宣称全网络 backward stable。

建议每个残差同时记录

\[
\kappa_{\rm sum}=\frac{\sum_k|t_k|}{|\sum_k t_k|}
\]

或有明确正则分母的版本；原和为零时保留零/无穷状态，不用任意 epsilon 隐藏病态情况。该量用于解释消去，不能自动成为用户目标之外的拒绝规则。

### 6.1 精度分层

| 区间 | 验收 | 不能做的事 |
|---|---|---|
| 普通 FP64 输入 | 固定 atol+rtol；loss、各阶输出、全部参数 | 仅比较 loss |
| tanh 饱和但导数正常数 | 高精度独立参考，适当相对误差 | 用大绝对容差把非零小导数清零也算通过 |
| 强消去 | 绝对误差/残差尺度/条件指标并列 | 仅报均值误差 |
| 次正规 | ULP 或绝对量子检查，分类记录 | 强求所有值相同相对精度 |
| 真实下溢 | 声明表示极限或单列缩放格式 | 静默修改精度合同、删掉失败样本 |

本轮用 420 位 mpmath 检查标量 a1，正负探针覆盖 0、15、20、21、40、100、300、355、360、370、372、373、374、380。普通区采用独立相对阈值，次正规/零区以最小 FP64 量子的绝对误差检查。**这只是当前 CPU 的标量量化检查，不是 H100 的尾部保证或整个 solver 的误差界。**

---

## 7. 命题 P4：内点三阶与边界函数值分开计算

令 `π₀:𝒜_{d,3}→R` 取零阶系数。它满足

\[
\pi_0(A+B)=\pi_0(A)+\pi_0(B),\quad
\pi_0(AB)=\pi_0(A)\pi_0(B),\quad
\pi_0(\phi(A))=\phi(\pi_0(A)).
\]

因此完整 jet 网络与普通函数值网络在零阶输出相同。若边界 loss 只依赖零阶输出，则

\[
L_B^{\rm full\ jet}(\theta)=L_B^{\rm value}(\theta).
\]

两边对参数可微，故其参数梯度相同。另由和的微分线性性，

\[
\boxed{\nabla_\theta L=\nabla_\theta L_I+\nabla_\theta L_B.}
\]

也可直接从 P2 验证：当边界输出 `barH_α=0 (|α|>0)` 时，对任何 `β≠0`，不存在能贡献的 `α≥β`，因此高阶输入伴随仍为零。高阶边界状态不会影响零阶 seed 的参数梯度。

### 实施合同

两个分路使用同一参数和稳定激活；**先合并梯度，再执行一次优化器更新**。不能内点更新一次、边界再更新一次。微批和多卡也仅改变求和组织，不改变各分量全局分母。正则项如有则全局计入一次。

对空分组返回零 loss 和零梯度，不对空 tensor 求 mean。出现不合法点时不能单独删除并沿用旧归一化。点排序须恢复原权重与目标对应，包括角点重复采样。

### 失效与推广条件

Neumann/Robin、traction、法向应力、边界 residual-gradient、坐标变换导数等可能要求更高边界阶数。跨样本归一化、非局部积分损失、训练中的自适应权重/采样也需额外分析。

当权重本身依赖 θ 时，简单“冻结权重再反传”会漏掉 `∂θw · ℓ`；只有原始数学目标已将这些权重明确 stop-gradient 时才合法。

---

## 8. 命题 P5：不增加 GEMM 次数的混合阶紧凑表示

### 8.1 新的数据流

直接双分路虽然合法，但每层可能从一个 GEMM 变成两个小 GEMM。更好的首选候选是

\[
\mathcal V=\bigoplus_{i\in I}\mathcal A_{d,3}\ \oplus\ \bigoplus_{j\in B}\mathbb R.
\]

每个点独立选择所需截断代数，将所有系数行紧凑存储：

```text
H_packed [Ni*Q + Nb, C]
┌─────────────────────────────────────┬─────────────────────┐
│ interior: sample0 Q rows, …         │ boundary: 1 row/point│
└─────────────────────────────────────┴─────────────────────┘
        │ 同一 W，一次 affine GEMM
        ↓
interior stable jet          boundary stable scalar tanh
        │                              │
        └──────── compact H + a1 ──────┘
                       ↓
  residual/seed 或边界 value/seed（各保留原权重）
                       ↓
一次 wgrad GEMM：DᵀH；bias 只 gather 各点的零阶行
                       ↓
一次 dgrad GEMM：D W；分段 JetVJP / scalar VJP
```

`a1` 仍是每点每通道一份；不因边界减少 Q 就删除其稳定状态。

### 8.2 证明

W 对样本、导数指标均使用同一系数，只作用于 channel 维；所以将所有有效 `(point,coefficient)` 行串接后，矩阵乘法与分路计算逐行相同。激活只在各样本自己的截断代数内计算，不跨样本混合。

设分路前向输入为 `H_I,H_B`，输出伴随为 `D_I,D_B`，则

\[
\begin{bmatrix}D_I\\D_B\end{bmatrix}^{T}
\begin{bmatrix}H_I\\H_B\end{bmatrix}
=D_I^TH_I+D_B^TH_B.
\]

所以 wgrad 可以由一次 packed GEMM 完成。对 bias，使用 `zero_rows` gather：内点每 Q 行取一行，边界所有行取入；不能对全部 packed 行直接求和。

前向与反向在直和各分量中分别满足 P1/P2/P4，因而得到原目标的相同一阶参数梯度。**这是表达与调度的等价性证明，不是“CUDA 编译器已实现最优执行”的证明。**

### 8.3 当前冻结问题的精确工作量账本

`Ni=2048,Nb=512,Q=10,C=64`：

| 项 | 原 all-points Q10 | 紧凑 mixed-order | 差异 |
|---|---:|---:|---:|
| affine/wgrad 的样本系数行 | 25,600 | 20,992 | −18% |
| 一层 H 容量 | 12.50 MiB | 10.25 MiB | −2.25 MiB |
| 同一层 a1 容量 | 1.25 MiB | 1.25 MiB | 不变 |
| H+a1 | 13.75 MiB | 11.50 MiB | −16.36% |
| 每通道 VJP 相关乘积数 | 89,600 | 72,192 | −19.43% |
| eliminated barH 的逻辑写+读 | 25.00 MiB | 20.50 MiB | 各自独立的融合机会 |

**18% 不是完整梯度省时、更不是求解加速。** 所有点的 `tanh(z₀)`、a1 计算仍存在；点级非线性函数次数并未随 Q 同比降低。weights 的读取、输出层尺寸、库 dispatch、L2 命中、padding、Graph、L-BFGS 和验收都有独立成本。

### 8.4 首个 GPU 实现应保持小范围

先用库 packed GEMM，配两个分段 activation/VJP 入口；不在第一版给 TMA uniform-Q 接口塞入不规则行。现有 TMA 的 owner/layout 假设需要显式修改，不能只改 tensor shape。

静态两段无需给每行存昂贵通用 tag：`Ni,Q,Nb` 和分段边界即可；若以后支持更多阶数，再增加 compact descriptor。若内核要求 tile 内完整 jet，必须让 padding/跨 tile 交接进入容量、性能和正确性记录。

比较三种合法调度：`dense`、`split`、`compact`。第二步才判断混合 epilogue 是否值得融合。允许 split 在某些形状获胜，也允许 compact 因库计划变化而回退。

### 8.5 不要把任意稀疏导数集合当作截断代数

完整的 `|α|≤r` 截断自然闭合。任意选几个 mixed partial 后删掉其他量，未必对激活复合与参数转置闭合。C8/C13 等算子专用压缩需另证明；本版不拿边界 Q0 的简单证明替它们背书。

---

## 9. 显式 residual/loss/seed：一个可审核的稀疏多项式合同

对每个内点，将

\[
\mathbf r=(r_u,r_v,d,\partial_xr_u,\partial_xr_v,\partial_xd,
\partial_yr_u,\partial_yr_v,\partial_yd)^T,
\quad M=\operatorname{diag}(1,1,1,.2,.2,.2,.2,.2,.2).
\]

则

\[
L_I=\frac12\sum_i w_i\mathbf r_i^TM\mathbf r_i,
\qquad \bar{\mathbf r}_i=w_iM\mathbf r_i.
\]

用原始导数坐标写：

\[
\bar J_i=R^T\left(\frac{\partial\mathbf r_i}{\partial D_i}\right)^T\bar{\mathbf r}_i.
\]

也可直接在归一化 jet 上构造稀疏多项式，但必须将 factorial 因子吸收入项系数。这样不应再额外乘一遍 R。

例如

\[
\partial_xr_u=u_x^2+u u_{xx}+v_xu_y+v u_{xy}+p_{xx}
-\nu(u_{xxx}+u_{xyy}).
\]

`u_xyy=2J_(1,2)`、`u_xxx=6J_(3,0)`；把两个量都误乘 6 会改变 loss 与参数梯度。

### 9.1 IR 与转置规则

本包 `ns_terms()` 生成 9 个输出的线性/二次项表，每项 `(coefficient, i, j)`，`j=-1` 表示线性项。对于 `r_k += c z_i z_j`：

\[
\bar z_i\mathrel{+}=c\bar r_kz_j,\qquad
\bar z_j\mathrel{+}=c\bar r_kz_i.
\]

当 i=j 时两项都要累加。此规则是多项式微分，不依赖构造完整 9×30 Jacobian。

边界直接 `seed₀=10 b⊙(value−target)`，高阶 seed 为零。block 内累积 loss 后由明确的 FP64 reduction 得到全局标量；不能未经测试使用原子浮点求和并许诺逐位重放。

### 9.2 归因对照

至少保留：当前 leaf-autograd；同目标编译 tensor residual/seed；显式 CPU/数学 reference；显式 CUDA 候选。不能把“已有 TorchJetCompiled 激活”误记为整个 loss/seed 都已编译。

先只换 seed 生成，不同时换 GEMM、loss、batch、precision 或 optimizer；随后再与 dense/split/compact 三种布局做组合消融。

本轮 9 个任意 jet/权重用例已与交接包 `common.residual_per_point` 的原 autograd seed 一致，包含空 batch、非均匀与零权重、不同幅值。它尚未验证新 CUDA reduction、Graph replay 参数更新和端到端收益。

---

## 10. IO 与计算成本：可以主张什么，不能主张什么

### 10.1 单层逻辑容量与运算

记输入/输出 channel 为 `Cin,Cout`，有效系数行 `M_eff`；原 dense 为 `BQ`，mixed 为 `NiQ+Nb`。按普通乘加计 2 FLOPs：

\[
F_{\rm affine}\simeq2M_{\rm eff}C_{in}C_{out},\quad
F_{\rm dgrad}\simeq2M_{\rm eff}C_{in}C_{out},\quad
F_{\rm wgrad}\simeq2M_{\rm eff}C_{in}C_{out}.
\]

第一层坐标 seed 的零/一阶结构可能有额外消除机会；若启用需另作对照，不能同时归给边界分路。

FP64 中一份 H 或 barH 是 `8BQC` 字节；融合 `D@W→JetVJP` 可以消除 barH 的一次写入和一次读取，即逻辑 `16BQC` 字节。**Dprev 仍要供下一层反向或 wgrad 使用，不能直接宣称所有中间状态均消失。**

checkpoint 压缩、重算、融合各有独立代价：重算 Z 可能需重新 GEMM；保存 a1 是 `8BC`；wgrad 仍有跨点归约。所有 baseline 使用相同 stable aux 后再比较，不能把精度修复成本混进融合收益。

### 10.2 有收益的条件

一个用于筛选而非预测保证的模型为

\[
T_U-T_F\approx T_{\rm eliminated\ actual\ transfers}+T_{\rm eliminated\ launches}
-T_{\rm layout}-T_{\rm sync}-T_{\rm spills}-T_{\rm recompute}-T_{\rm lost\ GEMM\ efficiency}.
\]

`16BQC` 是逻辑流量，不等于实测 DRAM 减少；中间张量可能被 L2 命中。stage 时间有重叠时，应分析依赖关键路径，不把所有 profiler 事件简单相加。

使用多资源下界/筛选量

\[
T_{\rm tile}\gtrsim\max\left(
\frac{F_{\rm MMA}}{P_{\rm MMA}},
\frac{F_{\rm scalar64}}{P_{\rm scalar64}},
\frac{V_{\rm SMEM}}{BW_{\rm SMEM}},
\frac{V_{\rm DRAM}}{BW_{\rm DRAM}},
T_{\rm exp/tanh},T_{\rm dependence}
\right).
\]

这里每个吞吐或流量均需来源/实测，不能把 BF16 的 FA4 吞吐常数代入 FP64，也不能用 occupancy 预测代替实际 active warps。

### 10.3 与 FA1 的边界

本版有明确的逻辑容量、运算数、特定中间张量消除及性能归因模型。**没有给出任意合法执行图的 IO 下界，没有证明我们的 tiling 在某一计算模型下达到最优，也没有消除全部 checkpoint 的全局最优证明。** FA1 的理论结论不能仅因项目名字带 Flash 而继承。[W10]

---

## 11. FA4 与 MLSys：迁移机制，而非复制硬件路径

### 11.1 FA4 的适用启发

FA4 针对 Blackwell 的非对称硬件增长，强调矩阵计算之外的 shared-memory、指数运算与调度成本，并联合调整算法和流水线。[W03] 对本项目的直接启发是：**TMA/GEMM 加快后，重新测量剩余代数、状态搬运与求解控制，而不是继续默认搬运最贵。**

FA4 的 TMEM、2-CTA、低精度 exponential 近似不是 H100 FP64 现成配方。尤其不能把用于低精度 attention 的 exp2 近似替换 stable a1 所依赖的 FP64 exp，再沿用旧容差。

代码阅读入口：[W04,W05]。当前 `softmax.py` 分开维护 row max/sum 与最终缩放；`flash_fwd_sm100.py` 将 softmax、correction、pipeline 状态和资源参数分开。其条件 rescaling 必须保持累积状态的尺度一致，不是无条件删除“小修正”。这些代码来自本次读取的 main，**未锁成实验 commit，也没有在本容器运行 FA4**。

对应到 FlashNS 的提案：

| FA4 机制启发 | FlashNS 决策 | 验证方式 |
|---|---|---|
| 瓶颈随硬件/算法迁移 | 把 seed、wgrad、标量导数和控制成本纳入账本 | 未插桩整步与分阶段 trace |
| 状态解释与运算取消一起设计 | 先证明边界 Q0/混合阶足够，再删导数 | P4/P5、dense/split/compact |
| producer/consumer 有明确状态所有权 | jet 组、零阶行、aux、loss partial 各自明确 owner | 边界 tile、canary、Sanitizer、stream |
| 特化资源而不是一种配置通吃 | 按实际 packed 行数/宽度冻结计划 | 同形 U/F、本机库强基线 |

### 11.2 更多 MLSys 工作的具体取舍

| 原工作 | 相关机制 | 本项目采用/不采用 |
|---|---|---|
| Checkmate，MLSys 2020 [W06] | 对保存与重算进行成本约束优化 | 给 H+a1、保存 Z0、层 checkpoint 重算分别记账；不用“重算永远省”假设 |
| Bolt，MLSys 2022 [W07] | 以硬件原生可配置模板缩小搜索与性能差距 | 库计划作为强基线，有限 tile/owner 搜索；不先替换全部 GEMM |
| Reducing Activation Recomputation，MLSys 2023 [W08] | 选择性重算而非统一重算 | 将数值必需状态与可廉价重建状态分开；精度合同优先 |
| DynaFlow，MLSys 2026 [W09] | 逻辑程序与设备内物理调度解耦 | dense/split/compact 共用数学 IR；仅在资源互补证据成立后试并行 |
| Flashlight，MLSys 2026 [W11] | 编译器内融合比固定模板具有更广表达性 | residual IR 同时产生编译 tensor 基线和显式路径；不宣称该系统已支持 NS |

这些是论文机制的有限借鉴与本文设计推论。没有安装/跑过这些系统，不把其公开 speedup 转移为 FlashNS 的性能预测。

---

## 12. 近邻工作审查：必须缩紧新颖性表述

本轮找到直接近邻：Cao、Lu、Zhang 的 **Verified residual-specific explicit derivative kernels for physics-informed learning and discretized PDE adjoints**，arXiv:2606.29702，2026-06-29。[W12]

论文 §2.2 已包含按残差选择 partial jet、逐层显式空间导数、将导数状态沿 batch 打包 GEMM；§3.1.1 明确采用 IC/BC/PDE all-points batching，容忍部分无用导数以减少小调用。其参数训练反向仍由 reverse-mode differentiation 完成。

因此以下不能直接作为新颖性标题：

> “首次用 Taylor jet 计算 PINN 导数”“首次把导数堆叠成 GEMM”“首次按 PDE 所需导数做特化”。

本项目可检验的区别应写为：

| 候选贡献 | 需要补齐的证据 | 当前状态 |
|---|---|---|
| 原生第一参数 VJP 与 stable checkpoint 的组合 | 全阶参数 seed、状态反例、尾部范围、强 compiled baseline | 数学论证＋CPU；已有部分历史 CUDA |
| 按点需求的 mixed-order packed 流 | 等价证明、相同 GEMM 次数、真实 GPU overhead 与完整求解 | 本版 CPU 原型，GPU 未做 |
| 函数值边界与高阶内点共享 affine/wgrad | 对 all-points 与两分路的匹配比较 | 数学＋CPU 通过，性能未知 |
| 针对同一 FP64 目标的可归因成本与收敛实验 | 固定停止规则、多个新种子、失败预算和 setup 分项 | v7 历史基线；新实验待做 |

这里只检查了该文相关章节，没有完成其实现的复现、全部附录/仓库审计，也没有完成全领域新颖性检索。**以上是候选贡献边界，不是“已经证明无人做过”。**

---

## 13. OpenAI 论文线：从已适配 RHS 推进到共享传播子

### 13.1 已有成果准确归类

上传记录已完成固定版本形式化构建及两个 Comparator 的四个声明核验，另完成论文 §7 局部复数波幅 RHS 的 CPU/GPU 数值适配。[H05] 本轮没有重新重放，因此本文使用“交接包记录”，不声称新验算形式化结果。

已适配的是局部幅度/压力算子，不是全局 profiles、全部 cutoff、全部修正层和奇性极限。Kovasznay PINN 的 solver speedup 不能写成官方构造的求解倍率。

### 13.2 局部约束的复核

记实向量 `n,n'`、线性算子 K、阻尼 δ、复幅度 t、源 f。当前适配为

\[
c=\frac{n\cdot(Kt+f)-n'\cdot t}{|n|^2},\qquad
 t'=-Kt-\delta t-f+nc,\qquad \pi=\frac{ic}{km},\quad km\ne0.
\]

本式对应已适配的论文局部对象。[W02,H05] 直接代入得

\[
\frac{d}{ds}(n\cdot t)=n'\cdot t+n\cdot t'=-\delta(n\cdot t).
\]

所以切向初值维持约束，非零缺陷按阻尼衰减；这不意味着通用三维 NS 压力都能用一个标量公式代替 Poisson。

本轮 20 个复数随机用例与独立 4×4 saddle system 求解对照，并检查缺陷恒等式。它们是有限局部代数测试，不复现完整官方流。

### 13.3 直接来自式 (7.18) 的可计算复用

在论文的二维移动标架中，记局部演化参数 s、模式 m，系统可写为

\[
a_m'=[A_0(s)-m^2d(s)I_2]a_m+g_m(s).
\]

其中 A0 与 d 对同一几何族中的各 m 共享；论文式 (7.18) 给出齐次传播子的关系。[W02，印刷页 78，式 (7.17)–(7.18)]

\[
\boxed{
V_m(s,t)=\exp\left[-(m^2-1)\int_t^s d(\xi)\,d\xi\right]V_1(s,t).
}
\]

**本文的复核推导：**令标量因子为 ρ。由 `ρ'=−(m²−1)dρ`、`V₁'=(A₀−dI)V₁`，有

\[
(\rho V_1)'=[A_0-m^2dI](\rho V_1),\quad(\rho V_1)(t,t)=I.
\]

由有限维线性初值问题唯一性得到该式。仅用到了标量与矩阵可交换；**不要求 A0 在不同时刻彼此可交换**。本轮用非对易的时间变化 A0 和四个模式作 CPU 独立积分对照。

### 13.4 不能忽略源项

受迫解为

\[
a_m(s)=V_m(s,t_0)a_m(t_0)+\int_{t_0}^sV_m(s,\tau)g_m(\tau)\,d\tau.
\]

**不能把完整 m=1 受迫解乘上一个最终 ρ。** 一个反例：`A0=0,d=1,g_m=1,a_m(0)=0`，则

\[
a_m(s)=\frac{1-e^{-m^2s}}{m^2},
\]

它一般不等于 `e^{-(m²−1)s}(1−e^{-s})`。本包保存此反例，防止为了“共享计算”改掉积分核。

### 13.5 新优化建议：几何分组＋局部传播复用

```text
按相同几何系数/时间区间分组，模式与源项作为 batch 维
                 ↓
每组每局部区间：求基础 2×2 传播子 V1 与积分 D=∫d
                 ↓
各模式：rho_m=exp(-(m²-1)D)，恢复区间 Vm
                 ↓
每个源项：保持各自 Duhamel / 等价数值积分
                 ↓
切向重建＋局部压力＋约束检查＋完整误差验收
```

复用 key 至少包含：来源版本、几何参数及系数 hash、移动标架版本、区间端点/节点、精度/误差策略；模式 m 不进入共享的 A0 key，但若 A0 或几何事实上依赖 m，就不能共享。

现有随机独立标签基准不一定具有共享几何，不能凭空当成高复用工作负载。应另建立真实同几何多模式的受控族，并给 baseline 完全相同输入与误差要求。

如果用正交标架 E，`t=Ea` 的方程包含 `−EᵀE' a`；不可以只把维数从 3 改为 2 而删去连接项。论文原标架未必正交，应使用其真实左逆及导数，不强套 Eᵀ。

**数值约束：**按短区间传播，避免形成病态全局基本矩阵再求逆；不要预存所有两时刻传播子而引入 O(T²) 容量。高 m 的指数阻尼下溢及强迫积分边界层需单独处理。若增加缩放状态，应同时推导源项、输出和伴随映射。

### 13.6 成本与验收

设 G 组几何、M 模式、S 源项、T 区间。独立积分可能重复 G×M 次系数/基础矩阵工作；共享路线将此部分变为 G 次基础工作加 G×M 次标量恢复，但源项求值和积分通常仍依赖模式/源，不能从总时间中删除。

下一实验应测 `M=1/4/16/64`、`S=1/8`、不同 T 与阻尼范围；比较相同容差下的完整轨迹计算。计入分组、基础构建、求积、重建、独立验证、传输和冷/热 setup。不能拿历史 resident RHS 的巨大倍率当作轨迹 speedup。

这是利用论文已有恒等式设计计算复用，**不把式 (7.18) 重新包装成我们的数学发现**。

---

## 14. 逐命题实验覆盖矩阵

| 命题/机制 | 本轮 CPU 覆盖 | 目标 GPU 必须新增 | 拒绝条件 |
|---|---|---|---|
| P1/P2 jet 与参数 VJP | d2/d3 全 mixed 输出、任意阶 seed、nested AD；d1–4 对偶 | Q10/Q20、真实宽度、实际源码编译、非默认流 | mixed factorial/完整参数不一致 |
| P3 stable 状态 | H-only 反例、标量尾部、四阶激活探针、原 header HOST | 对当前设备 libdevice 的普通/尾部高精度检查 | aux 丢失、静默清零、用 atol 掩盖小导数 |
| P4 边界拆分 | 非均匀/零权重、空组、五步 SGD | 实际 Adam/线搜索、重复点、重排、Graph | 分母/压力边界/更新次数改变 |
| P5 compact | dense/split/compact loss 和全部参数一致 | 库 GEMM 首版、tile 边界、各阶段资源与完整求解 | 额外 pack/launch 成本大于收益 |
| 显式 residual seed | 9 个任意 jet 用例 vs 原 loss autograd | CPU/compiled/CUDA，同一标量归约与 seed | loss 对但 seed 错；漏项/漏因子 |
| IO 成本 | 行数/容量/相关项精确计数 | 已授权 profiler 实际流量或 null | 逻辑字节冒充 DRAM 实测 |
| OpenAI 局部投影 | 20 例 saddle 对照与约束 | 来源参数域、频率边界、真实轨迹 | 约束失真、离散频率误求导 |
| OpenAI 多模式共享 | 非对易 A0、4 模式；受迫反例 | 共享族、误差对齐、端到端成本 | 源项被整体缩放、假造复用 |

### 14.1 正确性扩展的最小集合

保留冻结 Kovasznay 为主问题；新增问题不是更改主问题。先测 C=32/64/128 与奇数 channel、小 B、不整除 tile、多个坐标幅值/权重。比较边界占比 0/5%/20%/50%/80% 时，每个 case 内固定相同点集、分母和目标。

扩大至其他黏度、3D 稳态强迫 manufactured solution、不同边界算子时，应各自建立正确参考与停止条件。非定常三维模型要明确时间导数和混合时间空间项；不能将 Q20 空间内核改名当作完整四变量支持。

### 14.2 性能消融

第一轮先固定 backend 的 GEMM/activation，比较 3×2 的小矩阵：

| 布局 | 原 autograd seed | 显式/编译 seed |
|---|---|---|
| all-points dense | D-A | D-E |
| 两分路 split | S-A | S-E |
| mixed-order compact | C-A | C-E |

之后再对赢家分别替换 library、既有 F、TMA，保证相同 owner 主循环的 U/F 配对。通用边界消除、compiled seed 也应给强基线使用；不要故意让 baseline 计算无用导数来抬高“自研 kernel”收益。

### 14.3 完整求解与统计

先用旧种子回归；pilot 与正式新种子分离。可沿用 v7 的 pilot 361901/361902、正式 75201–75210，但应在观察正式结果前冻结 backend、预算、停止规则和调参记录。

每个 seed 必须报告初始化、目标评估次数、Adam/L-BFGS 更新数、是否达标、达到同一停止标准的总时间、optimization 热阶段时间、process 总时间。不得静默过滤不收敛种子。若有失败，不以只对共同成功者的 speedup 代表整组性能；另报告固定预算成功率和失败时间。

配对 ratio 在同种子上计算。置信区间的采样单位应是独立 seed/运行块，不是把几千个相关步骤当成几千次独立实验；小样本区间需说明限制。交替 backend 顺序、跨运行块/日期复测。1.10× 是研发目标，不是数学验收线或论文必需门槛。

六个原停止阈值和 500-block 协议从 `evidence/frozen_solver_protocol.json` 加载，不使用旧默认值。最终独立 scalar AD 验收频率/范围保持相同，不能以少验收制造加速。

---

## 15. PR/里程碑：一次只证明一个机制

| 阶段 | 交付 | 合入条件 |
|---|---|---|
| M0：数学合同 | JetSpec、P1–P4、seed 表、FP64 失败域 | 数学审查；本版 CPU 测试接入 CI |
| M1：显式 residual/seed | 稀疏 IR、compiled tensor、CUDA 候选 | 任意 jet/权重、完整 gradient、实际更新、Graph |
| M2：split 参考 | 稳定 value-only 边界与梯度相加 | 保持原归一化/压力 gauge/优化器次数 |
| M3：compact 库路径 | 静态两段 packed 行、一次 wgrad、零阶 bias gather | 与 D/S 对照，无 per-layer pack 膨胀；收益可归因 |
| M4：受控融合 | 在 M3 形状上决定 epilogue/owner/TMA 是否要改 | 同 mainloop U/F、Sanitizer、资源与整步结果 |
| M5：完整求解复测 | 新种子、强基线、失败与预算、同验收 | 达标时间或容量收益稳定，回退可解释 |
| O1：论文传播子复用 | 真实共享几何族＋V1/阻尼构建＋Duhamel | 相同轨迹误差，不遗漏源项或连接项 |

不要求全部阶段完成才可整理第一篇论文；但声称的每个机制必须有相应证据。数学不承担系统性能证明的工作，benchmark 也不替代等价性证明。

---

## 16. 本轮实际执行与交付

### 16.1 23 组检查的真实边界

`reports/cpu_validation.json`：**23/23 通过**；独立核心 20 组，读取原交接源码后的集成 3 组。

覆盖：basis/成本计数；四种维数的 JVP/VJP 对偶；三阶空间需求的四阶激活探针；标量 stable aux 尾部；二维/三维所有输出 mixed jet 及参数伴随；六种内点/边界/幅值分组的 dense/split/compact/nested 对照；五次 SGD；非法/零权重；局部复数投影；式 (7.18)；错误受迫缩放反例；原 loss seed；原 stable Python AD；原 `.cuh` 的 HOST 编译。

这些是分组测试数，子用例数在 JSON 分列。不是 23 次 GPU 实验，也不是 23 个完整 NS 求解。

### 16.2 复现

```bash
# 当前环境中实际执行过。CPU-only 环境即可，不安装 CUDA 或改驱动。
python tests/verify_math.py --output reports/cpu_validation_core.json

# 可选：需要用户上传交接包的完整解压目录。
# 原 header 仅用 g++ 编译为 HOST 动态库；不会加载交接包中的历史 .so。
python tests/verify_math.py \
  --handoff-root /path/to/FlashNS_GPT_Pro_Handoff_2026-09-09 \
  --output reports/cpu_validation.json

# 只复算历史 JSON 的配对统计，绝不是重跑 GPU。
python tests/audit_handoff.py /path/to/FlashNS_GPT_Pro_Handoff_2026-09-09
```

依赖：Python 3.11+、NumPy、SciPy、mpmath、CPU PyTorch；可选 g++。本次具体版本写入报告。测试程序不会修改原交接目录；可选 HOST 编译使用临时目录。

`reports/handoff_audit.json` 记录所查文件 SHA-256、24 个最终组合、17 个首轮成功、31 次尝试以及配对统计一致性。原 ZIP 的 manifest 校验通过 3,207 个 payload 文件，177,262,242 字节；manifest 自身不纳入自哈希。这只验证包内完整性，不验证外部世界的测量真实性。

### 16.3 论文的建议陈述

> We study the state and execution requirements of a fixed high-order scientific objective. We derive an explicit parameter-adjoint contract, characterize a numerically insufficient checkpoint representation, and evaluate demand-segmented dataflow that removes unused boundary derivatives while preserving globally normalized gradients.

其中 “evaluate” 的性能部分须待目标 GPU 和完整求解结果补齐；当前版本不能添写尚未取得的速度数值。

**贡献不在于把已有 Taylor 工具重新命名，而在于可审计地回答：此目标需要哪些状态、哪些运算可删、删后如何仍保持有效矩阵批量，以及误差和完整求解代价怎样变化。**

---

## 17. 来源与审查范围

### 上传交接材料（原文副本与本轮复核）

- **[H01]** `evidence/handoff_v7.md`；原 `01_ITERATION_RFC_V7_ZH.md`，历史结果/下一轮提案边界。
- **[H02]** `evidence/handoff_key_results.json`、`reports/handoff_audit.json`；原 suite 配对统计复算。原 suite 位于交接包 `flashns/experiments/pinn_solver/artifacts/extension1/suite.json`。
- **[H03]** `evidence/historical_pinn_solver_results.md` 与 `evidence/frozen_solver_protocol.json`；原 `problem.py` 等源码的 hash 见 audit。
- **[H04]** `evidence/historical_hopper_results.md`；`jet_stable.py`、`stable_jet.cuh` 的 hash 和 CPU 集成检查见 JSON。
- **[H05]** `evidence/historical_openai_adapter_results.md`；固定官方 commit `8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538`；论文历史字节 SHA-256 `0e779481c4da40bd28d1e642e1d8ca57447d129610df28dfa5a11e9af8ae228f`。本轮没有下载同 PDF 到容器重新核对其字节 hash。

### 公开一手资料（2026-09-09 核对）

- **[W01]** OpenAI, *On the Navier–Stokes Millennium Prize Problem*：https://openai.com/index/navier-stokes-solution/ 。本文不是官方结论独立裁决。
- **[W02]** OpenAI, *Finite Time Blowup for Navier–Stokes*：https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf 。本轮重点读 §7 和式 (7.13)、(7.17)、(7.18)，核对印刷页 76–78 图像；未重新审阅完整 166 页。
- **[W03]** Zadouri et al., *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*，2026，v1：https://arxiv.org/html/2603.05451v1 。重点 §3。
- **[W04]** FA4/FlashAttention 官方源码 `softmax.py`：https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/flash_attn/cute/softmax.py 。本轮阅读相关状态更新入口，不代表锁定/运行全部代码。
- **[W05]** 官方 `flash_fwd_sm100.py`：https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/flash_attn/cute/flash_fwd_sm100.py 。重点参数表、softmax/correction 与 pipeline 状态。
- **[W06]** Jain et al., *Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization*，MLSys 2020：https://proceedings.mlsys.org/paper_files/paper/2020/hash/0b816ae8f06f8dd3543dc3d9ef196cab-Abstract.html 。本文使用公开摘要级机制，不声称复现其求解器。
- **[W07]** Xing et al., *Bolt: Bridging the Gap between Auto-tuners and Hardware-native Performance*，MLSys 2022：https://proceedings.mlsys.org/paper_files/paper/2022/hash/1f8053a67ec8e0b57455713cefdd8218-Abstract.html 。机制级参考。
- **[W08]** Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*，MLSys 2023：https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html 。机制级参考。
- **[W09]** Pan et al., *DynaFlow: Transparent and Flexible Intra-Device Parallelism via Programmable Operator Scheduling*，MLSys 2026：https://proceedings.mlsys.org/paper_files/paper/2026/hash/bbd7d8bd780fcf7143add2317ba04638-Abstract-Conference.html 。机制级参考。
- **[W10]** Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*，2022：https://arxiv.org/abs/2205.14135 。只引用研究方向，不转移其 IO 最优性结论。
- **[W11]** You et al., *Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants*，MLSys 2026：https://proceedings.mlsys.org/paper_files/paper/2026/hash/bc52716d13d2d72ea0f335667d86c0f8-Abstract-Conference.html 。机制级参考。
- **[W12]** Cao, Lu, Zhang, *Verified residual-specific explicit derivative kernels for physics-informed learning and discretized PDE adjoints*，2026：https://arxiv.org/abs/2606.29702 ；https://arxiv.org/pdf/2606.29702 。核对 §2.2、§3.1.1，未复现其代码。

参考资料用于明确继承关系与设计依据；新的 GPU 数字、算法最优性和全局数学证明均不能由这些引用自动推出。


---

## 附录：本次研究定位的德语对照

| 中文 | Deutsch |
|---|---|
| 现在应补齐数学论证，同时扩大实验覆盖。 | Jetzt sollten wir die mathematischen Begründungen vervollständigen und zugleich die experimentelle Abdeckung erweitern. |
| 重点是把已有实现背后的机制写清楚。 | Im Mittelpunkt steht, die Mechanismen hinter der vorhandenen Implementierung präzise darzustellen. |
| 要证明阶乘归一化的三阶 jet 和显式 VJP 与原始 loss 的参数梯度一致，包括混合导数、权重和边界项。 | Wir müssen nachweisen, dass Jets dritter Ordnung mit Fakultätsnormierung und das explizite VJP dieselben Parametergradienten wie die ursprüngliche Verlustfunktion liefern, einschließlich gemischter Ableitungen, Gewichte und Randterme. |
| 要解释反传需要保存哪些状态，以及为什么 stable aux 必要。 | Wir müssen erklären, welche Zustände für die Rückwärtsrechnung gespeichert werden müssen und warum ein numerisch stabiler Hilfszustand erforderlich ist. |
| 应区分实数上的等价证明与 FP64 下的误差、饱和和下溢边界。 | Äquivalenzbeweise in reeller Arithmetik müssen von Fehlern sowie Sättigungs- und Unterlaufgrenzen in FP64 unterschieden werden. |
| 要推导融合省掉哪些中间张量，以及同步、布局转换和寄存器压力何时抵消收益。 | Wir müssen herleiten, welche Zwischentensoren durch Fusion entfallen und wann Synchronisation, Layoutumwandlungen und Registerdruck den Nutzen aufheben. |
| 目前不能声称我们拥有类似 FA1 的最优性证明。 | Derzeit können wir keinen Optimalitätsbeweis beanspruchen, der mit dem von FA1 vergleichbar wäre. |
| 内点高阶 jet 与边界函数值可以分别计算，但必须保持原归一化与梯度相加关系。 | Jets höherer Ordnung an inneren Punkten und Funktionswerte an Randpunkten können getrennt berechnet werden, wobei die ursprüngliche Normierung und die additive Zerlegung des Gradienten erhalten bleiben müssen. |
| 还要检查减少的计算是否转化成实际收益。 | Außerdem müssen wir prüfen, ob der verringerte Rechenaufwand tatsächlich zu einem praktischen Vorteil führt. |
| Taylor jet 本身是已有数学工具。 | Taylor-Jets sind bereits etablierte mathematische Werkzeuge. |
| 我们要提炼的贡献是：哪些状态必须保留、哪些计算可以消除、怎样更有效地组织数据流。 | Der herauszuarbeitende Beitrag besteht darin zu bestimmen, welche Zustände erhalten bleiben müssen, welche Berechnungen entfallen können und wie sich der Datenfluss effizienter organisieren lässt. |
| 每个命题都应对应正确性检查、匹配消融和不同规模的实验。 | Jedem mathematischen Satz sollten Korrektheitsprüfungen, Ablationsstudien unter vergleichbaren Bedingungen und Experimente in unterschiedlichen Größenordnungen zugeordnet werden. |

关键词：**die Äquivalenz**（等价性）、**der Hilfszustand**（辅助状态）、**der Unterlauf**（下溢）、**die Randbedingung**（边界条件）、**die Fakultätsnormierung**（阶乘归一化）、**die Ablationsstudie**（消融实验）、**der Datenfluss**（数据流）、**der Rechenaufwand**（计算量）。
