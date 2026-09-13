# OpenAI 官方构造：CPU 构建、Comparator 与有限局部数值适配

2026-09-09，固定官方仓库 commit `8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538`，完成原始默认目标 CPU 构建和两个未修改的 Comparator 挑战。挑战列出的 **4 个声明均获 nanoda 与 Lean 默认 kernel 接受**。另将论文 §7 的局部复数波幅方程适配为 FP64 CPU/GPU 算子，完成独立高精度、导数和轨迹检查。

官方来源：[NavierStokesAndEuler 固定版本](https://github.com/openai/NavierStokesAndEuler/tree/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538)、[官方论文](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf)。本地 [来源登记](../sources/openai_ns_formal/registry.json) 和 [适配器契约](../adapters/openai_ns_formal.json) 已记录实际执行状态。

## CPU 构建与证明重放

Lean 为 `4.34.0-rc2`，依赖按原 `lake-manifest.json` 固定。使用公开 Mathlib 缓存，未替换官方证明。默认 `lake build` 构建 NavierStokes、Euler 和 ComparatorChallenges，共 11,251 jobs，退出 0；2,496 个 tracked source 文件在各阶段前后哈希一致，git diff 为空。

| 阶段 | 声明/目标 | 结果 | 实测 wall time |
|---|---|---|---:|
| 原始 lake build | 三个默认目标 | 通过 | 1,054.387 s |
| Checker 可执行文件构建 | comparator、lean4export | 通过 | 9.073 s |
| Navier–Stokes Comparator | R3、periodic 两个声明 | 双 kernel 接受 | 935.729 s |
| Euler Comparator | R3 breakdown、compact smooth singularity 两个声明 | 双 kernel 接受 | 2,258.550 s（包含 GPU 计时期间的暂停） |

Euler 的两段已记录暂停共 528.800 s，包含在上面的原始 wall time 中；这些时长用于记录执行过程，不是未经暂停的 CPU 性能 A/B。

四个声明为 `NavierStokes.Comparator.navier_stokes_breakdown_R3`、`NavierStokes.Comparator.navier_stokes_breakdown_periodic`、`Euler.euler_breakdown_R3`、`Euler.exists_compact_smooth_euler_singularity`。原 permitted axioms 仍为 `propext`、`Quot.sound`、`Classical.choice`，`enable_nanoda=true`。每个日志均有 nanoda、Lean 接受和 Comparator 成功结束的记录。

这些是当前固定挑战配置下的实际重放结果。CPU 阶段包含构建、导出与 kernel 检查，未进行 GPU 证明加速。构建用 UID 65534，两个 Comparator 使用 UID 65533 的独立干净检查目录；不复制本项目构建缓存，仅复用锁定依赖和 Mathlib 缓存。

工具版本为 Comparator `19e111e2141cf333c7daff0f64c5f24acc91dd2e`、lean4export `cacf989bd75f608700820f6afc595f32e7a99a4d`、landrun `811cfff51ceaf3d9843708aa6d22e9b84ccac8b4`、nanoda `05055695879dfebb6628a67da88ceca6cd6b0421`；编译工具 Go 1.27.1、Rust/Cargo 1.98.1。版本、下载和二进制哈希见 [工具报告](../artifacts/openai_ns_formal/build/comparator-tools.json)。

宿主内核 Landlock ABI=4。固定 Comparator 源码本身给 landrun 传递 `--best-effort`，该行为未修改。容器没有用户 systemd，因此在原 Landrun 限制之外使用继承到 exec 子进程的 libseccomp 规则，拒绝 AF_UNIX `socket` 和 `socketpair`，验证 no_new_privs 和无有效 capabilities；允许/禁止目录写入及 exec 继承探针均通过。wrapper 仅给 Landrun 加入 `LEAN_NUM_THREADS=16`。这份环境说明不等同于声称宿主支持后续所有 Landlock ABI 的功能；完整探针见 [隔离记录](../artifacts/openai_ns_formal/comparator/checker-context.json) 和 [日志](../artifacts/openai_ns_formal/comparator/sandbox-self-test.log)。

原始证据：[lake build](../artifacts/openai_ns_formal/build/lake-build.json)、[NS 重放](../artifacts/openai_ns_formal/comparator/comparator-navier-stokes.json)、[Euler 重放](../artifacts/openai_ns_formal/comparator/comparator-euler.json)。脚本执行版本单独存放于 executed_sources（本地完整归档，未随 GitHub 源码发布），避免后续格式化覆盖来源证据。

## 有限局部对象

论文 PDF 共 166 页，SHA-256 为 `0e779481c4da40bd28d1e642e1d8ca57447d129610df28dfa5a11e9af8ae228f`。数值适配对应公式 (7.3)–(7.6)、(7.13)，以及 `NavierStokes/TangentProjection.lean` 中的 `tangentProj`、`projectedRhs`、`pressureCoefficient`。固定 Lean 注释使用早期编号 (27)/Lemma 8.4；已逐项检查本 PDF (7.13) 的代数对应，没有仅靠旧编号映射。

给定实向量 n、n′、阻尼 δ、线性算子 K、非零离散频率 k,m，以及复数幅度 t 和源 f，计算

\[
\alpha=\frac{n\cdot(Kt+f)-n'\cdot t}{|n|^2},\qquad
t'=-Kt-\delta t-f+n\alpha,\qquad
\pi=\frac{i\alpha}{km}.
\]

因此切向缺陷 c=n·t 满足 c′=−δc；初始切向的轨迹保持该约束，非零缺陷按精确标量指数衰减。每个输入含 24 个 FP64 分量，输出为幅度导数和压力的 8 个 FP64 分量，复数实虚部分开存储。

有限测试参数为 F∈[0.5,2]、a∈[2.25,5]、b∈[−2,2]、R∈[0.5,2]，局部 g=F(−a,b)、F_Z=G_Z=0、u*=1；ε 取 2 的 −8、−12、−16、−24、−32 次方，Ls∈{16,32,64,128}，m∈{−3,−1,1,2}。k=ceil(ε^−1/2)，k·p 按最近非零整数选取，离散取整在求导前固定。样本一半具有初始切向约束，另一半包含缺陷；另加入 b=0 的频率边界样本。

这组系数满足被测局部代数关系和正 λ0 条件。它没有实现论文 Theorem 4.6 的全局 profiles、q*、全部 cutoffs、修正层级或爆破极限。独立 [Kovasznay PINN 求解](pinn-solver-results.md) 与旧 Euler 冻结样条也各自有不同 source_id，不能合并为本构造的数值实现。

## 数值验收

独立参考直接求解公式 (7.5) 加约束组成的 4×4 saddle system，未调用被测投影闭式公式。100/120 位 mpmath 结果转成 FP64 后一致；GPU 检查使用 4,096 个样本，其中 96 个逐点对照高精度 saddle system。原算子阈值固定为 `1e-11 + 5e-13*abs(reference)`；NumPy、GPU eager、GPU fullgraph compiled 和非默认 stream 均通过。

对另 12 个标签检查脉冲坐标的一阶、二阶和三阶偏导，离散频率、复数幅度和源固定；Torch jacfwd 与 100 位 mpmath 微分在 `1e-10 + 2e-10*abs(reference)` 下通过。这是局部 RHS 的偏导检查，没有把它称为完整 ODE 轨迹的总导数。

8 条轨迹分别以投影公式+DOP853、saddle system+Radau 积分到时间 8，在 129 个时刻比较，并检查 c 的精确指数衰减。轨迹阈值 `3e-9 + 3e-8*abs(reference)`、缺陷阈值 `3e-10 + 3e-8*abs(reference)` 均保持固定。开发时 DOP853 的首次积分精度不足，随后将积分器收紧至 rtol=1e-13、atol=2e-15；没有放宽验收阈值。轨迹积分的作用是数值验收，不是 GPU 加速轨迹求解的性能结果。

完整证据见 [数值验证](../experiments/openai_ns_formal/artifacts/numerical-validation.json)，实现见 [projected_amplitude.py](../experiments/openai_ns_formal/projected_amplitude.py)。GPU 批量 RHS 的驻留/含传输 A/B 及首次中断记录在同一实验目录独立保存。

## 批量 GPU A/B

正式采用 [numerical-benchmark-v3.json](../experiments/openai_ns_formal/artifacts/numerical-benchmark-v3.json)：每个尺寸 15 轮，五个实现按固定种子随机交错，同一输入的所有输出均在计时间隔之外检查。CPU 同时测整体数组与 65,536 点分块的相同 NumPy 公式，包含输出分配和拼接；倍率取每轮两种 CPU 中较快者。GPU 驻留使用 CUDA events，含传输列从 pageable CPU 输入开始，包含 H2D、计算、D2H 和同步。

| 样本数 | NumPy 整体 (ms) | NumPy 分块 (ms) | GPU eager 驻留 (ms) | GPU compiled 驻留 (ms) | GPU compiled 含传输 (ms) | 配对较快 CPU / 含传输 GPU |
|---|---:|---:|---:|---:|---:|---:|
| 65,536 | 14.704 | 15.145 | 0.425 | 0.243 | 1.344 | 10.865× |
| 1,048,576 | 518.523 | 282.382 | 1.266 | 0.492 | 48.925 | 5.743× |

驻留 compiled 的配对倍率分别为 61.03×、577.12×，但不包含输入/输出传输，也不代表整个数学构造或轨迹积分的倍率。编译包装及首次调用另外记录为 1.972 s；本次已存在前期验证填充的编译器缓存，不能将其称为完全冷编译成本。

第一次性能运行中断后，增加了逐后端、逐轮日志与增量 JSON。第二次发现该主机的 NumPy 整体算子首调约 0.494 s，随后多次退化到 11–13 s，进程主要消耗内核态 CPU 时间；该默认大页运行再次中断并保留原始观测。第三次在导入 NumPy 前设置 `NUMPY_MADVISE_HUGEPAGE=0`，15 轮完整运行恢复到上表结果，整体/分块 CPU 的内核态时间中位数分别为 0.164/0.026 s。这个仅影响当前进程的分配提示是 [NumPy 官方支持的设置](https://numpy.org/doc/2.0/reference/global_state.html)；宿主机 hugepage、内核、驱动和权限未修改。

该复测说明退化与当前主机上的大页分配提示相关；未做内核跟踪来进一步区分具体机制。首轮和第二轮的慢速/中断结果不用于最终倍率。证据包括 [运行时诊断](../experiments/openai_ns_formal/artifacts/runtime-investigation.json)、[第二轮部分数据](../experiments/openai_ns_formal/artifacts/numerical-benchmark-v2.json) 和 原始日志（本地完整归档，未随 GitHub 源码发布）。
