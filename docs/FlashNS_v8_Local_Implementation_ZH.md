# FlashNS v8 本地落地记录

日期：2026-09-09。范围：按“先落地，再上 SSH”完成本地实现、数学回归及可执行 GPU 验收入口。本轮没有使用远程 SSH 或运行 GPU。

## 已实现

| 部分 | 实现 | 本地证据与剩余验收 |
|---|---|---|
| M0 数学合同 | 原始 v8 包 18 个 payload 及 manifest 原样导入；Q10/Q20 和 stable aux 沿用既有合同 | 原包 23 组检查重跑通过；18 个 SHA256 一致 |
| M1 显式 residual/seed | 九个稀疏多项式、显式转置、compiled tensor 入口、生成的 CUDA/HOST 共享核心 | 随机 jet/权重、完整梯度和实际更新已 CPU 对照；HOST C++ 27 用例通过；CUDA 编译及 Graph 待测 |
| M2 split | 内点完整 Q10，边界仅函数值；共享参数，完整梯度相加 | 与 dense 和独立 AD 一致，保留压力 gauge 与全局归一化 |
| M3 compact 库路径 | 静态 `[Ni*Q+Nb,C]` 混合行，分段非线性直接写入目标张量，单次 wgrad，零阶 bias gather | 8 次 GEMM 的真实调用检查通过；新 native CUDA 源码待 GPU 验收 |
| M4 受控融合 | 暂无新增 TMA/owner 融合 | 先用 GPU 阶段 trace 和整步结果决定，不能将 compact 直接塞进旧 uniform-Q TMA |
| M5 完整求解 | 原 Adam/L-BFGS、固定检查、独立接受、预注册种子、冻结选择、逐进程执行与证据入口已接好 | 尚未执行新 GPU 完整求解或确认统计 |
| O1 论文传播复用 | 原始受控 CPU 参考与反例已保留、重跑 | 真实几何族和带源项传播路径尚未实现，不作为本次 PINN 成果 |

核心代码：[ns_seed.py](../src/flashns/ns_seed.py)、[jet_packed.py](../src/flashns/jet_packed.py)、[pinn_packed.py](../src/flashns/pinn_packed.py)。新 CUDA 源码、生成器、ABI 与实验执行器统一在 [pinn_v8](../experiments/pinn_v8/README.md)，运行方式也见该文件。旧 solver 与旧 GPU 成果文件保持原样。

## 实际验证

[本次验证 JSON](../experiments/pinn_v8/artifacts/local-ready-20260909/validation.json) 记录全部环境、源码 SHA256、命令及检查细节。

- 项目测试：80 passed，0 failed，0 skipped。新增 v8 为 22 项。
- 原 v8 集成检查：23/23 组。
- 生成的 residual/seed 共享 C++ 数学核心：27/27 用例，涵盖 3 种黏性、3 种输入尺度、3 种权重。
- 原问题 2048+512 点、4547 个参数：dense/split/compact × autograd/explicit 六路径，完整 loss/gradient 对照通过。最大梯度误差与容差之比为 0.00021923；逐分量容差 `2e-12 + 2e-11*abs(reference)`。损失差在这六次对照中为零。
- 三次实际 Adam 参数更新与重复边界点用例通过；Dynamo 在 CPU 的 eager backend 捕获检查通过，不代表 GPU Inductor 验收。
- 算子调用检查：dense 8、split 16、compact 8 次 GEMM。compact 的 forward、dgrad、wgrad 都使用混合行维度。

本次 Python 3.12.13，NumPy 2.5.3，SciPy 1.18.1，SymPy 1.14.0，mpmath 1.3.0，PyTorch 2.14.0，pytest 9.1.1。Apple 主机的 HOST C++ 验证不覆盖 CUDA 同步、libdevice 或 GPU 代码生成。

## 科学与执行合同

内点保留完整三阶阶乘归一化 Q10；一般激活支持 Q20。非线性反向保留四阶激活导数所需信息，边界 value-only 同样保留 stable `a1`，不从已饱和的 `tanh` 输出恢复导数。本实现只承诺一阶参数 VJP，不提供 HVP/double backward。

loss 保留原黏性 0.07、梯度项 0.2、边界项 5、压力出流 gauge、固定数据和 6 项停止标准。FP64、TF32 关闭。原冻结扩展协议的 L-BFGS 上限为 500 blocks。显式 seed 使用独立于旧 residual 代码的稀疏项表；其 CPU 与 CUDA 生成器共享同一 IR，另以原包 NumPy 多项式及嵌套 AD 校验。

GPU build 保存实际源文件、命令、日志、二进制哈希；运行拒绝源码改变后的旧 build 或旧 preflight。quick preflight 只用于 Sanitizer，不允许产生性能结论。正式选择必须在新确认种子执行前冻结；失败、预算耗尽及所有尝试的进程成本保留。

本轮没有 GPU 测量。18% 只描述给定点集下逻辑行工作量：25600→20992。宽度 64 的单份隐藏状态由 12.5 MiB 变为 10.25 MiB；stable aux 保持 1.25 MiB。实际 allocator 峰值、整步与达标时间须独立测量。

## SSH 后验收顺序

使用 [run_v8_gpu.sh](../scripts/run_v8_gpu.sh) 先完成目标设备 build、完整 preflight、四项 Sanitizer、15 轮配对 Graph 测量和单独阶段 trace；按结果继续 pilot、旧种子回归、冻结候选及 10 个新种子确认。

首轮通过后仍需比较既有最强适用基线，并按同样机制回移通用优化，才能作库间总体结论。是否投入新的 TMA/epilogue 融合取决于阶段 profile。O1 真实传播复用应在单独工作负载和轨迹误差合同下继续。
