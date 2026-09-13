# CUDA jet 实机入口

这部分接入用户在 2026-09-08 提供的 `flashns_cuda_jet` 和 `CUDA_Design_ZH.md`。`input/` 中的九个文件保持原内容；其“未做 CUDA 实测”描述属于输入工件原有状态。本轮实测由外层 `run_cuda.py` 和 `artifacts/` 单独记录。输入原文件中的数学与代码归属保持不变。

在已有 CUDA 环境中，从 FlashNS 根目录执行：

```bash
PYTHONPATH=src python experiments/cuda_jet/run_cuda.py --benchmark
```

依赖 Python、PyTorch CUDA、mpmath 和 nvcc。脚本按实际设备架构编译共享库，用 `ctypes` 调用原始 launcher，并传入 PyTorch 当前 stream。输出采用独立、连续的 `[B,Q,C]` FP64 缓冲区。它提供显式 VJP，不注册 PyTorch double backward。

已实现的实验范围：

- 二维 Q=10、三维 Q=20，均为总次数不超过三阶的阶乘归一化 jet；不同 batch、奇数 channel、非默认 stream 与非法输入检查。
- CUDA 前向对照 Torch 多项式组合；CUDA VJP 对照该组合的自动微分。
- `2→8→8→3` 网络在三个输入尺度下，对照独立 nested AD 的全部输出 jet、二维稳态动量/散度及残差空间梯度 loss、全部权重和 bias 梯度。
- `B=4096/16384, C=64` 的激活前向加 VJP，比较 eager Torch、`torch.compile(fullgraph=True)` 和原生 CUDA。
- `2→64→64→3` 网络的完整一次参数梯度计算，双方使用相同库 GEMM、相同 loss 和输出 seed；进一步交替两种执行顺序进行配对计时。

计时包括对应 GPU 同步和输出分配；编译首调用单列。完整梯度步骤包含输入 jet 构建、forward/dgrad/wgrad、bias 归约和 loss seed，不包括优化器、数据生成、训练收敛或证书验证。完整网络的主计时不含 profiler 开销。profile 中框架算子和其子 kernel 有层级重叠，不能把所有事件时间相加。

100 位精度诊断确认 `1-tanh(z)^2` 的尾部问题：在 `z=15`，一阶导数相对误差约 `1.66e-4`；在 `z=20`，CUDA 结果为零，而参考值约 `1.70e-17`。从已舍入为 ±1 的零阶 H 无法恢复这些信息。需要额外保存稳定计算的导数量或由 Z 重算后重新验证，才能扩展到这类输入。

`validation.passed` 仅针对列出的普通输入，尾部诊断单独为失败。原生内核仍是独立激活/VJP kernel；设计文档中的 GEMM 融合、片上伴随量重建消融、精确 DRAM/occupancy 计数与全训练实验尚待后续实现。当前 ptxas 已报告寄存器与 spill；用户态 profiler 记录算子耗时，未取得硬件计数器数据。

结果见 `artifacts/cuda_a5000.json` 与 `artifacts/nvcc_build.log`；首次计时另存 `cuda_a5000_v1.json`，其实际 runner 留在 `artifacts/run_cuda_v1.py`。项目总报告位于 `../../docs/day0-results.md`。
