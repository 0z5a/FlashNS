# FlashNS v8：数学证明、数据流提案与 CPU 验证包

入口：[中文 RFC v8](FlashNS_Math_Dataflow_FA4_OpenAI_RFC_v8_ZH.md)。

本包没有新的 CUDA kernel、GPU 性能结果、完整新求解或 Lean 重放。它包含实数等价推导、可运行 CPU 原型、独立/集成检查、历史上传结果的统计复核，以及下一轮实验合同。

## 内容

| 文件/目录 | 内容 |
|---|---|
| `FlashNS_Math_Dataflow_FA4_OpenAI_RFC_v8_ZH.md` | 数学合同、FA4/MLSys 取舍、近邻工作、OpenAI 模态传播复用 |
| `src/reference_kernels.py` | 通用截断 jet、显式 VJP、residual/seed、mixed-order packed MLP、局部波幅参考 |
| `tests/verify_math.py` | 默认 20 组独立 CPU 检查；给定原交接包目录后增加 3 组 |
| `tests/audit_handoff.py` | 只从原始上传 JSON 复算历史统计 |
| `reports/cpu_validation.json` | 本轮 23/23 CPU 分组检查的真实结果 |
| `reports/cpu_validation_core.json` | 独立核心 20/20 的真实结果 |
| `reports/handoff_audit.json` | 24 最终组合、17 首轮成功、31 次尝试；原 JSON 统计一致性 |
| `reports/archive_integrity.json` | 原 ZIP 内部 3,207 payload 文件完整性检查 |
| `reports/cost_model.json` | 18% 行工作减少等精确逻辑计算；不是性能测量 |
| `experiment_plan.json` | 尚未运行的 GPU 与科学验收矩阵 |
| `evidence/` | 原交接材料的未改动副本与原冻结 solver 协议 |

`evidence/` 中历史 Markdown 内部的相对链接仍按原交接包目录组织，需回到完整原交接包访问，不能视为本包新生成的工件。原 ZIP、GPU 二进制、检查点和完整 Lean 树不重复分发。

## 运行

本轮实际环境：Python 3.13.5、NumPy 2.3.5、SciPy 1.17.0、mpmath 1.3.0、Torch 2.10.0+cpu。实际 Python 版本以 JSON 为准；下面不要求升级现有 GPU 主机依赖。

```bash
# 在已有 CPU 依赖的环境中，从本包根目录运行。
python tests/verify_math.py --output reports/my_cpu_core.json

# 可选交接源码集成；传入含 flashns/ 子目录的解压根目录。
python tests/verify_math.py \
  --handoff-root /path/to/FlashNS_GPT_Pro_Handoff_2026-09-09 \
  --output reports/my_cpu_with_handoff.json

python tests/audit_handoff.py \
  /path/to/FlashNS_GPT_Pro_Handoff_2026-09-09 \
  --output reports/my_handoff_audit.json
```

独立检查需要 NumPy/SciPy/mpmath/PyTorch；集成 header 检查另需 g++。不安装、加载或运行 CUDA。集成检查构建的是临时 HOST C++ 动态库，不会加载交接包历史 `.so`，也不读取 pickle/模型检查点。原交接目录只读。

## 如何解读结果

“23 组通过”包含多项子用例，不是 23 个完整 NS 求解。比较的是固定目标下的有限数值测试；实数证明见 RFC，GPU 性能、GPU 同步与正式收敛仍须在目标环境验证。

原三阶 Q10/Q20 表示与 stable aux 已有历史实现；新增 mixed-order 布局和显式 residual/seed 在本包只提供 CPU 参考。OpenAI 模态传播测试采用受控、时间变化且非对易的合成系数，验证论文恒等式与源项边界，不冒充完整论文构造的复现。

完整求解应使用 `evidence/frozen_solver_protocol.json`；它的 500-block L-BFGS 上限不应被旧代码默认 200 blocks 覆盖。新的 GPU 实验须在开跑前固定 baseline、点集、目标、dtype、种子和停止标准。
