# CUDA jet v5：A4000×4

完整实测、验收范围与失败边界见 [中文报告](../../docs/cuda-v5-results.md)。

本目录实现独立 `pinn_jet_local` 工作负载：完整 FP64 jet、stable tanh aux、手动参数 VJP、样本数据并行。它没有调用官方 OpenAI 或 Euler 证明构造。

## 文件与合同

- `generate_stable_header.py` / `stable_jet.cuh`：由保留的原始生成器派生稳定激活；原始输入目录和 v2 实现保持独立。
- `stable_kernels.cu`：激活/VJP 与 Cout=3 的 U3/F3。
- `stable_dgrad.cu`：匹配 N=64 / N=32 的 U/F，固定三阶段。
- `gpu.py`：当前 device/stream 的 C ABI 绑定；检查 dtype、布局、形状、库哈希和不支持的自动高阶反传。
- `common.py`：统一的加权完整目标、手动梯度和独立嵌套 AD 参考。
- `distributed.py`：真实 native 梯度、FP64 SUM、多步 SGD、强/弱扩展和逐 rank 记录。
- `recheck_small.py`：固定参数与点集前缀、交错梯度/SGD 的弱扩展复测。
- `independent_throughput.py`：同一组四个独立任务的 1/2/4 worker 吞吐。
- `validate.py`、`check_sanitizers.py`、`benchmark.py`、`profile_costs.py`、`audit_artifacts.py`：数值、调试、计时、归因和工件审计入口。

张量是同卡连续 FP64 `[B,Q,C]`，Q=10/20，系数为规范化 Taylor。额外 aux 是 `[B,C]` 的稳定 `a1`。默认当前流之外使用时，调用方负责输入就绪；wrapper 将所有使用的张量记录到实际 stream。输出独立分配，不允许 native 原地覆盖。仅手动一阶参数 VJP；不支持 HVP/double backward。

完整目标固定为二维 `2→64→64→3` 网络。`U3/F3` 只在 Cout=3 调专用核，其它 Cout 使用 B1。默认仍为 B1，没有自动性能 dispatch。传给 DP 的 `--backends` 默认是 `B1,U,F,F3`；其它候选有局部验收，不把它们描述为已跑过默认分布式套件。

## 在 CUDA 主机重放

先在独立工作副本中重放，保留已有 `artifacts/`。构建器会写固定的库路径；不得在其它 worker 使用库时重建。当前实测副本位于 `/root/flashns-v5-20260909`。

实际 Python 为 `/venv/main/bin/python`。新主机先采集现有环境，按报告选择依赖；不要把 A4000 环境直接当作 H100 已验证环境。

```bash
export PYTHONPATH=src:.deps
export FLASHNS_PYTHON=/venv/main/bin/python

# 未使用过的报告目录；数值验证和本地 benchmark 拒绝覆盖已有结果。
mkdir -p experiments/cuda_jet_v5/artifacts/replay-01

$FLASHNS_PYTHON scripts/preflight_multigpu.py --expect-gpus 4 --copy-smoke \
  --output experiments/cuda_jet_v5/artifacts/replay-01/environment.json

# 需要固定版本且 include 没有修改的 CUTLASS checkout：
# third_party/cutlass @ cb4247394dd82148787aed73e5dc7cef33cbf862
$FLASHNS_PYTHON experiments/cuda_jet_v5/generate_stable_header.py
$FLASHNS_PYTHON experiments/cuda_jet_v5/build.py
$FLASHNS_PYTHON -m pytest -q

for gpu in 0 1 2 3; do
  $FLASHNS_PYTHON experiments/cuda_jet_v5/validate.py --device "$gpu" \
    --output "experiments/cuda_jet_v5/artifacts/replay-01/validation_gpu${gpu}.json"
done

$FLASHNS_PYTHON experiments/cuda_jet_v5/check_sanitizers.py
$FLASHNS_PYTHON experiments/cuda_jet_v5/benchmark.py --device 0 --repeats 9 \
  --output experiments/cuda_jet_v5/artifacts/replay-01/local_benchmark.json

for world in 1 2 4; do
  $FLASHNS_PYTHON -m torch.distributed.run --standalone --nproc_per_node="$world" \
    experiments/cuda_jet_v5/distributed.py --suite full --repeats 8 \
    --output-dir "experiments/cuda_jet_v5/artifacts/replay-01/scaling_w${world}"
done

for world in 1 2 4; do
  $FLASHNS_PYTHON -m torch.distributed.run --standalone --nproc_per_node="$world" \
    experiments/cuda_jet_v5/recheck_small.py --repeats 20 \
    --output-dir "experiments/cuda_jet_v5/artifacts/replay-01/controlled_weak_w${world}"
done

$FLASHNS_PYTHON experiments/cuda_jet_v5/independent_throughput.py --repeats 9 \
  --output experiments/cuda_jet_v5/artifacts/replay-01/independent_throughput.json
$FLASHNS_PYTHON experiments/cuda_jet_v5/profile_costs.py \
  --output experiments/cuda_jet_v5/artifacts/replay-01/profile_costs.json
```

数值检查可以每卡独立运行；正式配对计时按上述顺序进行，避免其它 GPU 测试干扰。计时前提是相同构建的数值检查已经通过。`check_sanitizers.py` 同时尝试硬件计数器；此主机返回 `ERR_NVGPUCTRPERM`，该限制单独记录，不影响四种 Sanitizer 的通过状态。

完整梯度计时不等于科学收敛时间。DP 表明确计入同步/归约和可选 SGD，排除初始化、warmup、起始 barrier 与独立参考。原始 JSON 保存了各阶段和所有重复样本，不能只看最快的一次。

## 已保存的工件

`artifacts/` 中保留正式 `.so`、构建日志、四卡数值报告、Sanitizer 日志、局部计时、1/2/4 卡逐 rank 报告、弱扩展复测、独立任务吞吐和静态 SASS。二进制只用于本轮重放，不应跨 GPU 架构沿用结果。

`measured_sources_first.tar.gz` 保存早期测量的源码字节；后续驱动格式整理不改变 native 库，旧哈希仍可从该归档解析。`distributed_smoke_source.py`、`preflight_inventory_source.py` 保留较早 smoke 的源码。最终源码归档见 `measured_sources_final.tar.gz`。

审计本轮固定工件：

```bash
python experiments/cuda_jet_v5/audit_artifacts.py
```

该命令适用于附带的原始报告和库，只检查完整性及哈希，不代替重新运行 CUDA。
