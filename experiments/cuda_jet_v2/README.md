# FP64 dgrad + jet VJP：第一个融合原型

本目录落实 v2 方案的首个 P0–P2 实验，并补充十步固定 SGD 一致性检查。实测结论：正确性和显存节省成立，本次布局没有带来可归因于融合的稳定速度收益。因此它保持实验入口，原有库 GEMM 路径仍是默认实现。详细数据见 [结果报告](../../docs/cuda-v2-results.md)。

## 文件与接口

- `dgrad_jet_vjp.cu`：CUTLASS 主循环、完整 jet 的片上重排、融合/非融合两个消融入口。
- `run_fused.py`：PyTorch tensor/stream 适配、独立 AD 验收、三条执行路径、交错计时、完整梯度与 SGD 对照。
- `sanitizer_smoke.cu`：独立 C++ 数值 oracle、边界与 canary、重复 stream 执行。
- `check_sanitizers.py`：对最终共享库分别执行 memcheck、racecheck、initcheck、synccheck，保存各自日志。硬件 profiler 权限不足时保留错误。
- `../../src/flashns/jet_spec.py`：版本化 full-Taylor JetSpec；目前只注册二维/三维三阶基。

Python 接口是 `FusedDgrad(library)(D, W, Hprev, plan, fused=True)`，输入分别为 `[B,Q,Cout]`、`[Cout,Cin]`、`[B,Q,Cin]`。要求同一 CUDA device、FP64、连续布局。输出新分配，C ABI 拒绝输出与输入内存重叠。`fused=False` 用相同主循环写出 `barH`，可再调用 v1 原生 VJP。

当前 stream 决定执行队列；适配器通过 `record_stream` 保持分配生命周期。跨 stream 的输入就绪关系由调用方用 event/wait 建立，`record_stream` 不负责建立依赖。完整网络反传在 `torch.no_grad()` 内执行。该 API 是显式一阶参数 VJP，禁止把它当成可继续求导的自定义算子；grad enabled 且输入 requires_grad 时会报错。

激活仍采用 v1 的 `H_only_ordinary_fp64` 策略，以固定数学和精度比较融合。饱和区小导数不受支持；高精度失败记录保留在结果 JSON，不能把普通输入通过当作任意输入通过。

## 复现本次 A5000 结果

本次机器使用既有 `/venv/main/bin/python` 的 Torch `2.11.0+cu128`，轻量依赖隔离在项目 `.deps`；项目主 `uv.lock` 的 Torch 版本与该实测环境不同。精确版本见 [环境 JSON](artifacts/environment.json)。环境检查不修改系统驱动或服务。

唯一新增 C++ 后端是 [CUTLASS v4.7.1](https://github.com/NVIDIA/cutlass/releases/tag/v4.7.1)，锁定 commit `cb4247394dd82148787aed73e5dc7cef33cbf862`。缺少该 checkout 时，在项目根目录执行：

```bash
git clone --depth 1 --branch v4.7.1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
git -C third_party/cutlass rev-parse HEAD
```

构建脚本拒绝错误 commit 和修改过的 include 目录。本地仅用于阅读的 header 副本不替代完整 git checkout。CUTLASS 的原许可证保留在其目录中；没有修改其源码。

在已提供的 GPU 主机上，进入 `/root/flashns-day0-20260908`，依次运行：

```bash
PYTHONPATH=src:.deps /venv/main/bin/python scripts/doctor.py --require-gpu --output experiments/cuda_jet_v2/artifacts/environment.json
PYTHONPATH=src:.deps /venv/main/bin/python -m pytest -q
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet/run_cuda.py
PYTHONPATH=src:.deps /venv/main/bin/python experiments/cuda_jet_v2/run_fused.py --benchmark
/venv/main/bin/python experiments/cuda_jet_v2/check_sanitizers.py
```

第三条命令构建原始独立激活库，第四条构建融合库并完成验收与基准。最后对同一份最终融合库做 Sanitizer 检查；重新编译后需要重跑该检查，不能把旧二进制的记录作为新版本验证。`check_sanitizers.py` 的 standalone smoke 当前固定编译为 `sm_86`。

编译为 CUDA 12.8、C++17、`-O3 -arch=sm_86`，不开 fast math。编译命令、耗时和 `.so` 哈希均写入报告。计时先预热，再按轮次旋转三条路径的执行顺序；记录九轮 wall time 与 CUDA event time。先完成不带 profiler 的计时，再收集软件 profiler 数据。

## 验收与范围

`run_fused.py` 以固定 `atol=2e-12, rtol=2e-11` 检查局部 VJP、完整网络梯度和固定十步 SGD；失败返回非零并保存错误。不支持的 dtype/布局和 double backward/HVP 也会明确拒绝。

基准中的三条路径为：库 GEMM + 原生独立 VJP、同一 CUTLASS 主循环 + 原生独立 VJP、融合 CUTLASS。forward、wgrad、loss、输出 seed、精度和 full-jet 表示相同。所有速度结果都针对列出的合成 PINN 用例；十步 SGD 没有科学收敛停止条件。未完成 C13、首尾层特化、自动 tile 搜索、HVP、鲁棒饱和区激活或真实科学训练集成。

`resources.json` 和 `dgrad_sass.txt` 记录静态编译资源与指令。A5000 主机拒绝硬件计数器访问，`ncu.log` 保存 `ERR_NVGPUCTRPERM`；当前没有实际 DRAM 字节或 barrier stall 测量。报告中的逻辑流量节省只按消除的 tensor 字节计算。
