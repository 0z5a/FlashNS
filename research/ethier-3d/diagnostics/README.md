# Compiled C17 capture diagnostic (2026-10-17; not a result)

In the first compiled batch ([protocol-ext2](../src/protocol-ext2-20261017.json)), every compiled C17 case failed during CUDA Graph capture. The warm-up calls succeeded, but the capture call recompiled and reached a generator that `torch.compile` cannot trace.

This one-GPU diagnostic tries four routes, each in a fresh process, on development seed 642201. Each route goes through warm-up, capture, three in-place parameter updates checked against eager C17 (budget 2e-12 + 2e-11|ref|) and 200 timed replays.

| Route | Build | Capture | Replay |
| --- | --- | --- | --- |
| R0 | `make_fx` under an inner `no_grad`, then `torch.compile` (attempt 1) | fails, as in the batch | – |
| R1 | `torch.compile` of a function taking the six parameter tensors as explicit arguments, called under `no_grad` | passes; error ≤ 1.2e-4 of budget | 1.18 ms |
| R2 | `make_fx` without the inner `no_grad` | fails | – |
| R3 | `torch.compile` of a closure over the parameters | passes; error ≤ 1.2e-4 of budget | 1.18 ms |

Eager C17 without a graph took about 9 ms per evaluation. The second attempt uses R1, the 2D GraphEngine pattern ([external2.py](../src/external2.py)). These numbers come from one diagnostic process per route on a shared node. They are not registered results.

In the JSON files, site paths are normalized to `<runtime>/` and `<diagnostic>/`.
