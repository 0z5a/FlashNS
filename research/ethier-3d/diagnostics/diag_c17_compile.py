"""Diagnostic (not a result): which torch.compile route lets compiled C17 survive CUDA Graph capture.

    python diag_c17_compile.py SRC_DIR ROUTE OUT.json [cpu]

Routes:
  R0  make_fx of explicit_objective under an inner no_grad, then torch.compile (batch ext2 builder)
  R1  torch.compile of a function taking the six parameter tensors as explicit arguments, called under no_grad
  R2  make_fx without the inner no_grad (called under an outer no_grad), then torch.compile
  R3  torch.compile of a zero-argument closure over the parameters, called under no_grad
Each route: build, three warm calls on a side stream, CUDA Graph capture, then three in-place
parameter updates checked against eager explicit_objective (2e-12 + 2e-11|ref|), then 200 timed replays.
On CPU (last argument "cpu") there is no capture: the compiled call itself is checked.
"""
import json, sys, time, traceback
from pathlib import Path
src, route, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
cpu = len(sys.argv) > 4 and sys.argv[4] == 'cpu'
sys.path.insert(0, str(src))
import numpy as np, torch
from objective import explicit_objective
device = torch.device('cpu' if cpu else 'cuda')
torch.backends.cuda.matmul.allow_tf32 = False
protocol = json.loads((src / 'protocol-ext2-20261017.json').read_text())
raw = np.load(src / 'data-v2.npz')
data = {k: torch.as_tensor(raw[k], device=device).clone() for k in raw.files if not k[0].isdigit()}
flat = [torch.nn.Parameter(torch.as_tensor(raw[f'642201_{t}{i}'], device=device).clone()) for i in range(3) for t in ('W', 'b')]
params = [(flat[i], flat[i + 1]) for i in range(0, 6, 2)]
report = {'route': route, 'device': str(device), 'torch': torch.__version__}
try:
    from torch.fx.experimental.proxy_tensor import make_fx
    t0 = time.perf_counter()
    if route == 'R0':
        def run():
            with torch.no_grad():
                return explicit_objective(params, data, protocol)
        fn = torch.compile(make_fx(run)(), fullgraph=True, dynamic=False)
        call = fn
    elif route == 'R1':
        def f(W0, b0, W1, b1, W2, b2):
            return explicit_objective([(W0, b0), (W1, b1), (W2, b2)], data, protocol)
        fc = torch.compile(f, fullgraph=True, dynamic=False)
        def call():
            with torch.no_grad():
                return fc(*flat)
    elif route == 'R2':
        with torch.no_grad():
            gm = make_fx(lambda: explicit_objective(params, data, protocol))()
        fn = torch.compile(gm, fullgraph=True, dynamic=False)
        def call():
            with torch.no_grad():
                return fn()
    elif route == 'R3':
        fc = torch.compile(lambda: explicit_objective(params, data, protocol), fullgraph=True, dynamic=False)
        def call():
            with torch.no_grad():
                return fc()
    report['build_seconds'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if not cpu:
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            loss, grads = call()
        torch.cuda.synchronize()
        def replay():
            graph.replay(); return loss, grads
    else:
        replay = call
    report['warm_capture_seconds'] = time.perf_counter() - t0
    errors = []
    for step in range(3):
        l, g = replay()
        with torch.no_grad():
            rl, rg = explicit_objective(params, data, protocol)
        got = [l] + [t for pair in g for t in pair]; ref = [rl] + [t for pair in rg for t in pair]
        errors.append(max(float(((a - b).abs() / (2e-12 + 2e-11 * b.abs())).max()) for a, b in zip(got, ref)))
        with torch.no_grad():
            for p, d in zip(flat, [t for pair in g for t in pair]):
                p.add_(d, alpha=-1e-4)
    report['max_error_over_budget'] = errors
    report['pass'] = max(errors) <= 1
    for _ in range(20):
        replay()
    if not cpu: torch.cuda.synchronize()
    t0 = time.perf_counter(); reps = 3 if cpu else 200
    for _ in range(reps):
        replay()
    if not cpu: torch.cuda.synchronize()
    report['replay_ms'] = 1000 * (time.perf_counter() - t0) / reps
    if not cpu:
        with torch.no_grad():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(50):
                explicit_objective(params, data, protocol)
            torch.cuda.synchronize(); report['eager_ms'] = 1000 * (time.perf_counter() - t0) / 50
except Exception:
    report['error'] = traceback.format_exc()[-2500:]
    report['pass'] = False
out.write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != 'error'}), (report.get('error') or '')[-300:])
