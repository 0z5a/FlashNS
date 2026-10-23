"""Development benchmark (not a result): torch.optim.LBFGS optimizer overhead on GH200 and whether
a two-loop without host synchronizations reproduces torch's parameter trajectory bitwise.

Problem: a 2-64-64-3 tanh network (4,547 FP64 parameters, the 2D topology) fitted by mean squared
error to a smooth target at 2,048 points; the 2D solve's L-BFGS settings (history 50, strong Wolfe,
blocks of at most 20 iterations / 25 evaluations, lr 1, tolerances 1e-12 / 1e-15).

1. rounding forms: torch's add_(y, alpha=a) against the Triton update rounded once (fma) and twice;
2. trajectory: stock, tensor/fma, tensor/round2, graph/fma, graph/round2 from the same start for
   BLOCKS blocks; after every block the parameter SHA-256, iteration and evaluation counts and loss;
3. timing: TIMED further blocks per variant; optimizer overhead per iteration = (block time -
   evaluations x closure time) / iterations, with the closure time measured separately;
4. GPU time of one captured two-loop replay (CUDA events).
"""
import argparse
import hashlib
import json
import platform
import statistics
import time
import traceback
from pathlib import Path

import torch
import triton

import lbfgs_light as light

SETTINGS = dict(lr=1.0, max_iter=20, max_eval=25, tolerance_grad=1e-12, tolerance_change=1e-15,
                history_size=50, line_search_fn="strong_wolfe")


def sha(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


class Problem:
    def __init__(self, seed=20261023, points=2048):
        g = torch.Generator().manual_seed(seed)
        self.shapes = [(64, 2), (64,), (64, 64), (64,), (3, 64), (3,)]
        sizes = [torch.Size(s).numel() for s in self.shapes]
        self.n = sum(sizes)
        scale = [0.7, 0.0, 0.18, 0.0, 0.18, 0.0]
        parts = [torch.randn(s, generator=g, dtype=torch.float64).reshape(-1) * c for s, c in zip(self.shapes, scale)]
        self.x0 = torch.cat(parts).cuda()
        X = torch.rand(points, 2, generator=g, dtype=torch.float64) * 2 - 1
        T = torch.stack((torch.sin(3 * X[:, 0]) * torch.cos(2 * X[:, 1]), torch.exp(-X.pow(2).sum(1)),
                         X[:, 0] * X[:, 1] ** 2), 1)
        self.X, self.T = X.cuda(), T.cuda()
        self.evaluations = 0

    def views(self, flat):
        out, k = [], 0
        for s in self.shapes:
            m = torch.Size(s).numel()
            out.append(flat[k:k + m].view(s)); k += m
        return out

    def loss(self, flat):
        W1, b1, W2, b2, W3, b3 = self.views(flat)
        h = torch.tanh(self.X @ W1.T + b1)
        h = torch.tanh(h @ W2.T + b2)
        return ((h @ W3.T + b3 - self.T) ** 2).mean()

    def closure_for(self, x):
        def closure():
            self.evaluations += 1
            x.grad = None
            loss = self.loss(x)
            loss.backward()
            return loss.detach()
        return closure


def make(variant, x):
    if variant == "stock":
        return torch.optim.LBFGS([x], **SETTINGS)
    mode, form = variant.split("/")
    return light.LightLBFGS([x], mode=mode, form=form, **SETTINGS)


def rounding_forms(n=4547, trials=200):
    g = torch.Generator(device="cuda").manual_seed(7)
    out = {"first": {"fma": 0, "round2": 0}, "second": {"fma": 0, "round2": 0}, "elements": 0}
    for _ in range(trials):
        q = torch.randn(n, device="cuda", dtype=torch.float64, generator=g)
        y = torch.randn(n, device="cuda", dtype=torch.float64, generator=g)
        s = torch.randn(n, device="cuda", dtype=torch.float64, generator=g)
        ro = torch.rand((), device="cuda", dtype=torch.float64, generator=g) + 0.5
        dot1 = s.dot(q)
        al = dot1 * ro
        ref = q.clone().add_(y, alpha=-al)
        for form in ("fma", "round2"):
            got, a = q.clone(), torch.empty((), device="cuda", dtype=torch.float64)
            light.first(got, y, dot1, ro, a, form == "fma")
            out["first"][form] += int((got != ref).sum()) + int(a != al)
        r = ref.clone()
        dot2 = y.dot(r)
        be = dot2 * ro
        ref2 = r.clone().add_(s, alpha=al - be)
        for form in ("fma", "round2"):
            got = r.clone()
            light.second(got, s, dot2, ro, al, form == "fma")
            out["second"][form] += int((got != ref2).sum())
        out["elements"] += n
    return out


def run_blocks(problem, variant, blocks, timed):
    x = problem.x0.clone().requires_grad_(True)
    opt = make(variant, x)
    closure = problem.closure_for(x)
    trace, times = [], []
    for b in range(blocks + timed):
        e0, i0 = problem.evaluations, opt.state[x].get("n_iter", 0) if opt.state else 0
        torch.cuda.synchronize(); t = time.perf_counter()
        loss = opt.step(closure)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        st = opt.state[x]
        row = {"block": b, "sha256": sha(x), "n_iter": st["n_iter"], "func_evals": st["func_evals"],
               "loss": float(loss), "history": len(st["old_dirs"])}
        if b < blocks:
            trace.append(row)
        else:
            times.append((dt, problem.evaluations - e0, st["n_iter"] - i0))
    info = {k: getattr(opt, k) for k in ("graph_replays", "eager_two_loops", "ring_refills") if hasattr(opt, k)}
    return x, opt, trace, times, info


def closure_seconds(problem, repeats=300):
    x = problem.x0.clone().requires_grad_(True)
    c = problem.closure_for(x)
    for _ in range(20):
        float(c())
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(repeats):
        float(c())  # torch.optim.LBFGS converts every closure loss to a host number
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=60)
    p.add_argument("--timed", type=int, default=20)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_default_dtype(torch.float64)
    original, rewritten = light.rewrite()
    out = {"kind": "development benchmark, not a result", "torch": torch.__version__, "triton": triton.__version__,
           "gpu": torch.cuda.get_device_name(), "host": platform.machine(), "lbfgs_module_sha256": light.module_source_sha256(),
           "step_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
           "rewritten_step_sha256": hashlib.sha256(rewritten.encode()).hexdigest(), "settings": SETTINGS}

    def save():
        a.output.write_text(json.dumps(out, indent=1) + "\n")
    try:
        out["rounding_forms"] = rounding_forms()
    except BaseException:
        out["rounding_forms"] = {"failure": traceback.format_exc()[-2000:]}
    save()
    problem = Problem()
    out["parameters"] = problem.n
    out["closure_seconds"] = closure_seconds(problem)
    save()
    reference = None
    out["variants"] = {}
    for variant in ("stock", "tensor/fma", "tensor/round2", "graph/fma", "graph/round2"):
        row = out["variants"][variant] = {}
        try:
            problem.evaluations = 0
            x, opt, trace, times, info = run_blocks(problem, variant, a.blocks, a.timed)
            row.update(info)
            row["final"] = trace[-1]
            if variant == "stock":
                reference = trace
                row["trace"] = trace
            else:
                diverged = next((r["block"] for r, s in zip(trace, reference) if r != s), None)
                row["first_differing_block"] = diverged
                row["bitwise_identical_blocks"] = len(trace) if diverged is None else diverged
            c = out["closure_seconds"]
            per = [(dt - e * c) / i for dt, e, i in times if i]
            row["overhead_ms_per_iteration"] = {"median": 1e3 * statistics.median(per), "min": 1e3 * min(per), "max": 1e3 * max(per)}
            row["block_ms"] = 1e3 * statistics.median(dt for dt, _, _ in times)
            row["evaluations_per_iteration"] = sum(e for _, e, _ in times) / max(1, sum(i for _, _, i in times))
            if variant.startswith("graph") and getattr(opt, "_ring", None) and opt._ring["graphs"]:
                graph, _ = next(iter(opt._ring["graphs"].values()))
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(10):
                    graph.replay()
                start.record()
                for _ in range(200):
                    graph.replay()
                end.record(); torch.cuda.synchronize()
                row["graph_two_loop_gpu_ms"] = start.elapsed_time(end) / 200
                row["graphs_captured"] = len(opt._ring["graphs"])
            del opt, x
        except BaseException:
            row["failure"] = traceback.format_exc()[-3000:]
        save()
    print(json.dumps({k: v for k, v in out.items() if k != "variants"}, indent=1))
    for k, v in out["variants"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk not in ("trace",)})


if __name__ == "__main__":
    main()
