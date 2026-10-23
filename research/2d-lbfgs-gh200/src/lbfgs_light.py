"""torch.optim.LBFGS with only its two-loop recursion changed.

The installed torch's own LBFGS.step source is rewritten: the two-loop block is replaced by a call
to LightLBFGS._two_loop and every other line (memory update, strong-Wolfe line search, stop tests,
state carried across step() calls) is the runtime's code.

torch passes each two-loop coefficient as alpha=, which converts the device tensor to a host number:
two device synchronizations per history entry and iteration. Here the coefficient stays on the device:
- torch.dot as before (the same cuBLAS call on the same vectors);
- one small Triton kernel per entry computes the coefficient (al = dot * ro; c = al - dot * ro) and
  the update q <- q - al * y or r <- r + c * s. form='fma' rounds the update once (fma), form='round2'
  rounds the product and the sum separately; the one that reproduces torch's add_(alpha=) bitwise
  is selected by a test, not assumed.
mode='tensor' runs these kernels eagerly. mode='graph' also keeps a ring of padded, aligned copies of
the history rows once the history is full and replays one captured CUDA Graph per ring position.
"""
import hashlib
import inspect
import textwrap

import torch
import torch.optim.lbfgs as _lbfgs

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only checks of the rewrite need no Triton
    triton = tl = None

TWO_LOOP = (
    "q = flat_grad.neg()\n"
    "for i in range(num_old - 1, -1, -1):\n"
    "    al[i] = old_stps[i].dot(q) * ro[i]\n"
    "    q.add_(old_dirs[i], alpha=-al[i])\n"
    "\n"
    "# multiply by initial Hessian\n"
    "# r/d is the final direction\n"
    "d = r = torch.mul(q, H_diag)\n"
    "for i in range(num_old):\n"
    "    be_i = old_dirs[i].dot(r) * ro[i]\n"
    "    r.add_(old_stps[i], alpha=al[i] - be_i)\n")
REPLACEMENT = "d = r = self._two_loop(flat_grad, old_dirs, old_stps, ro, H_diag, al)\n"
BLOCK = 1024
_ORIGINAL = torch.optim.LBFGS  # the runtime class, kept even if torch.optim.LBFGS is later replaced


def module_source_sha256():
    return hashlib.sha256(inspect.getsource(_lbfgs).encode()).hexdigest()


def rewrite(source=None):
    """Return (original step source, rewritten step source); refuses a source without the expected block."""
    if source is None:
        source = textwrap.dedent(inspect.getsource(_ORIGINAL.step))
    hits = [w for w in range(0, 48, 4) if source.count(textwrap.indent(TWO_LOOP, " " * w)) == 1]
    if len(hits) != 1:
        raise RuntimeError("the installed torch LBFGS two-loop does not match the expected source")
    pad = " " * hits[0]
    rewritten = source.replace(textwrap.indent(TWO_LOOP, pad), textwrap.indent(REPLACEMENT, pad))
    if "alpha=-al[i]" in rewritten or "alpha=al[i] - be_i" in rewritten or rewritten.count("self._two_loop(") != 1:
        raise RuntimeError("rewrite incomplete")
    return source, rewritten


if triton is not None:
    @triton.jit
    def _first(q_ptr, y_ptr, dot_ptr, ro_ptr, al_ptr, n, FMA: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        al = tl.load(dot_ptr) * tl.load(ro_ptr)
        if pid == 0:
            tl.store(al_ptr, al)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        q = tl.load(q_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        if FMA:
            q = tl.fma(-al, y, q)
        else:
            q = q + (-al) * y
        tl.store(q_ptr + offs, q, mask=mask)


    @triton.jit
    def _second(r_ptr, s_ptr, dot_ptr, ro_ptr, al_ptr, n, FMA: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        be = tl.load(dot_ptr) * tl.load(ro_ptr)
        c = tl.load(al_ptr) - be
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        r = tl.load(r_ptr + offs, mask=mask)
        s = tl.load(s_ptr + offs, mask=mask)
        if FMA:
            r = tl.fma(c, s, r)
        else:
            r = r + c * s
        tl.store(r_ptr + offs, r, mask=mask)


def first(q, y, dot, ro, al, fma):
    n = q.numel()
    _first[(triton.cdiv(n, BLOCK),)](q, y, dot, ro, al, n, FMA=fma, BLOCK=BLOCK, enable_fp_fusion=False)


def second(r, s, dot, ro, al, fma):
    n = r.numel()
    _second[(triton.cdiv(n, BLOCK),)](r, s, dot, ro, al, n, FMA=fma, BLOCK=BLOCK, enable_fp_fusion=False)


class LightLBFGS(_ORIGINAL):
    def __init__(self, params, mode="tensor", form="fma", **kwargs):
        super().__init__(params, **kwargs)
        if mode not in ("tensor", "graph") or form not in ("fma", "round2"):
            raise ValueError((mode, form))
        self.mode, self.fma = mode, form == "fma"
        self.original_source, self.rewritten_source = rewrite()
        namespace = dict(vars(_lbfgs))
        exec(compile(self.rewritten_source, "<lbfgs_light>", "exec"), namespace)
        self._rewritten_step = namespace["step"]
        self.graph_replays = self.eager_two_loops = self.ring_refills = 0
        # Coverage only: two-loops that found the history unchanged (torch skipped the update, ys <= 1e-10).
        self.history_unchanged, self._seen_newest = 0, None
        self._ring = None

    def step(self, closure):
        return self._rewritten_step(self, closure)

    def _eager(self, flat_grad, old_dirs, old_stps, ro, H_diag, al):
        q = flat_grad.neg()
        for i in range(len(old_dirs) - 1, -1, -1):
            al[i] = flat_grad.new_empty(())
            first(q, old_dirs[i], old_stps[i].dot(q), ro[i], al[i], self.fma)
        r = torch.mul(q, H_diag)
        for i in range(len(old_dirs)):
            second(r, old_stps[i], old_dirs[i].dot(r), ro[i], al[i], self.fma)
        return r

    def _two_loop(self, flat_grad, old_dirs, old_stps, ro, H_diag, al):
        m = self.param_groups[0]["history_size"]
        if old_dirs and old_dirs[-1] is self._seen_newest:
            self.history_unchanged += 1
        self._seen_newest = old_dirs[-1] if old_dirs else None
        if self.mode == "tensor" or len(old_dirs) < m:
            self.eager_two_loops += 1
            return self._eager(flat_grad, old_dirs, old_stps, ro, H_diag, al)
        n = flat_grad.numel()
        if self._ring is None:
            pad = (n + 63) // 64 * 64  # each row starts on a 512-byte boundary, like a fresh allocation
            self._ring = {"Y": flat_grad.new_zeros(m, pad), "S": flat_grad.new_zeros(m, pad),
                          "ro": flat_grad.new_zeros(m), "al": flat_grad.new_zeros(m),
                          "H": flat_grad.new_zeros(()), "g": flat_grad.new_zeros(n),
                          "head": None, "newest": None, "graphs": {}}
        ring = self._ring
        Y, S, RO = ring["Y"], ring["S"], ring["ro"]
        if ring["head"] is not None and old_dirs[-1] is not ring["newest"]:
            if old_dirs[-2] is ring["newest"]:  # one entry appended, the oldest dropped
                slot = ring["head"]
                Y[slot, :n].copy_(old_dirs[-1]); S[slot, :n].copy_(old_stps[-1]); RO[slot].copy_(ro[-1])
                ring["head"], ring["newest"] = (slot + 1) % m, old_dirs[-1]
            else:
                ring["head"] = None
        if ring["head"] is None:
            for i in range(m):
                Y[i, :n].copy_(old_dirs[i]); S[i, :n].copy_(old_stps[i]); RO[i].copy_(ro[i])
            ring["head"], ring["newest"] = 0, old_dirs[-1]
            self.ring_refills += 1
        ring["H"].copy_(H_diag); ring["g"].copy_(flat_grad)
        head = ring["head"]
        if head not in ring["graphs"]:
            ring["graphs"][head] = self._capture(head, m, n)
        graph, out = ring["graphs"][head]
        graph.replay()
        self.graph_replays += 1
        return out.clone()

    def _capture(self, head, m, n):
        ring = self._ring
        rows = [(head + i) % m for i in range(m)]  # age order: rows[0] holds old_dirs[0]
        Y = [ring["Y"][k, :n] for k in rows]; S = [ring["S"][k, :n] for k in rows]
        RO = [ring["ro"][k] for k in rows]; AL = [ring["al"][k] for k in rows]
        fma = self.fma

        def body():
            q = ring["g"].neg()
            for i in range(m - 1, -1, -1):
                first(q, Y[i], S[i].dot(q), RO[i], AL[i], fma)
            r = torch.mul(q, ring["H"])
            for i in range(m):
                second(r, S[i], Y[i].dot(r), RO[i], AL[i], fma)
            return r
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = body()
        return graph, out
