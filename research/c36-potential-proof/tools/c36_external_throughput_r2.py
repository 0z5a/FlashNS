"""Revision 2 of tools/c36_external_throughput.py (post hoc): keep captured objects alive.

The registered driver's Builder returned only the CUDA Graph replay closure. The engine (or traced
module) that owns the prepared geometry and data the graph reads was then garbage collected, its
memory went back to the allocator, and any later allocation could overwrite what the next replay
reads. In the registered run the analytic graph failed its changed-parameter check this way: the
dense parameter-AD reference evaluated between two replays reused that memory. A development
diagnostic reproduced it (an evaluation between replays turns the replay into NaN) and showed that
the graph is correct while its owner is alive.

This revision changes only Builder.build: every returned callable holds a reference to the objects
it reads. Everything else, including the protocol, selection, timing and ratios, is the registered
driver's code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c36_external_throughput as registered  # noqa: E402


class KeepAlive:
    """A callable that keeps the objects a CUDA Graph or traced module reads alive."""

    def __init__(self, call, *owners):
        self.call, self.owners = call, owners

    def __call__(self):
        return self.call()


class Builder(registered.Builder):
    def build(self, method, variant, theta, chunk):
        import torch
        from box376.data import views
        from box376.external_c36 import PreparedPotentialExternalObjective, build_compiled_external
        from box376.potential_objective import PreparedPotentialNormalizedObjective
        from box376.potential_run import capture
        data, calibration, topology = self.data, self.calibration, self.topology
        if method == 'analytic':
            from torch_c36_packed import activate_packed
            from torch_c36_packed_vjp import activate_packed_vjp
            params = views(theta, topology)
            if variant == 'eager':
                engine = PreparedPotentialNormalizedObjective(data, calibration, activation=activate_packed, activation_vjp=activate_packed_vjp)
                return KeepAlive(lambda: engine(params, chunk), engine, params)
            forward = torch.compile(activate_packed, fullgraph=True, dynamic=False)
            transpose = torch.compile(activate_packed_vjp, fullgraph=True, dynamic=False)
            engine = PreparedPotentialNormalizedObjective(data, calibration, activation=forward, activation_vjp=transpose)
            call = lambda: engine(params, chunk)
            for _ in range(2):
                call()
            if variant == 'compiled':
                return KeepAlive(call, engine, params, forward, transpose)
            return KeepAlive(capture(call)[0], engine, params, forward, transpose, call)
        if variant == 'eager':
            objective = PreparedPotentialExternalObjective(data, calibration, method)
            params = views(theta, topology)
            return KeepAlive(lambda: objective(params, chunk), objective, params)
        traced = build_compiled_external(data, calibration, method, theta, topology, chunk, compile=False)
        if variant == 'traced':
            return KeepAlive(traced, traced)
        for _ in range(2):
            traced()
        return KeepAlive(capture(traced)[0], traced)


if __name__ == '__main__':
    registered.Builder = Builder
    registered.main()
