"""Development diagnostic (not a result): why the analytic CUDA Graph failed the changed-parameter check
in the external throughput driver while the same graph passes it in the trainers.

Cases, each in a fresh build: the throughput driver's Builder; the trainers' construction
(nn.Parameter, views of theta.detach()); the Builder after first building and running the compiled
variant, as in the driver's order. For each: error at the initialization, error after an in-place
parameter change (against the compiled engine evaluated at the changed parameters), and whether the
replay after the change still returns the values of the initialization (stale reads).
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from box376.data import flatten, initialize, views
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.potential_run import capture, load_problem


def ratio(actual, reference, protocol):
    worst = 0.0
    for a, b in zip(actual, reference):
        a, b = a.detach().double().cpu().numpy(), b.detach().double().cpu().numpy()
        worst = max(worst, float(np.max(np.abs(a - b) / (protocol['atol'] + protocol['rtol'] * np.abs(b)))))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from c36_external_throughput import Builder
    from torch_c36_packed import activate_packed
    from torch_c36_packed_vjp import activate_packed_vjp
    torch.set_default_dtype(torch.float64)
    protocol = json.loads(args.protocol.read_text())
    data, calibration = load_problem(args, protocol)
    topology = tuple(protocol['topology'])
    chunk = protocol['explicit_chunk']
    initial = flatten(initialize(topology, 642201)).cuda()
    changed = initial + 1e-3
    forward = torch.compile(activate_packed, fullgraph=True, dynamic=False)
    transpose = torch.compile(activate_packed_vjp, fullgraph=True, dynamic=False)
    plain = PreparedPotentialNormalizedObjective(data, calibration, activation=forward, activation_vjp=transpose)
    at_initial = tuple(v.clone() for v in plain(views(initial, topology), chunk))
    at_changed = tuple(v.clone() for v in plain(views(changed, topology), chunk))
    out = {'kind': 'development diagnostic, not a result', 'cases': {}}

    def check(name, call, theta):
        row = out['cases'][name] = {}
        try:
            row['initial_error'] = ratio(call(), at_initial, protocol)
            with torch.no_grad():
                theta.copy_(changed)
            value = tuple(v.clone() for v in call())
            torch.cuda.synchronize()
            row['changed_error'] = ratio(value, at_changed, protocol)
            row['changed_vs_initial_values'] = ratio(value, at_initial, protocol)
            row['loss_after_change'] = float(value[0]); row['loss_expected'] = float(at_changed[0]); row['loss_initial'] = float(at_initial[0])
        except BaseException:
            row['failure'] = traceback.format_exc()[-2000:]
        args.output.write_text(json.dumps(out, indent=1) + '\n')

    builder = Builder(data, calibration, topology, protocol)
    theta = initial.clone()
    check('builder_graph_only', builder.build('analytic', 'graph', theta, chunk), theta)

    theta = torch.nn.Parameter(initial.clone())
    params = views(theta.detach(), topology)
    engine = PreparedPotentialNormalizedObjective(data, calibration, activation=forward, activation_vjp=transpose)
    for _ in range(2):
        engine(params, chunk)
    check('trainer_construction', capture(lambda: engine(params, chunk))[0], theta)

    theta_c = initial.clone()
    compiled = builder.build('analytic', 'compiled', theta_c, chunk)
    compiled(); torch.cuda.synchronize()
    theta = initial.clone()
    check('builder_graph_after_compiled', builder.build('analytic', 'graph', theta, chunk), theta)

    theta = initial.clone()
    call = builder.build('analytic', 'graph', theta, chunk)
    value = call()
    _ = tuple(v.clone() for v in plain(views(changed, topology), chunk))  # an eager evaluation between replays, as in the driver
    check('builder_graph_eager_in_between', call, theta)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
