"""CPU component self-test of the C36 v7 optimizer integration (login-node sized).

    PYTHONPATH=src python3 tools/c36_v7_component_selftest.py --inputs DATA --protocol PROTOCOL --reference REF.npz --output OUT.json

The full trainer exceeds the login node's limits and the full dense inverse Hessian (20.6 GB) does
not belong on it, so the optimizer is driven exactly as box376.potential_train_v7 drives it, with
the same objective/assign callbacks writing into the live parameter tensor, on the real eager
analytic C36 objective of a small subset problem (3 interior points and every value point,
canonical denominators) and on a 200-parameter slice of the stored initial parameters. It checks
that the parameters equal the optimizer's point after every iteration, that the loss falls, that
the carried inverse-Hessian product matches an explicit one, and that the start-checkpoint hash
logic accepts the protocol's v5 checkpoint bytes and rejects a modified copy.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--start-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    import box376.potential_train_v6 as v6
    from torch_c36_packed import activate_packed
    from torch_c36_packed_vjp import activate_packed_vjp
    from box376.data import views
    from box376.potential_objective import PreparedPotentialNormalizedObjective
    from box376.ssbroyden import SSBroyden
    protocol = json.loads(args.protocol.read_text())
    v6.SUBSET_INTERIOR = v6.SUBSET_INTERIOR[:3]
    data, calibration = v6._cpu_problem(args, protocol)
    subset = v6.subset_problem(data)
    engine = PreparedPotentialNormalizedObjective(subset, calibration, activation=activate_packed, activation_vjp=activate_packed_vjp)
    topology = tuple(protocol['topology'])
    with np.load(args.reference, allow_pickle=False) as raw:
        theta = torch.nn.Parameter(torch.from_numpy(raw['seed361901_theta'].copy()))
    params = views(theta.detach(), topology)
    window = slice(0, 200)

    def evaluate():
        return engine(params, 128)

    def objective(x):
        with torch.no_grad():
            theta[window] = x
        loss, gradient, _ = evaluate()
        return loss.detach().clone(), gradient[window].detach().clone()

    def assign(x):
        with torch.no_grad():
            theta[window] = x

    loss, gradient, _ = evaluate()
    optimizer = SSBroyden(200, 'cpu', mode=protocol['training']['optimizer']['mode'], c1=protocol['training']['optimizer']['c1'],
                          c2=protocol['training']['optimizer']['c2'], max_ls=protocol['training']['optimizer']['max_line_search_evaluations'])
    optimizer.start(theta.detach()[window].clone(), float(loss), gradient[window].clone())
    records, consistent = [], True
    for _ in range(6):
        record = optimizer.step(objective, assign)
        records.append(record)
        consistent &= bool(torch.equal(theta.detach()[window], optimizer.x))
        loss_here = float(evaluate()[0])
        consistent &= abs(loss_here - optimizer.f) <= 1e-12 * abs(optimizer.f)
    report = {'loss_start': float(loss), 'loss_end': optimizer.f, 'iterations': len(records),
              'records': records, 'parameters_equal_optimizer_point_and_loss': consistent,
              'carried_product_relative_difference': optimizer.product_error()}
    if args.start_checkpoint is not None:
        start = protocol['starts']['continue:361901']
        ok_file = hashlib.sha256(args.start_checkpoint.read_bytes()).hexdigest() == start['checkpoint_sha256']
        stored = torch.load(args.start_checkpoint, map_location='cpu', weights_only=True)['theta']
        ok_theta = hashlib.sha256(stored.numpy().tobytes()).hexdigest() == start['theta_sha256']
        modified = stored.clone()
        modified[0] += 1e-12
        rejects = hashlib.sha256(modified.numpy().tobytes()).hexdigest() != start['theta_sha256']
        report['start_checkpoint'] = {'file_hash_matches': ok_file, 'theta_hash_matches': ok_theta, 'modified_copy_rejected': rejects}
    report['passed'] = (consistent and optimizer.f < float(loss) and report['carried_product_relative_difference'] < 1e-8
                        and all(report.get('start_checkpoint', {'x': True}).values()))
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}))
    if not report['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
