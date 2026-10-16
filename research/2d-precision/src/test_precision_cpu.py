"""CPU check of src/fp32-dtype.patch: FP64 outputs and FP32 error against the FP64 dense Q10 oracle.

    python test_precision_cpu.py SOURCE_TREE OUT.json

Run once on the patched and once on the unpatched tree: FP64 gradient hashes must agree, and the
unpatched tree must reject FP32.
"""
import json, sys
from pathlib import Path
src = Path(sys.argv[1]); out = Path(sys.argv[2])
sys.path[:0] = [str(src / "src"), str(src / "experiments/pinn_solver")]
import torch
torch.set_num_threads(4)
import problem
from flashns.pinn_packed import PackedPINN
from flashns.pinn_contracted_packed import ContractedPackedPINN
task = problem.Problem(interior_count=2048, device="cpu")
ni = task.interior_count
def data(dtype):
    return (task.x.to(dtype), ni, task.pde_weights[:ni].to(dtype), task.boundary_weights[ni:].to(dtype), task.target[ni:].to(dtype))
def run(make, dtype, theta):
    module = make(*data(dtype))
    w, b = problem.views(theta.to(dtype))
    loss, grads = module(w, b)
    return float(loss), torch.cat([g.flatten() for g in grads]).double()
result = {}
for seed in (862101, 862102):
    theta = problem.parameters(seed)
    oracle_loss, oracle_grad = run(lambda *d: PackedPINN(*d, layout="dense"), torch.float64, theta)
    for name, make in (("q10_compact", lambda *d: PackedPINN(*d, layout="compact")),
                       ("gl8_compact_output", lambda *d: ContractedPackedPINN(*d, vjp="output"))):
        for dtype in (torch.float64, torch.float32):
            try:
                loss, grad = run(make, dtype, theta)
                result[f"{seed}|{name}|{str(dtype)[6:]}"] = {
                    "loss": loss, "grad_sha": __import__("hashlib").sha256(grad.numpy().tobytes()).hexdigest()[:16],
                    "loss_rel_err": abs(loss - oracle_loss) / abs(oracle_loss),
                    "grad_rel_l2": float((grad - oracle_grad).norm() / oracle_grad.norm()),
                    "grad_max_rel": float((grad - oracle_grad).abs().max() / oracle_grad.abs().max())}
            except Exception as e:
                result[f"{seed}|{name}|{str(dtype)[6:]}"] = {"error": repr(e)[:300]}
out.write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
