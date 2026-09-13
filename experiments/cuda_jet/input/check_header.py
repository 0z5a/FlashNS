"""Compile the host view of the generated CUDA header and compare its formulas."""
from pathlib import Path
import ctypes
import json
import subprocess
import tempfile
import torch
from jet_reference import JetPlan,tanh_jet,tanh_jet_vjp

def main():
    torch.manual_seed(1209)
    root=Path(__file__).resolve().parent
    wrappers='#include "jet_primitives.cuh"\n'
    for d in (2,3):
        wrappers+=f'extern "C" void f{d}(double*z,double*h){{flashns::tanh_fwd_{d}d3(z,h);}}\n'
        wrappers+=f'extern "C" void b{d}(double*h,double*b,double*z){{flashns::tanh_vjp_{d}d3(h,b,z);}}\n'
    errors={}
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp); (path/'check.cpp').write_text(wrappers)
        subprocess.run(['g++','-std=c++17','-O3','-ffp-contract=off','-fPIC','-shared',
                        '-I',str(root),str(path/'check.cpp'),'-o',str(path/'check.so')],check=True)
        lib=ctypes.CDLL(str(path/'check.so'))
        ptr=ctypes.POINTER(ctypes.c_double)
        for dim in (2,3):
            f,b=getattr(lib,f'f{dim}'),getattr(lib,f'b{dim}')
            f.argtypes=[ptr,ptr]; b.argtypes=[ptr,ptr,ptr]; f.restype=b.restype=None
            plan=JetPlan.dense(dim)
            ef=eb=0.0
            for _ in range(100):
                z=torch.randn(1,plan.q,1,dtype=torch.float64)*0.4
                bh=torch.randn_like(z)
                hv=tanh_jet(z,plan)
                bv=tanh_jet_vjp(hv,bh,plan)
                za=(ctypes.c_double*plan.q)(*z.flatten().tolist())
                b_a=(ctypes.c_double*plan.q)(*bh.flatten().tolist())
                ha=(ctypes.c_double*plan.q)(); ba=(ctypes.c_double*plan.q)()
                f(za,ha); b(ha,b_a,ba)
                ht=torch.tensor(list(ha),dtype=torch.float64).view_as(hv)
                bt=torch.tensor(list(ba),dtype=torch.float64).view_as(bv)
                ef=max(ef,float((ht-hv).abs().max()))
                eb=max(eb,float((bt-bv).abs().max()))
                torch.testing.assert_close(ht,hv,rtol=2e-12,atol=2e-12)
                torch.testing.assert_close(bt,bv,rtol=2e-12,atol=2e-12)
            errors[f'{dim}d3']={'forward_max_abs':ef,'vjp_max_abs':eb,'cases':100}
    errors['scope']='Host g++ only; nvcc absent. No CUDA compilation, SASS inspection, or GPU timing.'
    print(json.dumps(errors,indent=2))
    (root/'header_validation.json').write_text(json.dumps(errors,indent=2)+'\n')
if __name__=='__main__': main()
