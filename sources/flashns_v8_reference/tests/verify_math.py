#!/usr/bin/env python3
"""Run deterministic CPU correctness checks. Never loads an uploaded GPU binary.

Usage:
  python tests/verify_math.py
  python tests/verify_math.py --handoff-root /path/to/unpacked/handoff
Optional integration reads the original Python reference and compiles an original
header as HOST C++ only; it does not compile CUDA, run Lean, or benchmark GPUs.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, importlib.util, json, math, platform, shutil
import subprocess, sys, tempfile, traceback
from pathlib import Path
import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import reference_kernels as ref
DT=torch.float64
RESULTS=[]


def check(name,fn):
    try:
        detail=fn() or {}
        RESULTS.append({'name':name,'passed':True,**detail})
    except Exception as e:
        RESULTS.append({'name':name,'passed':False,'error':str(e),'traceback':traceback.format_exc()})


def close(a,b,atol=2e-12,rtol=2e-11):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape: raise AssertionError(f'shape mismatch {a.shape} != {b.shape}')
    if not (np.isfinite(a).all() and np.isfinite(b).all()): raise AssertionError('nonfinite comparison')
    ratio=float(np.max(np.abs(a-b)/(atol+rtol*np.abs(b)),initial=0))
    if ratio>1: raise AssertionError(f'error/tolerance={ratio:.4g}')
    return {'max_abs_error':float(np.max(np.abs(a-b),initial=0)), 'max_error_over_tolerance':ratio}


def flatten(gs): return np.concatenate([g.reshape(-1) for g in gs])
def a1_t(z):
    r=torch.exp(torch.where(z>=0,-z,z)); return ((2*r)/(1+r*r))**2

class StableTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z): ctx.save_for_backward(z); return torch.tanh(z)
    @staticmethod
    def backward(ctx,dy):
        z,=ctx.saved_tensors
        return dy*a1_t(z)


def field_t(x,ws,bs):
    h=x
    for l,(w,b) in enumerate(zip(ws,bs)):
        h=h@w.T+b
        if l+1<len(ws): h=StableTanh.apply(h)
    return h


def nested_step(x,ws,bs,ni,pw,bw,target):
    ps=[torch.tensor(p,dtype=DT,requires_grad=True) for p in list(ws)+list(bs)]
    tw,tb=ps[:len(ws)],ps[len(ws):]
    loss=sum(p.sum()*0 for p in ps)
    if ni:
        xx=torch.tensor(x[:ni],dtype=DT,requires_grad=True)
        u,v,p=field_t(xx,tw,tb).unbind(-1)
        def grad(y):return torch.autograd.grad(y.sum(),xx,create_graph=True,retain_graph=True)[0]
        gu,gv,gp=grad(u),grad(v),grad(p)
        ru=u*gu[:,0]+v*gu[:,1]+gp[:,0]-.07*(grad(gu[:,0])[:,0]+grad(gu[:,1])[:,1])
        rv=u*gv[:,0]+v*gv[:,1]+gp[:,1]-.07*(grad(gv[:,0])[:,0]+grad(gv[:,1])[:,1])
        div=gu[:,0]+gv[:,1]
        per=.5*(ru**2+rv**2+div**2+.2*(grad(ru).square().sum(-1)+grad(rv).square().sum(-1)+grad(div).square().sum(-1)))
        loss=loss+(torch.tensor(pw,dtype=DT)*per).sum()
    if len(x)>ni:
        out=field_t(torch.tensor(x[ni:],dtype=DT),tw,tb)
        loss=loss+5*(torch.tensor(bw,dtype=DT)*(out-torch.tensor(target,dtype=DT))**2).sum()
    grads=torch.autograd.grad(loss,ps)
    return float(loss.detach()),[p.detach().numpy() for p in grads]


def make_case(ni,nb,width=5,seed=123,scale=.4):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(ni+nb,2))*scale
    widths=(2,width,width-1,3)
    ws=[rng.normal(size=(o,i))*.7/math.sqrt(i) for i,o in zip(widths,widths[1:])]
    bs=[rng.normal(size=o)*.1 for o in widths[1:]]
    # Deliberately unequal and partially zero weights, already normalized globally.
    pw=rng.uniform(.2,1.2,size=ni)/max(ni,1)
    if ni>=2:pw[0]=0
    bw=rng.uniform(.2,1.2,size=(nb,3))/max(nb,1)
    if nb: bw[:-1,2]=0
    target=rng.normal(size=(nb,3))*.3
    return x,ws,bs,ni,pw,bw,target


def test_indices():
    counts={d:len(ref.indices(d)) for d in (1,2,3,4)}
    paircounts={d:sum(map(len,ref.pairs(d))) for d in (1,2,3,4)}
    for d in counts:
        assert counts[d]==math.comb(d+3,3)
        assert paircounts[d]==math.comb(2*d+3,3)
    assert ref.indices(2)[8]==(2,1)
    assert math.prod(math.factorial(k) for k in (2,1))==2
    # Polynomial f=7*x^2*y + 11*x*y^2 + 13*x^3, at the origin.
    # D_xxy=14, D_xyy=22, D_xxx=78; normalized coefficients recover 7,11,13.
    raw=np.array([14.,22.,78.]); fac=np.array([2.,2.,6.])
    close(raw/fac,[7.,11.,13.],0,1e-15)
    seed=np.array([.3,-.2,.9]); perturb=np.array([1.1,.7,-.4])
    close(np.dot(seed,fac*perturb),np.dot(fac*seed,perturb))
    return {'Q':counts,'multiplication_pair_counts':paircounts}


def test_duality(dim):
    rng=np.random.default_rng(200+dim)
    z=rng.normal(size=(3,len(ref.indices(dim)),4))*.3
    h,aux=ref.tanh_forward(z,dim)
    dz=rng.normal(size=z.shape); dh=rng.normal(size=z.shape)
    jvp=ref.multiply(ref.derivative_jet(h,aux,dim),dz,dim)
    vjp=ref.tanh_vjp(h,aux,dh,dim)
    return close(np.sum(jvp*dh),np.sum(dz*vjp))


def test_fourth_derivative():
    # H_3=phi'''(z0)/6 along z=z0+epsilon; its z0 VJP needs phi''''.
    mp.mp.dps=100
    errs=[]
    for z0 in (0.,.3,-.7,15.,20.):
        z=np.zeros((1,4,1)); z[0,0,0]=z0; z[0,1,0]=1
        h,aux=ref.tanh_forward(z,1); seed=np.zeros_like(z);seed[0,3,0]=1
        actual=ref.tanh_vjp(h,aux,seed,1)[0,0,0]
        t=mp.tanh(mp.mpf(z0)); a=1-t*t
        expected=float((16*t-24*t**3)*a/6)
        errs.append(close(actual,expected,1e-30,2e-12)['max_error_over_tolerance'])
    return {'max_error_over_tolerance':max(errs)}



def test_full_jet_network(dim):
    rng=np.random.default_rng(313+dim)
    x=rng.normal(size=(2,dim))*.3
    widths=(dim,5,4,3)
    ws=[rng.normal(size=(o,i))*.4 for i,o in zip(widths,widths[1:])]
    bs=[rng.normal(size=o)*.1 for o in widths[1:]]
    h=ref.coordinate_jets(x,dim); checkpoints=[h];aux=[]
    for l,(w,b) in enumerate(zip(ws,bs)):
        z=h@w.T;z[:,0]+=b
        if l+1<len(ws):h,a=ref.tanh_forward(z,dim);aux.append(a)
        else:h=z
        checkpoints.append(h)
    seed=rng.normal(size=h.shape)
    d=seed.copy();dws=[None]*len(ws);dbs=[None]*len(ws)
    for l in reversed(range(len(ws))):
        dws[l]=d.reshape(-1,d.shape[-1]).T@checkpoints[l].reshape(-1,checkpoints[l].shape[-1])
        dbs[l]=d[:,0].sum(0)
        if l:d=ref.tanh_vjp(checkpoints[l],aux[l-1],d@ws[l],dim)
    ps=[torch.tensor(p,dtype=DT,requires_grad=True) for p in ws+bs]
    xx=torch.tensor(x,dtype=DT,requires_grad=True)
    out=field_t(xx,ps[:len(ws)],ps[len(ws):])
    derivative_cache={(0,)*dim:out}
    derivative_values=[]
    for alpha in ref.indices(dim):
        cols=[]
        for c in range(3):
            val=out[:,c]
            for axis,count in enumerate(alpha):
                for _ in range(count):
                    val=torch.autograd.grad(val.sum(),xx,create_graph=True,retain_graph=True)[0][:,axis]
            cols.append(val/math.prod(math.factorial(k) for k in alpha))
        derivative_values.append(torch.stack(cols,-1))
    jt=torch.stack(derivative_values,1)
    valcheck=close(h,jt.detach().numpy())
    loss=(jt*torch.tensor(seed,dtype=DT)).sum()
    grad=torch.autograd.grad(loss,ps)
    gradcheck=close(flatten(dws+dbs),flatten([g.detach().numpy() for g in grad]))
    return {'dim':dim,'all_output_coefficients':h.size,'arbitrary_all_order_seed':True,
            'value_check':valcheck,'parameter_vjp_check':gradcheck}


def test_state_and_tails():
    mp.mp.dps=420
    probes=[0.,15.,20.,21.,40.,100.,300.,355.,360.,370.,372.,373.,374.,380.]
    rows=[]
    tiny=mp.mpf(float(np.nextafter(0.,1.)))
    for z in probes:
        for sign in (1.,-1.):
            zz=sign*z
            r=mp.exp(-abs(mp.mpf(zz))); expected=(2*r/(1+r*r))**2
            actual=float(ref.stable_a1(np.array(zz)))
            absolute=abs(mp.mpf(actual)-expected)
            normal=expected>=mp.mpf(np.finfo(np.float64).tiny)
            if normal:
                rel=float(absolute/expected)
                assert rel<5e-13,(zz,rel)
            else:
                # Absolute quantum test, NOT a promised relative tolerance in subnormal range.
                assert absolute<=2*tiny,(zz,str(absolute))
                rel=None
            rows.append({'z':zz,'fp64_a1':actual,'mp_a1':mp.nstr(expected,25),
                         'class':'normal' if normal else ('subnormal' if actual else 'rounded_zero'),
                         'relative_error_when_normal':rel})
    assert np.tanh(20.)==np.tanh(21.)==1.
    assert ref.stable_a1(np.array(20.))!=ref.stable_a1(np.array(21.))
    return {'h_only_collision_confirmed':True,'tail_probes':rows,
            'scope':'scalar a1 on this CPU; not a global nonlinear error bound'}


def test_network(ni,nb,scale):
    args=make_case(ni,nb,seed=801+ni+nb,scale=scale)
    dense=ref.packed_step(*args,compact=False)
    compact=ref.packed_step(*args,compact=True)
    nested=nested_step(*args)
    ld=close(dense[0],nested[0]); gd=close(flatten(dense[1]),flatten(nested[1]))
    lc=close(compact[0],nested[0]);gc=close(flatten(compact[1]),flatten(nested[1]))
    close(dense[2]['interior_jets'],compact[2]['interior_jets'])
    close(dense[2]['boundary_values'],compact[2]['boundary_values'])
    x,ws,bs,ni,pw,bw,target=args
    part_i=ref.packed_step(x[:ni],ws,bs,ni,pw,np.empty((0,3)),np.empty((0,3)),compact=True)
    part_b=ref.packed_step(x[ni:],ws,bs,0,np.empty(0),bw,target,compact=True)
    close(part_i[0]+part_b[0],dense[0])
    close(flatten(part_i[1])+flatten(part_b[1]),flatten(dense[1]))
    return {'ni':ni,'nb':nb,'coordinate_scale':scale,'dense_rows':dense[2]['packed_rows'],
            'compact_rows':compact[2]['packed_rows'],'dense_grad_check':gd,'compact_grad_check':gc}


def test_updates():
    args=make_case(5,3,seed=909)
    x,ws,bs,ni,pw,bw,target=args
    parameters=[p.copy() for p in ws+bs]; other=[p.copy() for p in parameters]
    nw=len(ws);maxratio=0.
    for _ in range(5):
        d=ref.packed_step(x,parameters[:nw],parameters[nw:],ni,pw,bw,target,False)
        c=ref.packed_step(x,other[:nw],other[nw:],ni,pw,bw,target,True)
        maxratio=max(maxratio,close(flatten(d[1]),flatten(c[1]))['max_error_over_tolerance'])
        parameters=[p-.001*g for p,g in zip(parameters,d[1])]
        other=[p-.001*g for p,g in zip(other,c[1])]
    close(flatten(parameters),flatten(other))
    return {'sgd_updates':5,'max_gradient_error_over_tolerance':maxratio,
            'not_a_scientific_convergence_test':True}


def test_invalid_weights():
    z=np.zeros((2,10,3))
    for w in (np.array([-1.,1.]),np.array([np.nan,1.]),np.ones(3)):
        try: ref.residual_loss_seed(z,w)
        except ValueError: pass
        else: raise AssertionError('invalid weights accepted')
    l,s=ref.residual_loss_seed(z,np.zeros(2));assert l==0 and not s.any()
    return {'rejected_weight_cases':3,'zero_weight_case_passed':True}


def test_projected_amplitude():
    rng=np.random.default_rng(871)
    maxconstraint=0.;maxdirect=0.
    for _ in range(20):
        n=rng.normal(size=3);np_=rng.normal(size=3);F,RFR,GR=rng.normal(size=3)
        kmat=np.array([[0,-2*F,0],[2*F+RFR,0,0],[GR,0,0]])
        delta=.4; t=rng.normal(size=3)+1j*rng.normal(size=3)
        f=rng.normal(size=3)+1j*rng.normal(size=3)
        dt,c=ref.projected_amplitude(n,np_,kmat,delta,t,f)
        target=-delta*(n@t)-np_@t
        constraint=n@dt-target
        maxconstraint=max(maxconstraint,float(abs(constraint)))
        # Independent 4x4 saddle solve for dt and pressure multiplier c.
        mat=np.zeros((4,4));mat[:3,:3]=np.eye(3);mat[:3,3]=-n;mat[3,:3]=n
        rhs=np.r_[-kmat@t-delta*t-f,target]
        direct=np.linalg.solve(mat,rhs)
        maxdirect=max(maxdirect,float(np.max(np.abs(direct-np.r_[dt,c]))))
        assert abs(constraint)<2e-12
        assert np.max(np.abs(direct-np.r_[dt,c]))<2e-12
    return {'cases':20,'max_constraint_residual':maxconstraint,'max_vs_saddle_solve_error':maxdirect}


def test_modal_propagator():
    # Time-dependent noncommuting A0; no illicit matrix-commutativity assumption.
    def a0(t):return np.array([[.2*math.sin(t),1+.3*t],[-.6+.1*t,-.1*math.cos(2*t)]])
    def d(t):return .4+.1*t
    t0=.1;t1=.9
    integral=.4*(t1-t0)+.05*(t1*t1-t0*t0)
    def solve(m):
        def rhs(t,y):return ((a0(t)-m*m*d(t)*np.eye(2))@y.reshape(2,2)).reshape(-1)
        sol=solve_ivp(rhs,(t0,t1),np.eye(2).reshape(-1),method='DOP853',rtol=2e-12,atol=2e-14)
        assert sol.success
        return sol.y[:,-1].reshape(2,2)
    base=solve(1);details=[]
    for m in (1,2,3,4):
        expected=math.exp(-(m*m-1)*integral)*base
        details.append({'m':m,**close(solve(m),expected,2e-12,2e-10)})
    comm=a0(.2)@a0(.8)-a0(.8)@a0(.2)
    assert np.linalg.norm(comm)>.01
    return {'modes':details,'noncommuting_A0_verified':True,'not_a_speed_benchmark':True}


def test_forced_counterexample():
    m=2;t=.7
    exact=(1-math.exp(-m*m*t))/(m*m)
    false_rescale=math.exp(-(m*m-1)*t)*(1-math.exp(-t))
    assert abs(exact-false_rescale)>.1
    return {'m':m,'t':t,'correct_forced_solution':exact,'incorrect_whole_solution_rescale':false_rescale,
            'duhamel_integral_required':True}


def integration(handoff):
    root=handoff/'flashns'
    sys.path.insert(0,str(root/'src'))
    path=root/'experiments/cuda_jet_h100/common.py'
    spec=importlib.util.spec_from_file_location('v8_original_common',path)
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    import flashns.jet_stable as old
    def test_explicit_seed():
        rng=np.random.default_rng(441);checks=[]
        for batch in (0,1,7):
            for scale in (.01,.4,3.):
                j=rng.normal(size=(batch,10,3))*scale
                pw=rng.uniform(.1,1.,size=batch)/max(batch,1)
                if batch>1:pw[0]=0
                loss,seed=ref.residual_loss_seed(j,pw)
                jt=torch.tensor(j,dtype=DT,requires_grad=True)
                obj=(torch.tensor(pw,dtype=DT)*common.residual_per_point(jt)).sum()
                ds,=torch.autograd.grad(obj,jt)
                close(loss,float(obj.detach())); checks.append(close(seed,ds.detach().numpy()))
        return {'cases':9,'max_error_over_tolerance':max(x['max_error_over_tolerance'] for x in checks),
                'original_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    check('explicit_residual_seed_vs_original_autograd',test_explicit_seed)
    def test_old_jets():
        rng=np.random.default_rng(558);ratios=[]
        for dim in (2,3):
            for c in (1,7):
                z=rng.normal(size=(3,len(ref.indices(dim)),c))*.4
                h,a=ref.tanh_forward(z,dim);zh=torch.tensor(z,dtype=DT,requires_grad=True)
                ho,ao=old.tanh_jet(zh,dim)
                seed=rng.normal(size=z.shape)
                dz,=torch.autograd.grad((ho*torch.tensor(seed,dtype=DT)).sum(),zh)
                close(h,ho.detach().numpy());close(a,ao.detach().numpy())
                ratios.append(close(ref.tanh_vjp(h,a,seed,dim),dz.detach().numpy())['max_error_over_tolerance'])
        return {'cases':4,'max_error_over_tolerance':max(ratios)}
    check('generic_vjp_vs_original_python_autograd',test_old_jets)
    header=root/'experiments/cuda_jet_h100/stable_jet.cuh'
    def test_host_header():
        compiler=shutil.which('g++')
        if compiler is None:raise RuntimeError('g++ is required for optional host-header validation')
        with tempfile.TemporaryDirectory(prefix='flashns-v8-host-') as tmp:
            p=Path(tmp)
            # No historical shared library is loaded: build a fresh HOST-only wrapper.
            cpp='#include "'+str(header)+'"\nextern "C" {\n'
            for dim in (2,3):
                cpp+=f'void f{dim}(const double*z,double*h,double*a){{flashns_stable::tanh_fwd_{dim}d3(z,h,a);}}\n'
                cpp+=f'void v{dim}(const double*h,const double*b,double a,double*z){{flashns_stable::tanh_vjp_{dim}d3(h,b,a,z);}}\n'
            cpp+='}\n';(p/'wrapper.cpp').write_text(cpp)
            subprocess.run([compiler,'-std=c++17','-O2','-fno-fast-math','-shared','-fPIC',str(p/'wrapper.cpp'),'-o',str(p/'host.so')],check=True,capture_output=True)
            lib=ctypes.CDLL(str(p/'host.so')); ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
            rng=np.random.default_rng(781);worst=0.;cases=0
            for dim in (2,3):
                fw=getattr(lib,f'f{dim}');bw=getattr(lib,f'v{dim}')
                fw.argtypes=[ptr,ptr,ptr];bw.argtypes=[ptr,ptr,ctypes.c_double,ptr]
                for z0 in (0.,.5,15.,20.,-20.):
                    q=len(ref.indices(dim));z=rng.normal(size=q)*.3;z[0]=z0
                    h=np.zeros(q);aux=np.zeros(1);fw(z,h,aux)
                    seed=rng.normal(size=q);dz=np.zeros(q);bw(h,seed,float(aux[0]),dz)
                    hr,ar=ref.tanh_forward(z.reshape(1,q,1),dim)
                    dr=ref.tanh_vjp(hr,ar,seed.reshape(1,q,1),dim)
                    worst=max(worst,close(h,hr.reshape(-1))['max_error_over_tolerance'],close(dz,dr.reshape(-1))['max_error_over_tolerance'])
                    cases+=1
            return {'cases':cases,'max_error_over_tolerance':worst,'target':'HOST C++ only, not CUDA',
                    'header_sha256':hashlib.sha256(header.read_bytes()).hexdigest()}
    check('original_unrolled_header_host_only',test_host_header)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff-root',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/cpu_validation.json')
    args=parser.parse_args()
    torch.set_num_threads(1)
    check('normalized_basis_and_cost_counts',test_indices)
    for d in (1,2,3,4):check(f'jet_jvp_vjp_duality_d{d}',lambda d=d:test_duality(d))
    check('third_space_order_requires_fourth_activation_derivative',test_fourth_derivative)
    check('h_only_collision_and_scalar_tail_quantization',test_state_and_tails)
    for dim in (2,3):check(f'all_mixed_output_jets_and_parameter_vjp_d{dim}',lambda dim=dim:test_full_jet_network(dim))
    for ni,nb,scale in ((5,3,.4),(3,2,.01),(4,5,3.),(1,0,.4),(0,3,.4),(0,0,.4)):
        check(f'dense_split_compact_nested_ni{ni}_nb{nb}_scale{scale}',lambda ni=ni,nb=nb,scale=scale:test_network(ni,nb,scale))
    check('five_identical_sgd_updates',test_updates)
    check('weight_validation_and_zero_contributions',test_invalid_weights)
    check('openai_projected_amplitude_vs_saddle_system',test_projected_amplitude)
    check('openai_equation_7_18_propagator_identity',test_modal_propagator)
    check('forced_solution_naive_rescale_rejected',test_forced_counterexample)
    if args.handoff_root:integration(args.handoff_root.resolve())
    report={'schema_version':1,'status':'new CPU checks only',
            'environment':{'python':platform.python_version(),'platform':platform.platform(),
                           'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'mpmath':mp.__version__,
                           'cuda_available':torch.cuda.is_available()},
            'new_gpu_benchmark':False,'new_lean_replay':False,
            'checks_passed':sum(x['passed'] for x in RESULTS),'checks_total':len(RESULTS),
            'all_passed':all(x['passed'] for x in RESULTS),'checks':RESULTS}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('all_passed','checks_passed','checks_total')},indent=2))
    for x in RESULTS:
        if not x['passed']:print(x['name'],x['error'])
    return 0 if report['all_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
