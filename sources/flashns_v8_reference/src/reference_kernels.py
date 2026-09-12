"""CPU reference for FlashNS v8 mathematical contracts, NOT a CUDA implementation.

The implementation intentionally prioritizes transparent indices over performance.
All quadrature/loss weights passed to these routines are already globally normalized.
Only a first parameter VJP is exposed. No optimizer state is updated here.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
from math import factorial
from typing import Sequence
import numpy as np


@lru_cache(None)
def indices(dim: int, order: int = 3) -> tuple[tuple[int, ...], ...]:
    if dim < 1 or order < 0:
        raise ValueError("dim must be positive and order nonnegative")
    return tuple(sorted((a for a in product(range(order+1), repeat=dim)
                         if sum(a) <= order), key=lambda a: (sum(a), a)))


@lru_cache(None)
def pairs(dim: int, order: int = 3):
    ix = indices(dim, order); lookup = {a:i for i,a in enumerate(ix)}
    return tuple(tuple((j, lookup[tuple(x-y for x,y in zip(a,b))])
                       for j,b in enumerate(ix) if all(y <= x for x,y in zip(a,b)))
                 for a in ix)


def multiply(a: np.ndarray, b: np.ndarray, dim: int) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 3 or a.shape[1] != len(indices(dim)):
        raise ValueError("expected equal [batch,Q,channels] full cubic jets")
    out = np.zeros_like(a)
    for k, terms in enumerate(pairs(dim)):
        for i,j in terms:
            out[:,k] += a[:,i] * b[:,j]
    return out


def stable_a1(z: np.ndarray) -> np.ndarray:
    r = np.exp(-np.abs(z))
    v = (2*r)/(1+r*r)
    return v*v


def tanh_forward(z: np.ndarray, dim: int) -> tuple[np.ndarray,np.ndarray]:
    t = np.tanh(z[:,0]); aux = stable_a1(z[:,0])
    d = z.copy(); d[:,0] = 0
    d2 = multiply(d,d,dim); d3 = multiply(d2,d,dim)
    h = aux[:,None]*d - (t*aux)[:,None]*d2 + (aux*(t*t-1/3))[:,None]*d3
    h[:,0] = t
    return h, aux


def derivative_jet(h: np.ndarray, aux: np.ndarray, dim: int) -> np.ndarray:
    g = -multiply(h,h,dim); g[:,0] = aux
    return g


def tanh_vjp(h: np.ndarray, aux: np.ndarray, dh: np.ndarray, dim: int) -> np.ndarray:
    g = derivative_jet(h,aux,dim); out = np.zeros_like(h)
    # Transpose of multiplication by g in the factorial-normalized basis.
    for alpha, terms in enumerate(pairs(dim)):
        for beta,gamma in terms:
            out[:,beta] += dh[:,alpha] * g[:,gamma]
    return out


def coordinate_jets(x: np.ndarray, dim: int) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError("coordinate dimensionality mismatch")
    ix = indices(dim); lookup = {a:i for i,a in enumerate(ix)}
    out = np.zeros((len(x),len(ix),dim), dtype=np.float64)
    out[:,0] = x
    for j in range(dim):
        e = tuple(int(k == j) for k in range(dim))
        out[:,lookup[e],j] = 1
    return out


# A residual is a sparse polynomial of flattened normalized jet entries.
# Term representation (coefficient, i, j); j == -1 denotes a linear term.
@lru_cache(None)
def ns_terms(nu: float = 0.07):
    ix = indices(2); look = {a:i for i,a in enumerate(ix)}
    def slot(field,a): return look[a]*3+field
    def deriv(field,a,axis,count=1):
        t=list(a); t[axis]+=count
        return factorial(t[axis])/factorial(a[axis]), slot(field,tuple(t))
    outputs=[]
    for alpha in ((0,0),(1,0),(0,1)):
        for field in (0,1):
            terms=[]
            for beta in ix:
                if not all(b <= a for a,b in zip(alpha,beta)): continue
                gamma=tuple(a-b for a,b in zip(alpha,beta))
                for advfield,axis in ((0,0),(1,1)):
                    coeff,j=deriv(field,gamma,axis)
                    terms.append((coeff,slot(advfield,beta),j))
            coeff,i=deriv(2,alpha,field)
            terms.append((coeff,i,-1))
            for axis in (0,1):
                coeff,i=deriv(field,alpha,axis,2)
                terms.append((-nu*coeff,i,-1))
            outputs.append(tuple(terms))
        outputs.append(tuple((coeff,i,-1) for coeff,i in
                             (deriv(0,alpha,0),deriv(1,alpha,1))))
    return tuple(outputs)


def ns_residuals(jets: np.ndarray, nu: float=0.07) -> np.ndarray:
    if jets.shape[1:] != (10,3): raise ValueError("2D cubic u,v,p jets required")
    z=jets.reshape(len(jets),30); r=np.zeros((len(jets),9),dtype=np.float64)
    for k,terms in enumerate(ns_terms(nu)):
        for c,i,j in terms:
            r[:,k] += c*z[:,i] if j < 0 else c*z[:,i]*z[:,j]
    return r


def residual_loss_seed(jets: np.ndarray, weights: np.ndarray,
                       nu: float=.07, grad_weight: float=.2):
    if weights.shape != (len(jets),) or not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError("finite nonnegative globally normalized point weights required")
    r=ns_residuals(jets,nu)
    metric=np.array([1.,1.,1.]+[grad_weight]*6)
    dr=weights[:,None]*r*metric
    loss=.5*np.sum(weights[:,None]*r*r*metric)
    z=jets.reshape(len(jets),30); dz=np.zeros_like(z)
    for k,terms in enumerate(ns_terms(nu)):
        for c,i,j in terms:
            if j < 0: dz[:,i] += c*dr[:,k]
            else:
                dz[:,i] += c*dr[:,k]*z[:,j]
                dz[:,j] += c*dr[:,k]*z[:,i]
    return float(loss),dz.reshape(jets.shape)


def pointwise_mlp(x,weights,biases):
    y=x
    for l,(w,b) in enumerate(zip(weights,biases)):
        y=y@w.T+b
        if l+1<len(weights): y=np.tanh(y)
    return y


def packed_step(x: np.ndarray, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray],
                ni: int, pde_weights: np.ndarray, bc_weights: np.ndarray, targets: np.ndarray,
                compact: bool=True):
    """One GEMM per affine/dgrad/wgrad in the packed reference.

    The first ni points demand Q=10; the remaining points demand values only.
    Dense mode propagates Q=10 at every point, with the identical objective.
    Both modes use stable auxiliary derivatives for all nonlinear layers.
    """
    b=len(x); q=10
    if not 0<=ni<=b or x.shape!=(b,2) or pde_weights.shape!=(ni,):
        raise ValueError("bad interior split / coordinates / weights")
    if bc_weights.shape!=(b-ni,3) or targets.shape!=(b-ni,3):
        raise ValueError("bad boundary arrays")
    if not np.isfinite(bc_weights).all() or (bc_weights<0).any():
        raise ValueError("bad boundary weights")
    orders=np.array([3]*ni+([0]*(b-ni) if compact else [3]*(b-ni)))
    counts=np.where(orders==3,q,1)
    offsets=np.concatenate(([0],np.cumsum(counts))).astype(int)
    zero_rows=offsets[:-1]
    h=np.zeros((int(offsets[-1]),2),dtype=np.float64)
    full=coordinate_jets(x,2)
    for k in range(b):
        h[offsets[k]:offsets[k+1]]=full[k] if orders[k]==3 else x[k:k+1]
    checkpoints=[h]; saved_aux=[]
    for l,(w,bias) in enumerate(zip(weights,biases)):
        z=h@w.T; z[zero_rows]+=bias
        if l+1<len(weights):
            h=np.empty_like(z); aux=np.empty((b,z.shape[1]),dtype=np.float64)
            for order in (0,3):
                pts=np.flatnonzero(orders==order)
                if not len(pts): continue
                rows=offsets[pts,None]+np.arange(1 if order==0 else q)[None,:]
                block=z[rows]
                if order==3: hh,aa=tanh_forward(block,2)
                else: hh=np.tanh(block); aa=stable_a1(block[:,0])
                h[rows]=hh; aux[pts]=aa
            saved_aux.append(aux)
        else:
            h=z
        checkpoints.append(h)
    ji=h[:ni*q].reshape(ni,q,3)
    lp,ds=residual_loss_seed(ji,pde_weights)
    bv=h[zero_rows[ni:]]; err=bv-targets
    loss=lp+5*np.sum(bc_weights*err*err)
    d=np.zeros_like(h); d[:ni*q]=ds.reshape(ni*q,3)
    d[zero_rows[ni:]]+=10*bc_weights*err
    dws=[None]*len(weights); dbs=[None]*len(weights)
    for l in reversed(range(len(weights))):
        dws[l]=d.T@checkpoints[l]; dbs[l]=d[zero_rows].sum(0)
        if l:
            dh=d@weights[l]; d=np.empty_like(dh)
            for order in (0,3):
                pts=np.flatnonzero(orders==order)
                if not len(pts): continue
                rows=offsets[pts,None]+np.arange(1 if order==0 else q)[None,:]
                if order==3:
                    d[rows]=tanh_vjp(checkpoints[l][rows],saved_aux[l-1][pts],dh[rows],2)
                else:
                    d[rows]=dh[rows]*saved_aux[l-1][pts,None,:]
    return float(loss),dws+dbs,{"interior_jets":ji,"boundary_values":bv,
                              "packed_rows":int(offsets[-1])}


def projected_amplitude(n,nprime,kmat,delta,t,force):
    """Complex local amplitude rule from the paper; not a complete NS solver."""
    n=np.asarray(n,float); nprime=np.asarray(nprime,float)
    norm=float(n@n)
    if norm<=0 or not np.isfinite(norm): raise ValueError("nonzero finite n required")
    c=(n@(kmat@t+force)-nprime@t)/norm
    dt=-kmat@t-delta*t-force+n*c
    return dt,c
