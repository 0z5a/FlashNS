"""Wider transient 3D state: full spatial cubic plus dt and grad dt.
Uses independent normalized Taylor algebra, retaining 24 channels in layers.
"""
import numpy as np
from taylor_oracle import MULTI,PRODUCT,INDEX,to_c17
KEEP=[i for i,a in enumerate(MULTI) if a[3]==0 or (a[3]==1 and sum(a[:3])<=1)]
assert len(KEEP)==24
REINDEX={q:i for i,q in enumerate(KEEP)}
PAIRS=[(REINDEX[i],REINDEX[j],REINDEX[k]) for i,j,k in PRODUCT if i in REINDEX and j in REINDEX and k in REINDEX]

def mul(a,b):
    c=np.zeros_like(a)
    for i,j,k in PAIRS:c[...,k]+=a[...,i]*b[...,j]
    return c

def forward(parameters,points):
    dtype=np.result_type(points,*[a.dtype for layer in parameters for a in layer]);s=np.zeros((len(points),4,24),dtype=dtype);s[:,:,0]=points
    for i in range(4):s[:,i,REINDEX[INDEX[tuple(int(j==i) for j in range(4))]]]=1
    for k,(W,b) in enumerate(parameters):
        s=np.einsum('nwq,ow->noq',s,W);s[:,:,0]+=b
        if k<len(parameters)-1:
            z=s[...,0];y=np.tanh(z);a1=1-y*y;a2=-2*y*a1;a3=(6*y*y-2)*a1
            d=s.copy();d[...,0]=0;d2=mul(d,d);s=a1[...,None]*d+a2[...,None]*d2/2+a3[...,None]*mul(d2,d)/6;s[...,0]+=y
    return s

def contract(s):
    full=np.zeros(s.shape[:-1]+(35,),dtype=s.dtype);full[...,KEEP]=s;return to_c17(full)
