"""Independent four-variable truncated Taylor algebra; no C17 propagation.
All 35 normalized monomials of total degree <=3 are retained, including time.
"""
import itertools,math
import numpy as np
MULTI=sorted([a for a in itertools.product(range(4),repeat=4) if sum(a)<=3],key=lambda a:(sum(a),a));INDEX={a:i for i,a in enumerate(MULTI)}
PRODUCT=[(i,j,INDEX[tuple(x+y for x,y in zip(a,b))]) for i,a in enumerate(MULTI) for j,b in enumerate(MULTI) if sum(a)+sum(b)<=3]

def multiply(a,b):
    c=np.zeros_like(a)
    for i,j,k in PRODUCT:c[...,k]+=a[...,i]*b[...,j]
    return c

def activate(a):
    z=a[...,0];y=np.tanh(z);d=1-y*y;d2=-2*y*d;d3=(6*y*y-2)*d
    v=a.copy();v[...,0]=0;v2=multiply(v,v);v3=multiply(v2,v)
    r=d[...,None]*v+d2[...,None]*v2/2+d3[...,None]*v3/6;r[...,0]+=y
    return r

def derivative(a,axes):
    m=tuple(axes.count(i) for i in range(4));return a[...,INDEX[m]]*math.prod(math.factorial(k) for k in m)

def forward(parameters,points):
    dtype=np.result_type(points,*[p.dtype for layer in parameters for p in layer]);a=np.zeros((len(points),4,35),dtype=dtype);a[:,:,0]=points
    for i in range(4):a[:,i,INDEX[tuple(int(j==i) for j in range(4))]]=1
    for k,(W,b) in enumerate(parameters):
        a=np.einsum('nwq,ow->noq',a,W);a[:,:,0]+=b
        if k<len(parameters)-1:a=activate(a)
    return a

def to_c17(a):
    pairs=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
    return np.stack([a[...,0]]+[derivative(a,[i]) for i in range(3)]+[derivative(a,[i,j]) for i,j in pairs]+[sum(derivative(a,[i,j,j]) for j in range(3)) for i in range(3)]+[derivative(a,[3])]+[derivative(a,[i,3]) for i in range(3)],-1)

def loss(parameters,points,force,force_gradient,weights,nu=.1,lambda_g=.2):
    a=forward(parameters,points);v=a[:,:3,0];R=[];RG=[]
    for i in range(3):
        R.append(derivative(a[:,i],[3])+sum(v[:,j]*derivative(a[:,i],[j]) for j in range(3))-nu*sum(derivative(a[:,i],[j,j]) for j in range(3))+derivative(a[:,3],[i])-force[:,i])
        for k in range(3):RG.append(derivative(a[:,i],[k,3])+sum(derivative(a[:,j],[k])*derivative(a[:,i],[j])+v[:,j]*derivative(a[:,i],[j,k]) for j in range(3))-nu*sum(derivative(a[:,i],[k,j,j]) for j in range(3))+derivative(a[:,3],[i,k])-force_gradient[:,i,k])
    R.append(sum(derivative(a[:,i],[i]) for i in range(3)))
    for k in range(3):RG.append(sum(derivative(a[:,i],[i,k]) for i in range(3)))
    return np.sum(weights*(sum(x*x for x in R)+lambda_g*sum(x*x for x in RG)))
