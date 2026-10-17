"""Independent scalar-coordinate AD and wider A24 Torch baseline.
No C17 activation formulas are used in either reference propagation.
"""
import torch
from a24 import KEEP,REINDEX,PAIRS
from taylor_oracle import INDEX

def a24_forward(parameters,points):
    s=points.new_zeros((len(points),4,24));s[:,:,0]=points
    for i in range(4):s[:,i,REINDEX[INDEX[tuple(int(j==i) for j in range(4))]]]=1
    def mul(a,b):
        # Out-of-place coefficient accumulation keeps the autograd path valid.
        return torch.stack([sum((a[...,i]*b[...,j] for i,j,k in PAIRS if k==out),torch.zeros_like(a[...,0])) for out in range(24)],-1)
    for k,(W,b) in enumerate(parameters):
        s=torch.einsum('nwq,ow->noq',s,W);z=s[...,0]+b
        # Affine bias is restricted to the constant polynomial coefficient.
        s=torch.cat((z[...,None],s[...,1:]),-1)
        if k<len(parameters)-1:
            y=torch.tanh(z);a1=1-y*y;a2=-2*y*a1;a3=(6*y*y-2)*a1
            d=torch.cat((torch.zeros_like(z[...,None]),s[...,1:]),-1);d2=mul(d,d);s=a1[...,None]*d+a2[...,None]*d2/2+a3[...,None]*mul(d2,d)/6;s=torch.cat(((s[...,0]+y)[...,None],s[...,1:]),-1)
    return s

def a24_contract(s):
    import math
    def derivative(axes):
        m=tuple(axes.count(i) for i in range(4));return s[...,REINDEX[INDEX[m]]]*math.prod(math.factorial(k) for k in m)
    hp=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
    return torch.stack([s[...,0]]+[derivative([i]) for i in range(3)]+[derivative([i,j]) for i,j in hp]+[sum(derivative([i,j,j]) for j in range(3)) for i in range(3)]+[derivative([3])]+[derivative([i,3]) for i in range(3)],-1)

def scalar_ad_state(parameters,points):
    x=points.detach().clone().requires_grad_(True);value=x
    for k,(W,b) in enumerate(parameters):
        value=value@W.T+b
        if k<len(parameters)-1:value=torch.tanh(value)
    def grad(z):return torch.autograd.grad(z.sum(),x,create_graph=True,retain_graph=True)[0]
    hp=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2));states=[]
    for c in range(4):
        g=grad(value[:,c]);H=torch.stack([grad(g[:,i]) for i in range(4)],1);lap=sum(H[:,i,i] for i in range(3));third=grad(lap)
        states.append(torch.cat((value[:,c,None],g[:,:3],torch.stack([H[:,i,j] for i,j in hp],1),third[:,:3],g[:,3,None],H[:,3,:3]),1))
    return torch.stack(states,1)

def physical_loss(state,force,force_gradient,weights,nu=.1,lambda_g=.2):
    # Independent direct consumer; intentionally does not call C17 seed routine.
    hp=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
    def H(c,i,j):return state[:,c,4+hp.index((min(i,j),max(i,j)))]
    R=[];G=[]
    for i in range(3):
        R.append(state[:,i,13]+sum(state[:,j,0]*state[:,i,1+j] for j in range(3))-nu*sum(H(i,j,j) for j in range(3))+state[:,3,1+i]-force[:,i])
        for k in range(3):G.append(state[:,i,14+k]+sum(state[:,j,1+k]*state[:,i,1+j]+state[:,j,0]*H(i,j,k) for j in range(3))-nu*state[:,i,10+k]+H(3,i,k)-force_gradient[:,i,k])
    R.append(sum(state[:,i,1+i] for i in range(3)))
    for k in range(3):G.append(sum(H(i,i,k) for i in range(3)))
    return (weights*(sum(z*z for z in R)+lambda_g*sum(z*z for z in G))).sum()

def reference_loss_gradient(parameters,points,force,force_gradient,weights,backend):
    p=[(W.detach().clone().requires_grad_(True),b.detach().clone().requires_grad_(True)) for W,b in parameters]
    state=scalar_ad_state(p,points) if backend=='scalar_ad' else a24_contract(a24_forward(p,points))
    loss=physical_loss(state,force,force_gradient,weights)
    grads=torch.autograd.grad(loss,[x for layer in p for x in layer]);return loss.detach(),[(grads[i],grads[i+1]) for i in range(0,len(grads),2)],state.detach()
