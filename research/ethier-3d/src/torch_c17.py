"""FP64 Torch CPU/CUDA correctness executor for transient 3D spatial gPINN C17.

Coordinates are (x1,x2,x3,t); diffusion excludes time. State [N,width,17]
stores value, g(3), packed H(6), grad spatial Laplacian(3), dt, grad dt(3).
This is an explicit first parameter VJP. GPU qualification remains separately required.
"""
import torch
PAIRS=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))

def unpack(s):
    H=torch.empty(s.shape[:-1]+(3,3),dtype=s.dtype,device=s.device)
    for q,(i,j) in enumerate(PAIRS):H[...,i,j]=H[...,j,i]=s[...,4+q]
    return s[...,0],s[...,1:4],H,s[...,10:13],s[...,13],s[...,14:17]

def pack(v,g,H,l,m,n):
    return torch.cat((v[...,None],g,torch.stack([H[...,i,j] for i,j in PAIRS],-1),l,m[...,None],n),-1)

def coefficients(z):
    y=torch.tanh(z)
    e=torch.exp(-2*torch.abs(z))
    a1=4*e/(1+e)**2
    a2=-2*y*a1;a3=4*y*y*a1-2*a1*a1;a4=16*y*a1*a1-8*y**3*a1
    return y,a1,a2,a3,a4

def activate(s):
    z,g,H,l,m,n=unpack(s);y,a1,a2,a3,_=coefficients(z)
    tr=H.diagonal(dim1=-2,dim2=-1).sum(-1);gg=torch.sum(g*g,-1)
    A=2*torch.einsum('...ij,...j->...i',H,g)+g*tr[...,None]
    return pack(y,a1[...,None]*g,a1[...,None,None]*H+a2[...,None,None]*g[..., :,None]*g[...,None,:],a1[...,None]*l+a2[...,None]*A+a3[...,None]*g*gg[...,None],a1*m,a1[...,None]*n+a2[...,None]*m[...,None]*g)

def activate_vjp(s,b):
    z,g,H,l,m,n=unpack(s);bz,bg,B,bl,bm,bn=unpack(b)
    B=B*(torch.eye(3,dtype=s.dtype,device=s.device)*.5+.5)
    _,a1,a2,a3,a4=coefficients(z);tr=H.diagonal(dim1=-2,dim2=-1).sum(-1);gg=torch.sum(g*g,-1);dot=torch.sum(bl*g,-1)
    A=2*torch.einsum('...ij,...j->...i',H,g)+g*tr[...,None]
    v=a1*bz+a2*torch.sum(bg*g,-1)+torch.sum(B*(a2[...,None,None]*H+a3[...,None,None]*g[..., :,None]*g[...,None,:]),(-2,-1))+torch.sum(bl*(a2[...,None]*l+a3[...,None]*A+a4[...,None]*g*gg[...,None]),-1)+bm*a2*m+torch.sum(bn*(a2[...,None]*n+a3[...,None]*m[...,None]*g),-1)
    grad=a1[...,None]*bg+a2[...,None]*torch.einsum('...ij,...j->...i',B+torch.swapaxes(B,-1,-2),g)+a2[...,None]*(2*torch.einsum('...ji,...j->...i',H,bl)+tr[...,None]*bl)+a3[...,None]*(gg[...,None]*bl+2*dot[...,None]*g)+a2[...,None]*m[...,None]*bn
    BH=a1[...,None,None]*B+a2[...,None,None]*(2*bl[..., :,None]*g[...,None,:]+dot[...,None,None]*torch.eye(3,dtype=s.dtype,device=s.device))
    hp=torch.stack([BH[...,i,j] if i==j else BH[...,i,j]+BH[...,j,i] for i,j in PAIRS],-1)
    return torch.cat((v[...,None],grad,hp,a1[...,None]*bl,(a1*bm+a2*torch.sum(bn*g,-1))[...,None],a1[...,None]*bn),-1)

def coordinate_state(points):
    s=torch.zeros((len(points),4,17),dtype=points.dtype,device=points.device);s[:,:,0]=points
    for i in range(3):s[:,i,1+i]=1
    s[:,3,13]=1
    return s

def forward(parameters,points):
    s=coordinate_state(points);tape=[]
    for k,(W,bias) in enumerate(parameters):
        z=torch.einsum('nwq,ow->noq',s,W);z[:,:,0]+=bias
        tape.append((s,z));s=activate(z) if k<len(parameters)-1 else z
    return s,tape

def parameter_vjp(parameters,tape,cotangent):
    b=cotangent;grads=[]
    for k in range(len(parameters)-1,-1,-1):
        s,z=tape[k];W,_=parameters[k]
        if k<len(parameters)-1:b=activate_vjp(z,b)
        grads.append((torch.einsum('noq,nwq->ow',b,s),torch.sum(b[:,:,0],0)))
        b=torch.einsum('noq,ow->nwq',b,W)
    return list(reversed(grads))

def observations(s,force,force_gradient,nu):
    if s.shape[1:]!=(4,17):raise ValueError('Outputs must be u1,u2,u3,p')
    u,g,H,l,m,n=unpack(s);v=u[:,:3];gu=g[:,:3];Hu=H[:,:3]
    R=m[:,:3]+torch.einsum('nj,nij->ni',v,gu)-nu*Hu.diagonal(dim1=-2,dim2=-1).sum(-1)+g[:,3]-force
    RG=n[:,:3]+torch.einsum('nij,njk->nik',gu,gu)+torch.einsum('nj,nijk->nik',v,Hu)-nu*l[:,:3]+H[:,3]-force_gradient
    D=torch.einsum('nii->n',gu);DG=sum(Hu[:,i,i,:] for i in range(3))
    return torch.cat((R,D[:,None]),1),torch.cat((RG,DG[:,None,:]),1)

def loss_and_seed(s,force,force_gradient,weights,nu=.1,lambda_g=.2):
    R,RG=observations(s,force,force_gradient,nu)
    loss=torch.sum(weights*(torch.sum(R*R,1)+lambda_g*torch.sum(RG*RG,(1,2))))
    br=2*weights[:,None]*R;bg=2*lambda_g*weights[:,None,None]*RG
    b=torch.zeros_like(s);u,g,H,_,_,_=unpack(s)
    def addH(component,i,j,val):b[:,component,4+PAIRS.index((min(i,j),max(i,j)))]+=val
    for i in range(3):
        b[:,i,13]+=br[:,i];b[:,3,1+i]+=br[:,i];b[:,i,1+i]+=br[:,3]
        for j in range(3):
            b[:,j,0]+=br[:,i]*g[:,i,j];b[:,i,1+j]+=br[:,i]*u[:,j];addH(i,j,j,-nu*br[:,i])
        for k in range(3):
            v=bg[:,i,k];b[:,i,14+k]+=v;b[:,i,10+k]-=nu*v;addH(3,i,k,v);addH(i,i,k,bg[:,3,k])
            for j in range(3):
                b[:,i,1+j]+=v*g[:,j,k];b[:,j,1+k]+=v*g[:,i,j];b[:,j,0]+=v*H[:,i,j,k];addH(i,j,k,v*u[:,j])
    return loss,b

def loss_gradient(parameters,points,force,force_gradient,weights,**kwargs):
    s,tape=forward(parameters,points);loss,seed=loss_and_seed(s,force,force_gradient,weights,**kwargs)
    return loss,parameter_vjp(parameters,tape,seed)
