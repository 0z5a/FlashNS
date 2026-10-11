"""Vectorized physical source jets for Math376 single compact-curl box.

Spatial potential Taylor coefficients through degree4 and dt through space2.
Explicit stable switch derivatives; physical partial_t holds x fixed.
The field/forcing are fixed and never depend on trainable parameters.
"""
import itertools,math
import numpy as np

MULTI=sorted([a for a in itertools.product(range(5),repeat=3) if sum(a)<=4],key=lambda a:(sum(a),a))
INDEX={a:i for i,a in enumerate(MULTI)}
FACT=np.array([math.prod(math.factorial(v) for v in a) for a in MULTI])
PRODUCT=[(i,j,INDEX[tuple(x+y for x,y in zip(a,b))]) for i,a in enumerate(MULTI) for j,b in enumerate(MULTI) if sum(a)+sum(b)<=4]
C0=np.array([3/8]*3);DISPLACEMENT=np.array([1/8,1/16,1/32]);H0=np.array([1/16]*3)
LOGS=np.array([np.log(2.),np.log(1.5),-np.log(2.)-np.log(1.5)])


def switch(q):
    """Value and raw derivatives0..4; plateau branches never divide by zero."""
    q=np.asarray(q,dtype=float);result=np.zeros(q.shape+(5,));result[...,0]=q>=1
    interior=(q>0)&(q<1)
    if not np.any(interior):return result
    a=q[interior];h=1/(1-a)-1/a
    # |h|>700: all four derivatives are below representable useful physical
    # tolerance. Avoid multiplying underflow zero by overflowing powers.
    active=np.abs(h)<=700;work=np.zeros((len(a),5));work[:,0]=h>0
    b=a[active];hh=h[active];e=np.exp(-np.abs(hh));sigma=np.where(hh>=0,1/(1+e),e/(1+e));d=e/(1+e)**2
    g=[None,d,(1-2*sigma)*d,(1-6*d)*d,(1-2*sigma)*(1-12*d)*d]
    H=[None]+[math.factorial(n)*((1-b)**(-n-1)+(-1)**(n+1)*b**(-n-1)) for n in range(1,5)]
    work[active,0]=sigma
    work[active,1]=g[1]*H[1]
    work[active,2]=g[2]*H[1]**2+g[1]*H[2]
    work[active,3]=g[3]*H[1]**3+3*g[2]*H[1]*H[2]+g[1]*H[3]
    work[active,4]=g[4]*H[1]**4+6*g[3]*H[1]**2*H[2]+g[2]*(3*H[2]**2+4*H[1]*H[3])+g[1]*H[4]
    result[interior]=work
    assert np.all(np.isfinite(result))
    return result


def motion(t):
    derivatives=switch(2*np.asarray(t)-.5);s=derivatives[...,0];sp=2*derivatives[...,1];spp=4*derivatives[...,2]
    D=np.empty(s.shape+(3,));D[...,0]=np.exp(s*LOGS[0]);D[...,1]=np.exp(s*LOGS[1]);D[...,2]=1/(D[...,0]*D[...,1])
    ell=sp[...,None]*LOGS;ell[...,2]=-ell[...,0]-ell[...,1]
    ell_t=spp[...,None]*LOGS;ell_t[...,2]=-ell_t[...,0]-ell_t[...,1]
    return {'s':s,'sp':sp,'spp':spp,'D':D,'h':H0*D,'c':C0+s[...,None]*DISPLACEMENT,'v':sp[...,None]*DISPLACEMENT,'acceleration':spp[...,None]*DISPLACEMENT,'ell':ell,'ell_t':ell_t}


def multiply(a,b,maximum=4):
    c=np.zeros_like(a+b)
    for i,j,k in PRODUCT:
        if sum(MULTI[k])<=maximum:c[...,k]+=a[...,i]*b[...,j]
    return c


def source(points,eta=1/32,nu=.01):
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=4 or not np.all(np.isfinite(points)):raise ValueError('Finite Nx4 physical coordinates required')
    if eta<=0 or nu<=0:raise ValueError('Positive eta and viscosity required')
    n=len(points);m=motion(points[:,3]);y=points[:,:3]-m['c'];v=m['v'];acc=m['acceleration'];ell=m['ell'];ell_t=m['ell_t'];h=m['h'];ht=h*ell
    axis=[];axis_t=[]
    for j in range(3):
        minus=switch((h[:,j]+2*eta-y[:,j])/eta);plus=switch((h[:,j]+2*eta+y[:,j])/eta)
        mt=(ht[:,j]+v[:,j])/eta;pt=(ht[:,j]-v[:,j])/eta
        coefficients=np.empty((n,5));tcoefficients=np.zeros((n,3))
        for degree in range(5):
            coefficients[:,degree]=sum(math.comb(degree,k)*minus[:,k]*plus[:,degree-k]*(-1/eta)**k*(1/eta)**(degree-k) for k in range(degree+1))/math.factorial(degree)
            if degree<=2:
                tcoefficients[:,degree]=sum(math.comb(degree,k)*(minus[:,k+1]*mt*plus[:,degree-k]+minus[:,k]*plus[:,degree-k+1]*pt)*(-1/eta)**k*(1/eta)**(degree-k) for k in range(degree+1))/math.factorial(degree)
        axis.append(coefficients);axis_t.append(tcoefficients)
    chi=np.stack([axis[0][:,a]*axis[1][:,b]*axis[2][:,c] for a,b,c in MULTI],-1)
    chi_t=np.zeros_like(chi)
    for q,alpha in enumerate(MULTI):
        if sum(alpha)<=2:
            chi_t[:,q]=sum(axis_t[j][:,alpha[j]]*np.prod(np.stack([axis[k][:,alpha[k]] for k in range(3) if k!=j],-1),-1) for j in range(3))
    potentials=[];time_potentials=[]
    for i,j,k in [(0,1,2),(1,2,0),(2,0,1)]:
        B=np.zeros_like(chi);Bt=np.zeros_like(chi);a=(ell[:,j]-ell[:,k])/3;at=(ell_t[:,j]-ell_t[:,k])/3
        B[:,0]=.5*(v[:,j]*y[:,k]-v[:,k]*y[:,j])+a*y[:,j]*y[:,k]
        Bt[:,0]=.5*(acc[:,j]*y[:,k]-acc[:,k]*y[:,j])+at*y[:,j]*y[:,k]-a*(v[:,j]*y[:,k]+y[:,j]*v[:,k])
        ej=tuple(int(r==j) for r in range(3));ek=tuple(int(r==k) for r in range(3));ejk=tuple(int(r==j)+int(r==k) for r in range(3))
        B[:,INDEX[ej]]=-.5*v[:,k]+a*y[:,k];B[:,INDEX[ek]]=.5*v[:,j]+a*y[:,j];B[:,INDEX[ejk]]=a
        Bt[:,INDEX[ej]]=-.5*acc[:,k]+at*y[:,k]-a*v[:,k];Bt[:,INDEX[ek]]=.5*acc[:,j]+at*y[:,j]-a*v[:,j];Bt[:,INDEX[ejk]]=at
        potentials.append(multiply(chi,B));time_potentials.append(multiply(chi_t,B,2)+multiply(chi,Bt,2))
    A=np.stack(potentials,1);At=np.stack(time_potentials,1)
    def derivative(p,axes):
        alpha=tuple(axes.count(j) for j in range(3));q=INDEX[alpha];return p[...,q]*FACT[q]
    def curl_derivative(p,i,axes):
        j=(i+1)%3;k=(i+2)%3;return derivative(p[:,k],axes+[j])-derivative(p[:,j],axes+[k])
    U=np.stack([curl_derivative(A,i,[]) for i in range(3)],-1)
    G=np.stack([np.stack([curl_derivative(A,i,[j]) for j in range(3)],-1) for i in range(3)],1)
    H=np.stack([np.stack([np.stack([curl_derivative(A,i,[j,k]) for k in range(3)],-1) for j in range(3)],1) for i in range(3)],1)
    lap=np.stack([sum(curl_derivative(A,i,[j,j]) for j in range(3)) for i in range(3)],-1)
    gradlap=np.stack([np.stack([sum(curl_derivative(A,i,[j,j,k]) for j in range(3)) for k in range(3)],-1) for i in range(3)],1)
    Ut=np.stack([curl_derivative(At,i,[]) for i in range(3)],-1)
    GUt=np.stack([np.stack([curl_derivative(At,i,[j]) for j in range(3)],-1) for i in range(3)],1)
    convection=np.einsum('nj,nij->ni',U,G)
    f=Ut+convection-nu*lap
    df=GUt+np.einsum('nij,njk->nik',G,G)+np.einsum('nj,nijk->nik',U,H)-nu*gradlap
    result={'U':U,'G':G,'H':H,'Ut':Ut,'GUt':GUt,'lap':lap,'gradlap':gradlap,'f':f,'df':df,'convection':convection,'divergence':np.trace(G,axis1=1,axis2=2),'div_force':np.trace(df,axis1=1,axis2=2),'trace_gradU_squared':np.einsum('nij,nji->n',G,G),'chi':chi[:,0]}
    assert all(np.all(np.isfinite(a)) for a in result.values())
    return result


def velocity(points,eta=1/32):
    points=np.asarray(points,dtype=float);m=motion(points[:,3]);y=points[:,:3]-m['c'];v=m['v'];ell=m['ell'];h=m['h'];axis=[];grad_axis=[]
    for j in range(3):
        minus=switch((h[:,j]+2*eta-y[:,j])/eta);plus=switch((h[:,j]+2*eta+y[:,j])/eta)
        axis.append(minus[:,0]*plus[:,0]);grad_axis.append((-minus[:,1]*plus[:,0]+minus[:,0]*plus[:,1])/eta)
    chi=axis[0]*axis[1]*axis[2]
    grad=np.stack([grad_axis[j]*axis[(j+1)%3]*axis[(j+2)%3] for j in range(3)],-1)
    B=.5*np.cross(v,y)+np.cross(ell*y,y)/3
    return chi[:,None]*(v+ell*y)+np.cross(grad,B)


def trajectory(t,initial):
    initial=np.asarray(initial,dtype=float);m=motion(np.asarray(t));return m['c']+m['D']*(initial-C0)
