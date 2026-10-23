"""Independent mpmath full four-variable degree4 Taylor potential oracle.

No import of production switch derivatives or spatial coefficient generator.
Uses the paper's E-ratio switch, generic polynomial exp/inverse, then curl.
"""
import itertools,math
import mpmath as mp
MULTI=sorted([a for a in itertools.product(range(5),repeat=4) if sum(a)<=4],key=lambda a:(sum(a),a))
INDEX={a:i for i,a in enumerate(MULTI)}
PRODUCT=[(i,j,INDEX[tuple(x+y for x,y in zip(a,b))]) for i,a in enumerate(MULTI) for j,b in enumerate(MULTI) if sum(a)+sum(b)<=4]

class Jet:
    def __init__(self,value):
        self.c=[mp.mpf('0') for _ in MULTI];self.c[0]=mp.mpf(value)
    @classmethod
    def variable(cls,value,axis):
        result=cls(value);result.c[INDEX[tuple(int(i==axis) for i in range(4))]]=mp.mpf('1');return result
    @classmethod
    def coefficients(cls,values):
        result=cls(0);result.c=list(values);return result
    def __add__(self,other):
        if not isinstance(other,Jet):other=Jet(other)
        return Jet.coefficients([a+b for a,b in zip(self.c,other.c)])
    __radd__=__add__
    def __neg__(self):return Jet.coefficients([-a for a in self.c])
    def __sub__(self,other):return self+-other if isinstance(other,Jet) else self+(-mp.mpf(other))
    def __rsub__(self,other):return other+-self
    def __mul__(self,other):
        if not isinstance(other,Jet):return Jet.coefficients([a*mp.mpf(other) for a in self.c])
        result=[mp.mpf('0') for _ in MULTI]
        for i,j,k in PRODUCT:
            if self.c[i] and other.c[j]:result[k]+=self.c[i]*other.c[j]
        return Jet.coefficients(result)
    __rmul__=__mul__
    def inverse(self):
        assert self.c[0]!=0;delta=(self-self.c[0])*(-1/self.c[0]);result=Jet(1);power=Jet(1)
        for _ in range(4):power=power*delta;result=result+power
        return result*(1/self.c[0])
    def __truediv__(self,other):return self*other.inverse() if isinstance(other,Jet) else self*(1/mp.mpf(other))
    def __rtruediv__(self,other):return self.inverse()*other
    def exp(self):
        delta=self-self.c[0];power=Jet(1);result=Jet(1)
        for k in range(1,5):power=power*delta;result=result+power/math.factorial(k)
        return result*mp.exp(self.c[0])
    def derivative(self,axes):
        multi=tuple(axes.count(i) for i in range(4));return self.c[INDEX[multi]]*math.prod(math.factorial(i) for i in multi)

def sigma(q):
    if q.c[0]<=0:return Jet(0)
    if q.c[0]>=1:return Jet(1)
    left=(-q.inverse()).exp();right=(-(1-q).inverse()).exp();return left/(left+right)

def sigma_prime(q):
    if q.c[0]<=0 or q.c[0]>=1:return Jet(0)
    left=(-q.inverse()).exp();right=(-(1-q).inverse()).exp();den=left+right
    return left*right/den/den*(q.inverse()*q.inverse()+(1-q).inverse()*(1-q).inverse())

def source(point,eta=None,nu=None):
    eta=mp.mpf(1)/32 if eta is None else mp.mpf(eta)
    nu=mp.mpf(1)/100 if nu is None else mp.mpf(nu)
    x=[Jet.variable(value,i) for i,value in enumerate(point)];t=x[3];q=2*t-mp.mpf(1)/2;s=sigma(q);sp=2*sigma_prime(q)
    logs=[mp.log(2),mp.log(mp.mpf(3)/2),-mp.log(2)-mp.log(mp.mpf(3)/2)]
    displacement=[mp.mpf(1)/8,mp.mpf(1)/16,mp.mpf(1)/32]
    c=[mp.mpf(3)/8+s*d for d in displacement];v=[sp*d for d in displacement]
    D=[(s*logs[0]).exp(),(s*logs[1]).exp()];D.append(1/(D[0]*D[1]));h=[d/16 for d in D]
    ell=[sp*logs[0],sp*logs[1]];ell.append(-ell[0]-ell[1]);y=[x[i]-c[i] for i in range(3)]
    chi=Jet(1)
    for j in range(3):chi=chi*sigma((h[j]+2*eta-y[j])/eta)*sigma((h[j]+2*eta+y[j])/eta)
    A=[]
    for i,j,k in [(0,1,2),(1,2,0),(2,0,1)]:A.append(chi*((v[j]*y[k]-v[k]*y[j])/2+(ell[j]-ell[k])*y[j]*y[k]/3))
    def d(i,axes):
        j=(i+1)%3;k=(i+2)%3;return A[k].derivative(axes+[j])-A[j].derivative(axes+[k])
    U=[d(i,[]) for i in range(3)];G=[[d(i,[j]) for j in range(3)] for i in range(3)]
    H=[[[d(i,[j,k]) for k in range(3)] for j in range(3)] for i in range(3)]
    Ut=[d(i,[3]) for i in range(3)];GUt=[[d(i,[j,3]) for j in range(3)] for i in range(3)]
    lap=[sum(d(i,[j,j]) for j in range(3)) for i in range(3)]
    gradlap=[[sum(d(i,[k,j,j]) for j in range(3)) for k in range(3)] for i in range(3)]
    conv=[sum(U[j]*G[i][j] for j in range(3)) for i in range(3)]
    f=[Ut[i]+conv[i]-nu*lap[i] for i in range(3)]
    df=[[GUt[i][k]+sum(G[j][k]*G[i][j]+U[j]*H[i][j][k] for j in range(3))-nu*gradlap[i][k] for k in range(3)] for i in range(3)]
    return {'U':U,'G':G,'H':H,'Ut':Ut,'GUt':GUt,'lap':lap,'gradlap':gradlap,'f':f,'df':df,'convection':conv,'divergence':sum(G[i][i] for i in range(3)),'div_force':sum(df[i][i] for i in range(3)),'trace_gradU_squared':sum(G[i][j]*G[j][i] for i in range(3) for j in range(3))}
