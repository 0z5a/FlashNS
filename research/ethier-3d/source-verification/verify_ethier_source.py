from pathlib import Path
import json,hashlib,time
import mpmath as mp
import numpy as np
import ethier_source as source
ROOT=Path(__file__).parent;started=time.time();records=[];anchors=[]
points=[['0','0','0','0'],['-.3','.2','-.1','.15'],['.5','-.5','.5','.25'],['.07','-.12','.21','.05']]
for precision in [80,160]:
 with mp.workdps(precision):
  a=mp.mpf('.5');d=mp.mpf('1');nu=mp.mpf('1')
  for raw in points:
   q=tuple(mp.mpf(x) for x in raw)
   funcs=[lambda *x,i=i:source.mp_fields(*x,a,d,nu)[i] for i in range(4)]
   def diff(i,axes):return mp.diff(funcs[i],q,tuple(axes.count(j) for j in range(4)))
   value=[f(*q) for f in funcs];momentum=[diff(i,[3])+sum(value[j]*diff(i,[j]) for j in range(3))-nu*sum(diff(i,[j,j]) for j in range(3))+diff(3,[i]) for i in range(3)]
   divergence=sum(diff(i,[i]) for i in range(3));gradient=[]
   for i in range(3):
    for k in range(3):gradient.append(diff(i,[3,k])+sum(diff(j,[k])*diff(i,[j])+value[j]*diff(i,[j,k]) for j in range(3))-nu*sum(diff(i,[j,j,k]) for j in range(3))+diff(3,[i,k]))
   gradient+= [sum(diff(i,[i,k]) for i in range(3)) for k in range(3)]
   maximum=max(abs(v) for v in momentum+[divergence]+gradient);assert maximum<mp.mpf('1e-65')
   records.append({'precision_dps':precision,'point':raw,'max_16_observation_residual':mp.nstr(maximum,12),'momentum':[mp.nstr(v,12) for v in momentum],'divergence':mp.nstr(divergence,12)})
   double=source.fields(np.array([[float(x) for x in q]]))[0];high=np.array([float(x) for x in value]);assert np.max(np.abs(double-high))<2e-15
   if precision==160:anchors.append(high.tolist())
report={'state':'ETHIER_PHYSICAL_16_OBSERVATIONS_HIGH_PRECISION_PASS','parameters':{'a':.5,'d':1.,'nu':1.},'coordinates':['x','y','z','t'],'pressure_gauge':'p(x,t)-p(0,0,0,t)','force':'ZERO','precision_dps':[80,160],'checks':records,'reference_values_160dps':anchors,'wall_seconds':time.time()-started,'source_sha256':hashlib.sha256((ROOT/'ethier_source.py').read_bytes()).hexdigest(),'primary_reference':'Ethier and Steinman (1994), Exact fully 3D Navier-Stokes solutions for benchmarking, Eq.15','training_convergence':'NOT_RUN'};(ROOT/'evidence/ethier-source-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'state':report['state'],'points':len(points),'precisions':2,'wall_seconds':report['wall_seconds']}))
