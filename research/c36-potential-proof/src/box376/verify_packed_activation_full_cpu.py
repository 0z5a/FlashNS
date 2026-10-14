"""CPU full-point checks against preserved native anchors; GPU qualification is separate."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
from torch_c36_packed import activate_packed
from torch_c36_packed_vjp import activate_packed_vjp
from box376.data import FAMILIES,flatten,initialize,torch_data,views
from box376.frozen_data import load_cache
from box376.normalized_objective import normalization_key
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.potential_ad_objective import PreparedPotentialADObjective
import argparse
parser=argparse.ArgumentParser(description="Full canonical CPU qualification of opt-in packed C36 activation")
for name in ['inputs','protocol','gather-proof','training','output']:
 parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--activation-mode',choices=['eager','compiled','analytic'],default='eager')
args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=False)
gather=args.gather_proof;proof=json.loads((gather/'proof.json').read_text())
assert proof['state']=='PASS' and proof['points']==4999 and proof['parameters_per_seed']==50692
assert proof['reference_sha256']=='172174ea49a82a21e0616571ae303560d275317d8b5319c22ea49dbd628d8b66'
assert hashlib.sha256((gather/'raw-proof.npz').read_bytes()).hexdigest()==proof['raw_sha256']=='7d1db3ce8fb6dbaeeeabff46e4e981633b1761c5dd7cc8c82eaf79c0c939320b'
with np.load(gather/'raw-proof.npz',allow_pickle=False) as raw:
 reference={seed:tuple(raw[f'seed{seed}-explicit128-native_reference_{name}'].copy() for name in ['loss','gradient','numerators']) for seed in [361901,361902]}
assert hashlib.sha256((args.inputs/'training-source.npz').read_bytes()).hexdigest()=='d378c28991dd9a852297f22c94f0727d4c0c1e5d4140fe718324859c09b78bc2'
cal=json.loads((args.inputs/'calibration.json').read_text());data=torch_data(load_cache(args.inputs/'training-source.npz'));data['normalization_key']=normalization_key(cal)
torch.set_num_threads(1)
print(json.dumps({'phase':'prepare-full-4999-point-CPU-geometry','unix':time.time()}),flush=True)
packed=PreparedPotentialNormalizedObjective(data,cal,activation=activate_packed);dense=PreparedPotentialNormalizedObjective(data,cal);ad=PreparedPotentialADObjective(data,cal)
variants=[('packed128',packed,128),('packed64',packed,64),('dense128',dense,128),('parameterAD128',ad,128)]
if args.activation_mode!='eager':
 activate_packed(torch.zeros(128,128,36,dtype=torch.float64))
 compiled=torch.compile(activate_packed,fullgraph=True,dynamic=False)
 transpose=torch.compile(activate_packed_vjp,fullgraph=True,dynamic=False) if args.activation_mode=='analytic' else None
 packed=PreparedPotentialNormalizedObjective(data,cal,activation=compiled,activation_vjp=transpose)
 packed_ad=PreparedPotentialADObjective(data,cal,activation=activate_packed)
 compiled_ad=PreparedPotentialADObjective(data,cal,activation=compiled)
 variants=[('packed128',packed,128),('packed64',packed,64),('packedParameterAD128',packed_ad,128),('compiledParameterAD128',compiled_ad,128),('originalParameterAD128',ad,128)]
arrays={};checks=[];timings=[];topology=(4,128,128,128,128,4)
def check(key,actual,expected):
 for name,x,y in zip(['loss','gradient','numerators'],actual,expected):
  x=x.detach().numpy();y=np.asarray(y);assert x.dtype==y.dtype==np.float64 and x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
  ratio=float(np.max(np.abs(x-y)/(1e-11+1e-9*np.abs(y))));assert ratio<=1,(key,name,ratio)
  arrays[key+'_actual_'+name]=x;arrays[key+'_reference_'+name]=y.copy();checks.append({'name':key+'_'+name,'maximum_error_over_original_budget':ratio})
 print(json.dumps({'phase':key,'checks':len(checks),'unix':time.time()}),flush=True)
for seed in [361901,361902]:
 checkpoint=torch.load(args.training/f'seed{seed}-initial.pt',map_location='cpu',weights_only=True)
 assert checkpoint['representation']=='C36-curl-potential'
 assert checkpoint['protocol_sha256']==hashlib.sha256(args.protocol.read_bytes()).hexdigest()
 theta=checkpoint['theta'].clone()
 parameters=views(theta,topology)
 for label,engine,chunk in variants:
  began=time.perf_counter();actual=engine(parameters,chunk);duration=time.perf_counter()-began
  check(f'seed{seed}-{label}-frozen-native',actual,reference[seed]);timings.append({'seed':seed,'engine':label,'evaluation_seconds':duration,'scope':'Single-call CPU diagnostics only; not formal paired GPU speed'})
 raw_key=f'seed{seed}-changed'
 theta=theta+.001;parameters=views(theta,topology);expected=ad(parameters,128)
 check(raw_key+'-packed128-parameterAD',packed(parameters,128),tuple(x.numpy() for x in expected))
 check(raw_key+'-packed64-parameterAD',packed(parameters,64),tuple(x.numpy() for x in expected))
np.savez_compressed(output/'raw-proof.npz',**arrays)
report={'state':'PASS','device':'cpu','points':sum(len(data['families'][n]['points']) for n in FAMILIES),'parameters_per_seed':50692,'seeds':[361901,361902],'checks':checks,'atol':1e-11,'rtol':1e-9,'tolerance_relaxed':False,'reference_provenance':{'complete_native_GPU_raw_sha256':proof['reference_sha256'],'preserved_gather_raw_sha256':proof['raw_sha256'],'fresh_native_coordinate_AD_repeated':False,'initial_theta_loaded_from_original_verified_checkpoints':True,'regenerated_weights_not_used':True},'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),'raw_sha256':hashlib.sha256((output/'raw-proof.npz').read_bytes()).hexdigest(),'GPU_qualification_run':False,'CUDA_graph_qualification_run':False,'default_activation_changed':False,'diagnostic_timings':timings,'activation_mode':args.activation_mode,'torch_version':torch.__version__,'source_sha256':{n:hashlib.sha256((Path(__file__).resolve().parents[1]/n).read_bytes()).hexdigest() for n in ['torch_c36_packed.py','torch_c36_packed_vjp.py','torch_c36.py','box376/potential_model.py','box376/potential_objective.py','box376/potential_ad_objective.py']},'finished':time.time()}
(output/'proof.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(checks),'full canonical comparisons',flush=True)
