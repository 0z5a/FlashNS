"""Check complete CPU gradients on canonical subsets; GPU qualification is separate."""
from pathlib import Path
import copy,hashlib,json,time
import argparse
import numpy as np
import torch
from torch_c36_packed import activate_packed
from box376.data import FAMILIES,flatten,initialize,torch_data,views
from box376.frozen_data import load_cache
from box376.normalized_objective import normalization_key
from box376.potential_objective import PreparedPotentialNormalizedObjective
from box376.potential_ad_objective import PreparedPotentialADObjective
parser=argparse.ArgumentParser(description="CPU subset qualification of opt-in packed C36 activation")
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--protocol',type=Path,required=True)
parser.add_argument('--training',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
output=args.output;output.mkdir(parents=True,exist_ok=False)
protocol=json.loads(args.protocol.read_text())
assert tuple(protocol['topology'])==(4,128,128,128,128,4)
torch.set_num_threads(1)
calibration=json.loads((args.inputs/'calibration.json').read_text());data=torch_data(load_cache(args.inputs/'training-source.npz'));data['normalization_key']=normalization_key(calibration)
probe=copy.deepcopy(data);selected={}
for name in FAMILIES:
 family=probe['families'][name];indices=np.array([1965,2489,1079,2043]) if name=='interior' else np.arange(min(4,len(family['points'])))
 selected[name]=indices.tolist()
 for key in ['points','weights','target','force','force_gradient']:
  if key in family:family[key]=family[key][indices].clone()
objective=PreparedPotentialNormalizedObjective(probe,calibration);packed=PreparedPotentialNormalizedObjective(probe,calibration,activation=activate_packed);baseline=PreparedPotentialADObjective(probe,calibration)
topology=(4,128,128,128,128,4);arrays={};checks=[]
for seed in [361901,361902]:
 final=args.training/f'seed{seed}-final.pt'
 checkpoint=torch.load(final,map_location='cpu',weights_only=True)
 assert checkpoint['protocol_sha256']==hashlib.sha256(args.protocol.read_bytes()).hexdigest()
 assert checkpoint['representation']=='C36-curl-potential'
 trained=checkpoint['theta']
 original=flatten(initialize(topology,seed))
 for label,theta in [('initial',original),('changed',original+.007),('trained',trained)]:
  parameters=views(theta,topology);reference=objective(parameters,2);ad=baseline(parameters,2)
  for chunk in [2,4]:
   actual=packed(parameters,chunk)
   for name,a,b,c in zip(['loss','gradient','numerators'],actual,reference,ad):
    value=a.detach().numpy();old=b.detach().numpy();auto=c.detach().numpy()
    ratio=float(np.max(np.abs(value-old)/(1e-11+1e-9*np.abs(old))));ad_ratio=float(np.max(np.abs(value-auto)/(1e-11+1e-9*np.abs(auto))))
    assert value.dtype==old.dtype==auto.dtype==np.float64 and value.shape==old.shape==auto.shape and np.isfinite(value).all()
    assert ratio<=1 and ad_ratio<=1,(seed,label,chunk,name,ratio,ad_ratio)
    key=f'{seed}_{label}_chunk{chunk}_{name}';arrays[key]=value;arrays[key+'_dense']=old;arrays[key+'_parameter_ad']=auto
    checks.append({'seed':seed,'parameters':theta.numel(),'case':label,'chunk':chunk,'quantity':name,'maximum_error_over_dense_budget':ratio,'maximum_error_over_parameter_AD_budget':ad_ratio})
  print(json.dumps({'seed':seed,'case':label,'checks':len(checks)}),flush=True)
np.savez_compressed(output/'raw-proof.npz',**arrays)
result={'state':'PASS','device':'cpu','topology':topology,'parameters':50692,'canonical_subset_indices':selected,'full_4999_point_GPU_qualification':False,'default_implementation_changed':False,'tolerance':{'atol':1e-11,'rtol':1e-9},'checks':checks,'raw_sha256':hashlib.sha256((output/'raw-proof.npz').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),'source_sha256':{name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest() for name in ['torch_c36_packed.py','torch_c36.py','box376/potential_model.py','box376/potential_objective.py']},'finished':time.time()}
(output/'proof.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(checks),'full parameter comparisons')
