"""Offbox NumPy audit of attempt 2 of the compiled external-baseline batch (audit_ext.py with its protocol)."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import c17
import taylor_oracle as q35


def values(p,x):
 for i,(W,b) in enumerate(p):
  x=x@W.T+b
  if i<len(p)-1:x=np.tanh(x)
 return x


def value_gradient(p,x,y,indices,coefficient):
 v=x;tape=[]
 for i,(W,b) in enumerate(p):
  old=v;v=old@W.T+b
  if i<len(p)-1:v=np.tanh(v)
  tape.append((old,v))
 diff=v[:,indices]-y[:,indices];loss=coefficient*np.sum(diff*diff)/len(v)
 seed=np.zeros_like(v);seed[:,indices]=2*coefficient*diff/len(v);grads=[]
 for i in range(len(p)-1,-1,-1):
  old,y=tape[i]
  if i<len(p)-1:seed=seed*(1-y*y)
  grads.append((seed.T@old,np.sum(seed,axis=0)));seed=seed@p[i][0]
 return loss,list(reversed(grads))


def objective(p,d,protocol):
 x=d['interior'];loss,g=c17.loss_gradient(p,x,np.zeros((len(x),3)),np.zeros((len(x),3,3)),np.full(len(x),1/len(x)),nu=protocol['parameters']['nu'],lambda_g=protocol['lambda_g'])
 for key,idx,coef in [('boundary',slice(0,3),protocol['boundary_weight']),('initial',slice(0,3),protocol['initial_weight']),('gauge',slice(3,4),protocol['gauge_weight'])]:
  l,add=value_gradient(p,d[key],d[key+'_target'],idx,coef);loss+=l;g=[(W+a,b+c) for (W,b),(a,c) in zip(g,add)]
 return loss,g


def observations(s,nu):
 hp=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
 def H(c,i,j):return s[:,c,4+hp.index((min(i,j),max(i,j)))]
 r=[];rg=[]
 for i in range(3):
  r.append(s[:,i,13]+sum(s[:,j,0]*s[:,i,1+j] for j in range(3))-nu*sum(H(i,j,j) for j in range(3))+s[:,3,1+i])
  rg.append(np.stack([s[:,i,14+k]+sum(s[:,j,1+k]*s[:,i,1+j]+s[:,j,0]*H(i,j,k) for j in range(3))-nu*s[:,i,10+k]+H(3,i,k) for k in range(3)],-1))
 r.append(sum(s[:,i,1+i] for i in range(3)));rg.append(np.stack([sum(H(i,i,k) for i in range(3)) for k in range(3)],-1))
 return np.stack(r,-1),np.stack(rg,1)


def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--self-test',action='store_true',help='driver self-test: admit CPU smoke runs');args=a.parse_args();root=args.root
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 manifest=json.loads((root/'offbox-manifest.json').read_text())
 for name,h in manifest['files'].items():assert sha(root/name)==h,name
 source=json.loads((root/'manifest.json').read_text())
 for name,v in source['files'].items():assert sha(root/name)==v['sha256']
 protocol=json.loads((root/'protocol-ext3-20261017.json').read_text());data_file=root/'data-v2.npz';assert sha(data_file)==protocol['data_file_sha256']
 raw=np.load(data_file);data={k:raw[k] for k in raw.files if not k[0].isdigit()};small={k:v[:7] for k,v in data.items()}
 controller=json.loads((root/'evidence/controller.json').read_text());rows=[];pairs=0;entries=0;maximum=0.
 def compare(a,b,lo=2e-12,hi=2e-11):
  nonlocal entries,maximum
  assert a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
  error=np.abs(a-b);assert np.all(error<=lo+hi*np.abs(b)),float(np.max(error))
  entries+=a.size;maximum=max(maximum,float(np.max(error)))
 for child in controller['children']:
  directory=root/'evidence'/child['id'];summary=json.loads((directory/'summary.json').read_text());assert args.self_test or not summary['cpu_smoke_only']
  row={'id':child['id'],'arm':child['arm'],'seed':child['seed'],'actual_exit_code':child['actual_exit_code'],'state':summary['state'],'accepted':False}
  if child.get('stopped_at_deadline') or summary['state'] not in ('ACCEPTED','NONCONVERGED','RUNTIME_FAILURE','CPU_SMOKE_COMPLETE'):
   row.update(state='STOPPED_AT_DEADLINE_OR_INCOMPLETE',last_state=summary['state']);rows.append(row);continue
  if summary['state']=='RUNTIME_FAILURE':
   assert child['actual_exit_code']!=0;row['error']=summary['error'];rows.append(row);continue
  assert child['actual_exit_code']==summary['actual_exit_code']==0
  assert sha(directory/'qualification.npz')==summary['qualification_sha256'];qr=np.load(directory/'qualification.npz')
  for scope,d in [('scalar_ad',small),('full_replay',data)]:
   for step in range(3):
    name=f'{scope}_update{step}';p=[(qr[f'{name}_W{i}'],qr[f'{name}_b{i}']) for i in range(3)]
    for layer,(W,b) in enumerate(p):
     if step==0:
      assert np.array_equal(W,raw[f"{child['seed']}_W{layer}"]) and np.array_equal(b,raw[f"{child['seed']}_b{layer}"])
     else:
      previous=f'{scope}_update{step-1}'
      compare(W,qr[f'{previous}_W{layer}']-1e-4*qr[f'{previous}_{1+2*layer}_actual'])
      compare(b,qr[f'{previous}_b{layer}']-1e-4*qr[f'{previous}_{2+2*layer}_actual'])
    loss,g=objective(p,d,protocol)
    expected=[np.asarray(loss)]+[v for pair in g for v in pair]
    for i,v in enumerate(expected):
     compare(qr[f'{name}_{i}_actual'],v);compare(qr[f'{name}_{i}_reference'],v);pairs+=2
  assert sha(directory/'final-raw.npz')==summary['final_raw_sha256'];fr=np.load(directory/'final-raw.npz');p=[(fr[f'final_W{i}'],fr[f'final_b{i}']) for i in range(3)]
  s=q35.to_c17(q35.forward(p,data['validation']));r,rg=observations(s,protocol['parameters']['nu'])
  compare(fr['validation_state'],s,5e-11,5e-10);compare(fr['residual'],r,5e-11,5e-10);compare(fr['residual_gradient'],rg,5e-11,5e-10)
  preds={k:values(p,data[k]) for k in ['validation','validation_boundary','validation_initial','validation_gauge']}
  for key,pred in preds.items():compare(fr[key+'_prediction'],pred,5e-12,5e-11)
  scales=protocol['scales'];rms=lambda v:float(np.sqrt(np.mean(v*v)))
  pred=preds['validation'];target=data['validation_target']
  metrics={'normalized_momentum_rms':rms(r[:,:3])/scales['R'],'normalized_divergence_rms':rms(r[:,3])*scales['L']/scales['U'],'normalized_spatial_residual_gradient_rms':rms(rg)*scales['L']/scales['R'],'normalized_boundary_velocity_rms':rms(preds['validation_boundary'][:,:3]-data['validation_boundary_target'][:,:3])/scales['U'],'normalized_initial_velocity_rms':rms(preds['validation_initial'][:,:3]-data['validation_initial_target'][:,:3])/scales['U'],'normalized_pressure_gauge_rms':rms(preds['validation_gauge'][:,3])/scales['P'],'velocity_relative_l2':float(np.linalg.norm(pred[:,:3]-target[:,:3])/np.linalg.norm(target[:,:3])),'pressure_relative_l2':float(np.linalg.norm(pred[:,3]-target[:,3])/np.linalg.norm(target[:,3]))}
  for key,value in metrics.items():assert abs(value-summary['final_metrics'][key])<=5e-10+5e-9*abs(value)
  passed=all(metrics[k]<=limit for k,limit in protocol['acceptance'].items())
  assert passed==summary['accepted'];row.update(accepted=passed,independent_metrics=metrics,timing_seconds=summary['timing_seconds'],closure_evaluations=summary['closure_evaluations']);rows.append(row)
 expected={(s,arm) for s in protocol['initialization_seeds'] for arm in protocol['arms']}
 seen=[(r['seed'],r['arm']) for r in rows]
 assert len(seen)==len(set(seen)) and set(seen)<=expected
 result={'state':'OFFBOX_NUMPY_COMPLETE_OBJECTIVE_AND_Q35_AUDIT_PASS','protocol_version':protocol['protocol_version'],'source_files_verified':len(source['files']),'raw_files_verified':len(manifest['files']),'comparisons':pairs,'numeric_entries':entries,'max_absolute_error':maximum,'rows':rows,'planned_cells':len(expected),'attempted_cells':len(rows),'not_run_cells':sorted([list(c) for c in expected-set(seen)]),'accepted_count':sum(r['accepted'] for r in rows),'formal_seeds':protocol['initialization_seeds'],'external_solver_results':False}
 by={(r['seed'],r['arm']):r for r in rows}
 result['paired_ratios']={}
 for arm in protocol['arms']:
  if arm=='c17_graph':continue
  ratios={}
  for s in protocol['initialization_seeds']:
   c=by.get((s,'c17_graph'));b=by.get((s,arm))
   if c and b and c['accepted'] and b['accepted']:
    ratios[str(s)]=b['timing_seconds']['solve_setup_plus_optimization']/c['timing_seconds']['solve_setup_plus_optimization']
  observed=list(ratios.values())
  result['paired_ratios'][arm]={'baseline_over_c17_setup_plus_optimization':ratios,'accepted_pairs':len(observed),
   'geometric_mean':float(np.exp(np.mean(np.log(observed)))) if observed else None,'range':[min(observed),max(observed)] if observed else None}
 result['supplementary']={}
 for arm in protocol['arms']:
  if arm=='c17_graph':continue
  opt,per=[],[]
  for s in protocol['initialization_seeds']:
   c=by.get((s,'c17_graph'));b=by.get((s,arm))
   if c and b and c['accepted'] and b['accepted']:
    oc,ob=c['timing_seconds']['optimization_including_stop_checks'],b['timing_seconds']['optimization_including_stop_checks']
    opt.append(ob/oc);per.append((ob/b['closure_evaluations'])/(oc/c['closure_evaluations']))
  gm=lambda v:float(np.exp(np.mean(np.log(v)))) if v else None
  result['supplementary'][arm]={'optimization_only_geometric_mean':gm(opt),'optimization_only_range':[min(opt),max(opt)] if opt else None,
   'per_evaluation_geometric_mean':gm(per),'per_evaluation_range':[min(per),max(per)] if per else None,'pairs':len(opt),
   'accepted':sum(1 for r in rows if r['arm']==arm and r['accepted']),'attempted':sum(1 for r in rows if r['arm']==arm)}
 assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
