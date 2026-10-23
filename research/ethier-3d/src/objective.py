import torch
import torch_c17 as c17
from torch_references import a24_forward,a24_contract,physical_loss

def values(parameters,x):
 for k,(W,b) in enumerate(parameters):
  x=x@W.T+b
  if k<len(parameters)-1:x=torch.tanh(x)
 return x

def value_gradient(parameters,x,target,indices,coefficient):
 tape=[];v=x
 for k,(W,b) in enumerate(parameters):
  z=v@W.T+b;tape.append((v,z));v=torch.tanh(z) if k<len(parameters)-1 else z
 delta=v[:,indices]-target[:,indices];loss=coefficient*(delta*delta).sum()/len(x);seed=torch.zeros_like(v);seed[:,indices]=2*coefficient*delta/len(x);grads=[]
 for k in range(len(parameters)-1,-1,-1):
  old,z=tape[k];W,_=parameters[k]
  if k<len(parameters)-1:seed=seed*c17.coefficients(z)[1]
  grads.append((seed.T@old,seed.sum(0)));seed=seed@W
 return loss,list(reversed(grads))

def explicit_objective(parameters,data,protocol):
 x=data['interior'];f=x.new_zeros((len(x),3));df=x.new_zeros((len(x),3,3));w=x.new_full((len(x),),1/len(x))
 loss,grads=c17.loss_gradient(parameters,x,f,df,w,nu=protocol['parameters']['nu'],lambda_g=protocol['lambda_g'])
 for key,idx,coef in [('boundary',slice(0,3),protocol['boundary_weight']),('initial',slice(0,3),protocol['initial_weight']),('gauge',slice(3,4),protocol['gauge_weight'])]:
  l,g=value_gradient(parameters,data[key],data[key+'_target'],idx,coef);loss=loss+l;grads=[(W+dW,b+db) for (W,b),(dW,db) in zip(grads,g)]
 return loss,grads

def reference_objective(parameters,data,protocol,scalar_ad=False,naive=False):
 from torch_references import scalar_ad_state
 p=[(W.detach().clone().requires_grad_(True),b.detach().clone().requires_grad_(True)) for W,b in parameters];x=data['interior']
 if naive:
  from naive_q35 import forward,contract
  state=contract(forward(p,x))
 else:state=scalar_ad_state(p,x) if scalar_ad else a24_contract(a24_forward(p,x))
 f=x.new_zeros((len(x),3));df=x.new_zeros((len(x),3,3));w=x.new_full((len(x),),1/len(x));loss=physical_loss(state,f,df,w,nu=protocol['parameters']['nu'],lambda_g=protocol['lambda_g'])
 for key,idx,coef in [('boundary',slice(0,3),protocol['boundary_weight']),('initial',slice(0,3),protocol['initial_weight']),('gauge',slice(3,4),protocol['gauge_weight'])]:
  diff=values(p,data[key])[:,idx]-data[key+'_target'][:,idx];loss=loss+coef*(diff*diff).sum()/len(diff)
 g=torch.autograd.grad(loss,[v for layer in p for v in layer]);return loss.detach(),[(g[i],g[i+1]) for i in range(0,len(g),2)]

def independent_observations(s,nu):
 hp=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
 def H(c,i,j):return s[:,c,4+hp.index((min(i,j),max(i,j)))]
 r=[];g=[]
 for i in range(3):
  r.append(s[:,i,13]+sum(s[:,j,0]*s[:,i,1+j] for j in range(3))-nu*sum(H(i,j,j) for j in range(3))+s[:,3,1+i])
  g.append(torch.stack([s[:,i,14+k]+sum(s[:,j,1+k]*s[:,i,1+j]+s[:,j,0]*H(i,j,k) for j in range(3))-nu*s[:,i,10+k]+H(3,i,k) for k in range(3)],-1))
 r.append(sum(s[:,i,1+i] for i in range(3)))
 g.append(torch.stack([sum(H(i,i,k) for i in range(3)) for k in range(3)],-1))
 return torch.stack(r,-1),torch.stack(g,1)

def metrics(parameters,data,protocol,independent=False):
 from torch_references import scalar_ad_state
 x=data['validation'];s=scalar_ad_state(parameters,x) if independent else c17.forward(parameters,x)[0];f=x.new_zeros((len(x),3));df=x.new_zeros((len(x),3,3))
 R,RG=independent_observations(s,protocol['parameters']['nu']) if independent else c17.observations(s,f,df,protocol['parameters']['nu'])
 pred=values(parameters,x);target=data['validation_target'];scale=protocol['scales']
 def rms(x):return (x*x).mean().sqrt()
 result={'normalized_momentum_rms':rms(R[:,:3])/scale['R'],'normalized_divergence_rms':rms(R[:,3])*scale['L']/scale['U'],'normalized_spatial_residual_gradient_rms':rms(RG)*scale['L']/scale['R'],'normalized_boundary_velocity_rms':rms(values(parameters,data['validation_boundary'])[:,:3]-data['validation_boundary_target'][:,:3])/scale['U'],'normalized_initial_velocity_rms':rms(values(parameters,data['validation_initial'])[:,:3]-data['validation_initial_target'][:,:3])/scale['U'],'normalized_pressure_gauge_rms':rms(values(parameters,data['validation_gauge'])[:,3])/scale['P'],'velocity_relative_l2':torch.linalg.vector_norm(pred[:,:3]-target[:,:3])/torch.linalg.vector_norm(target[:,:3]),'pressure_relative_l2':torch.linalg.vector_norm(pred[:,3]-target[:,3])/torch.linalg.vector_norm(target[:,3])}
 return {k:float(v.detach().cpu()) for k,v in result.items()}

def accepted(result,protocol):return all(result[k]<=v for k,v in protocol['acceptance'].items())
