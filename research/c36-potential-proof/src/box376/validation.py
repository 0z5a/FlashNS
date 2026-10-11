"""Independent physical observations and development acceptance diagnostics.

Taylor-state generation uses the wider A24 reference, not C17 propagation.
The NumPy residual consumer can also accept independently observed fields.
Zero reference pressure and inactive regions use fixed absolute scales.
"""
import numpy as np
import torch

from box376.objective import physical_parameters, values
from torch_references import a24_forward, a24_contract

HP = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
LIMITS = {"velocity_relative_l2":1e-2,"normalized_divergence_rms":1e-3,
          "normalized_momentum_rms":1e-2,"normalized_momentum_gradient_rms":2e-2,
          "normalized_divergence_gradient_rms":2e-2,
          "normalized_corrected_pressure_rms":1e-3,"normalized_pressure_gauge_rms":1e-3}


def finite_array(value, shape, name):
    value = np.asarray(value)
    if value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Finite FP64 {name} with shape {shape} required")
    return value


def fields_from_state(state):
    state = np.asarray(state)
    n = len(state)
    finite_array(state,(n,4,17),"state")
    H = np.empty((n,4,3,3),dtype=np.float64)
    for index,(i,j) in enumerate(HP):
        H[:,:,i,j] = H[:,:,j,i] = state[:,:,4+index]
    return {"U":state[:,:3,0],"G":state[:,:3,1:4],"H":H[:,:3],
            "Ut":state[:,:3,13],"GUt":state[:,:3,14:17],
            "lap":np.trace(H[:,:3],axis1=2,axis2=3),"gradlap":state[:,:3,10:13],
            "p":state[:,3,0],"grad_p":state[:,3,1:4],"H_p":H[:,3]}


def observations(fields, reference, nu):
    n = len(fields["U"])
    shapes = {"U":(n,3),"G":(n,3,3),"H":(n,3,3,3),"Ut":(n,3),
              "GUt":(n,3,3),"lap":(n,3),"gradlap":(n,3,3),
              "grad_p":(n,3),"H_p":(n,3,3),"p":(n,)}
    for name,shape in shapes.items():finite_array(fields[name],shape,name)
    force = finite_array(reference["f"],(n,3),"force")
    derivative = finite_array(reference["df"],(n,3,3),"force gradient")
    if not np.isfinite(nu) or nu <= 0:raise ValueError("Positive finite viscosity required")
    u,g,H = fields["U"],fields["G"],fields["H"]
    residual = fields["Ut"] + np.einsum("nj,nij->ni",u,g) - nu*fields["lap"] + fields["grad_p"] - force
    spatial = fields["GUt"] + np.einsum("nij,njk->nik",g,g) + np.einsum("nj,nijk->nik",u,H) - nu*fields["gradlap"] + fields["H_p"] - derivative
    divergence = np.trace(g,axis1=1,axis2=2)
    divergence_gradient = sum(H[:,i,i,:] for i in range(3))
    return residual,spatial,divergence,divergence_gradient


def weighted_rms(value,weights):
    value = np.asarray(value,dtype=np.float64)
    weights = finite_array(weights,(len(value),),"weights")
    if len(value)==0 or np.any(weights<0) or not weights.sum()>0 or not np.isfinite(value).all():
        raise ValueError("Nonempty finite observations with positive total weight required")
    return float(np.sqrt(np.sum(weights*np.mean(value.reshape(len(value),-1)**2,axis=1))/weights.sum()))


def field_metrics(fields,reference,weights,gauge_at_same_times,scales,nu=.01):
    if any(not np.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError("Fixed positive finite physical scales required")
    n = len(fields["U"])
    target = finite_array(reference["U"],(n,3),"reference velocity")
    gauge = finite_array(gauge_at_same_times,(n,),"pressure gauge at matching times")
    residual,spatial,divergence,dg = observations(fields,reference,nu)
    delta = fields["U"]-target
    denominator = weighted_rms(target,weights)
    result = {"velocity_relative_l2":weighted_rms(delta,weights)/denominator if denominator>0 else None,
              "velocity_reference_is_zero":denominator==0,
              "normalized_velocity_error_rms":weighted_rms(delta,weights)/scales["velocity"],
              "normalized_divergence_rms":weighted_rms(divergence,weights)/scales["continuity_residual"],
              "normalized_divergence_gradient_rms":weighted_rms(dg,weights)/scales["continuity_residual_gradient"],
              "normalized_momentum_rms":weighted_rms(residual,weights)/scales["momentum_residual"],
              "normalized_momentum_gradient_rms":weighted_rms(spatial,weights)/scales["momentum_residual_gradient"],
              "normalized_corrected_pressure_rms":weighted_rms(fields["p"]-gauge,weights)/scales["pressure"],
              "normalized_pressure_gauge_rms":weighted_rms(gauge,weights)/scales["pressure"],
              "velocity_component_error_rms":[weighted_rms(delta[:,i],weights) for i in range(3)],
              "velocity_component_relative_l2":[weighted_rms(delta[:,i],weights)/weighted_rms(target[:,i],weights) if weighted_rms(target[:,i],weights)>0 else None for i in range(3)]}
    return result,{"residual":residual,"spatial_residual_gradient":spatial,"divergence":divergence,"divergence_gradient":dg}


def field_passes_proposed_gate(metrics):
    for name,limit in LIMITS.items():
        value = metrics.get(name)
        if name=="velocity_relative_l2" and metrics.get("velocity_reference_is_zero") is True:
            value = metrics.get("normalized_velocity_error_rms")
            limit = 1e-3
        if not isinstance(value,(int,float)) or not np.isfinite(value) or value>limit or value<0:
            return False
    return True


def a24_state(parameters,points,chunk=128):
    if chunk<1:raise ValueError("Positive chunk required")
    points = finite_array(points,(len(points),4),"physical points")
    physical = physical_parameters(parameters)
    with torch.no_grad():
        states = [a24_contract(a24_forward(physical,torch.from_numpy(points[begin:begin+chunk].copy()))).numpy()
                  for begin in range(0,len(points),chunk)]
    return np.concatenate(states,axis=0)


def network_values(parameters,points):
    points = finite_array(points,(len(points),4),"physical value points")
    with torch.no_grad():return values(parameters,torch.from_numpy(points.copy())).numpy()


def network_dataset_metrics(parameters,dataset,scales,nu=.01,chunk=128):
    points,weights = dataset["points"],dataset["weights"]
    state = a24_state(parameters,points,chunk)
    fields = fields_from_state(state)
    gauge_points = np.c_[np.full((len(points),3),1/16),points[:,3]]
    gauge = network_values(parameters,gauge_points)[:,3]
    metrics,observed = field_metrics(fields,dataset["reference"],weights,gauge,scales,nu)
    regions = {}
    if "regions" in dataset:
        for region in np.unique(dataset["regions"]):
            selected = dataset["regions"]==region
            regional,_ = field_metrics({k:v[selected] for k,v in fields.items()},
                                      {k:v[selected] for k,v in dataset["reference"].items()},
                                      weights[selected],gauge[selected],scales,nu)
            regions[str(int(region))] = regional
    return ({"global":metrics,"regions":regions,"proposed_field_gate_pass":field_passes_proposed_gate(metrics)},
            {"state":state,"gauge":gauge,**observed})


def integrate_network_trajectories(parameters,initial):
    from scipy.integrate import solve_ivp
    from box376.source import trajectory
    initial = finite_array(initial,(len(initial),3),"initial particles")
    times = np.unique(np.r_[np.linspace(0,1,129),.25,.375,.5,.625,.75])
    exact = trajectory(times[:,None],initial[None,:,:])
    raw,records = {"initial":initial,"times":times,"exact":exact},[]
    previous = None

    def rhs(time,flat):
        points = np.c_[flat.reshape(-1,3),np.full(len(initial),time)]
        return network_values(parameters,points)[:,:3].reshape(-1)

    for index,(rtol,atol,max_step) in enumerate(((1e-9,1e-11,1/64),(1e-11,1e-13,1/128))):
        solution = solve_ivp(rhs,(0,1),initial.reshape(-1),method="DOP853",rtol=rtol,atol=atol,
                             max_step=max_step,dense_output=True)
        if not solution.success:raise RuntimeError(solution.message)
        actual = solution.sol(times).T.reshape(len(times),len(initial),3)
        if not np.isfinite(actual).all():raise ValueError("Nonfinite integrated path")
        error = np.linalg.norm(actual-exact,axis=-1)
        record = {"rtol":rtol,"atol":atol,"max_step":max_step,"nfev":solution.nfev,"accepted_steps":len(solution.t)-1,
                  "maximum_path_error":float(error.max()),"maximum_endpoint_error":float(error[-1].max())}
        if previous is not None:
            record["maximum_refinement_change"] = float(np.linalg.norm(actual-previous,axis=-1).max())
        records.append(record)
        raw[f"integrated-{index}"] = actual
        previous = actual
    refinement = records[-1]["maximum_refinement_change"]
    return ({"particles":len(initial),"output_times":len(times),"records":records,
             "analytic_path_used_in_rhs":False,"velocity_is_the_observed_network":True,
             "integrator_refinement_limit":1e-7,"proposed_endpoint_limit":1e-3,
             "proposed_trajectory_gate_pass":refinement<=1e-7 and records[-1]["maximum_endpoint_error"]<=1e-3},raw)
