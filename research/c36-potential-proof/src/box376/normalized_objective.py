"""Development objective with immutable, source-calibrated physical scales.

This is a distinct objective for later training. Existing raw-objective CPU/GPU
qualifications and profile packets are unchanged. Formal training is not frozen.
"""
import hashlib
import json

import torch

import torch_c17 as c17
from box376.capture_objective import PreparedObjective
from box376.data import COEFFICIENTS, FAMILIES, flatten
from box376.prepared_ad_objective import PreparedADObjective


def normalization_key(calibration):
    payload = {"calibration_raw_sha256":calibration["raw_sha256"],
               "source_nu":calibration["source_nu"],"source_eta":calibration["source_eta"],
               "scales":calibration["candidate_fixed_physical_scales"]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()


class Normalization:
    def prepare_normalization(self, data, calibration):
        if data["nu"] != calibration["source_nu"] or data["eta"] != calibration["source_eta"]:
            raise ValueError("Calibration belongs to a different physical source")
        self.normalization_key = normalization_key(calibration)
        if data["normalization_key"] != self.normalization_key:
            raise ValueError("Training normalization key mismatch")
        scales = calibration["candidate_fixed_physical_scales"]
        if any(not (0 < float(value) < float("inf")) for value in scales.values()):
            raise ValueError("Positive finite physical scales required")
        prototype = data["families"]["interior"]["points"]
        momentum, continuity = scales["momentum_residual"], scales["continuity_residual"]
        momentum_gradient, continuity_gradient = scales["momentum_residual_gradient"], scales["continuity_residual_gradient"]
        self.residual_inverse_square = prototype.new_tensor([1/momentum**2]*3+[1/continuity**2])
        self.gradient_inverse_square = prototype.new_tensor([[1/momentum_gradient**2]*3]*3+[[1/continuity_gradient**2]*3])
        self.velocity_inverse_square = prototype.new_tensor(1/scales["velocity"]**2)
        self.pressure_inverse_square = prototype.new_tensor(1/scales["pressure"]**2)
        self.constants += (self.residual_inverse_square,self.gradient_inverse_square,
                           self.velocity_inverse_square,self.pressure_inverse_square)
        self.constant_addresses = tuple(value.data_ptr() for value in self.constants)

    def guard(self, data):
        if data["normalization_key"] != self.normalization_key:
            raise ValueError("Prepared normalized objective belongs to another calibration")
        if data["metadata"]["cache_key"] != self.source_cache_key or data["metadata"]["source_version"] != self.source_version:
            raise ValueError("Prepared objective belongs to another source cache")
        if tuple(value.data_ptr() for value in self.constants) != self.constant_addresses:
            raise ValueError("Prepared constant storage changed")


class PreparedNormalizedObjective(Normalization, PreparedObjective):
    def __init__(self, data, calibration):
        super().__init__(data)
        self.prepare_normalization(data,calibration)

    def interior(self, state, force, force_gradient, weights, nu, lambda_g):
        residual, derivative = c17.observations(state,force,force_gradient,nu)
        loss = (weights*((residual*residual*self.residual_inverse_square).sum(1)
                         +lambda_g*(derivative*derivative*self.gradient_inverse_square).sum((1,2)))).sum()
        br = 2*weights[:,None]*residual*self.residual_inverse_square
        bg = 2*lambda_g*weights[:,None,None]*derivative*self.gradient_inverse_square
        seed = torch.zeros_like(state)
        value, gradient, hessian, _, _, _ = c17.unpack(state)

        def add_hessian(component,i,j,contribution):
            seed[:,component,4+c17.PAIRS.index((min(i,j),max(i,j)))] += contribution

        for i in range(3):
            seed[:,i,13] += br[:,i]
            seed[:,3,1+i] += br[:,i]
            seed[:,i,1+i] += br[:,3]
            for j in range(3):
                seed[:,j,0] += br[:,i]*gradient[:,i,j]
                seed[:,i,1+j] += br[:,i]*value[:,j]
                add_hessian(i,j,j,-nu*br[:,i])
            for k in range(3):
                cotangent = bg[:,i,k]
                seed[:,i,14+k] += cotangent
                seed[:,i,10+k] -= nu*cotangent
                add_hessian(3,i,k,cotangent)
                add_hessian(i,i,k,bg[:,3,k])
                for j in range(3):
                    seed[:,i,1+j] += cotangent*gradient[:,j,k]
                    seed[:,j,1+k] += cotangent*gradient[:,i,j]
                    seed[:,j,0] += cotangent*hessian[:,i,j,k]
                    add_hessian(i,j,k,cotangent*value[:,j])
        return loss,seed

    def __call__(self, parameters, data, chunk):
        if chunk < 1:
            raise ValueError("Positive chunk required")
        self.guard(data)
        numerators = parameters[0][0].new_zeros(len(FAMILIES))
        gradient = torch.zeros_like(flatten(parameters))
        for index,name in enumerate(FAMILIES):
            family = data["families"][name]
            for begin in range(0,len(family["points"]),chunk):
                end = min(begin+chunk,len(family["points"]))
                points,weights = family["points"][begin:end],family["weights"][begin:end]
                if name == "interior":
                    state,tape = self.forward(parameters,points)
                    numerator,seed = self.interior(state,family["force"][begin:end],family["force_gradient"][begin:end],
                                                   weights,data["nu"],data["lambda_g"])
                    gradients = c17.parameter_vjp(parameters,tape,seed*(COEFFICIENTS[index]/family["W"]))
                else:
                    indices = slice(3,4) if name == "gauge" else slice(0,3)
                    factor = self.pressure_inverse_square if name == "gauge" else self.velocity_inverse_square
                    numerator,gradients = self.value_gradient(parameters,points,family["target"][begin:end],indices,weights)
                    numerator *= factor
                    gradients = [(w*factor*(COEFFICIENTS[index]/family["W"]),b*factor*(COEFFICIENTS[index]/family["W"])) for w,b in gradients]
                numerators[index] += numerator
                gradient += flatten(gradients)
        return (numerators*self.coefficients/self.denominators).sum(),gradient,numerators


class PreparedNormalizedADObjective(Normalization, PreparedADObjective):
    def __init__(self, data, calibration, backend="a24"):
        super().__init__(data,backend)
        self.prepare_normalization(data,calibration)

    def independent_interior(self, state, force, force_gradient, weights, nu, lambda_g):
        # Direct scalar-component AD consumer, independent of C17 observations
        # and its explicit consumer transpose.
        pairs = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))

        def hessian(component,i,j):
            return state[:,component,4+pairs.index((min(i,j),max(i,j)))]

        terms = []
        for i in range(3):
            residual = state[:,i,13]+sum(state[:,j,0]*state[:,i,1+j] for j in range(3))
            residual = residual-nu*sum(hessian(i,j,j) for j in range(3))+state[:,3,1+i]-force[:,i]
            terms.append(residual*residual*self.residual_inverse_square[i])
            for k in range(3):
                derivative = state[:,i,14+k]+sum(state[:,j,1+k]*state[:,i,1+j]+state[:,j,0]*hessian(i,j,k) for j in range(3))
                derivative = derivative-nu*state[:,i,10+k]+hessian(3,i,k)-force_gradient[:,i,k]
                terms.append(lambda_g*derivative*derivative*self.gradient_inverse_square[i,k])
        divergence = sum(state[:,i,1+i] for i in range(3))
        terms.append(divergence*divergence*self.residual_inverse_square[3])
        for k in range(3):
            derivative = sum(hessian(i,i,k) for i in range(3))
            terms.append(lambda_g*derivative*derivative*self.gradient_inverse_square[3,k])
        return (weights*sum(terms)).sum()

    def __call__(self, parameters, data, chunk):
        if chunk < 1:
            raise ValueError("Positive chunk required")
        self.guard(data)
        p = [(w.detach().clone().requires_grad_(True),b.detach().clone().requires_grad_(True)) for w,b in parameters]
        leaves = [value for layer in p for value in layer]
        numerators = parameters[0][0].new_zeros(len(FAMILIES))
        gradient = torch.zeros_like(flatten(parameters))
        for index,name in enumerate(FAMILIES):
            family = data["families"][name]
            for begin in range(0,len(family["points"]),chunk):
                end = min(begin+chunk,len(family["points"]))
                points,weights = family["points"][begin:end],family["weights"][begin:end]
                if name == "interior":
                    state = self.contract(self.propagate(self.physical_parameters(p),points))
                    numerator = self.independent_interior(state,family["force"][begin:end],family["force_gradient"][begin:end],
                                                          weights,data["nu"],data["lambda_g"])
                else:
                    indices = slice(3,4) if name == "gauge" else slice(0,3)
                    factor = self.pressure_inverse_square if name == "gauge" else self.velocity_inverse_square
                    difference = self.values(p,points)[:,indices]-family["target"][begin:end,indices]
                    numerator = (weights*(difference*difference).sum(-1)*factor).sum()
                derivatives = torch.autograd.grad(numerator*(COEFFICIENTS[index]/family["W"]),leaves)
                gradient += torch.cat([value.reshape(-1) for value in derivatives])
                numerators[index] += numerator.detach()
        return (numerators*self.coefficients/self.denominators).sum(),gradient,numerators
