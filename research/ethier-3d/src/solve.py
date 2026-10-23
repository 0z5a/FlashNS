"""Finite complete PINN solve. Scientific nonconvergence is retained, exit0.

Exceptions exit1; no retries, timeout, altered seeds, or relaxed acceptance.
The development CPU smoke is explicitly excluded from GPU/scientific results.
"""
import time
WORKER_START=time.perf_counter()
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from objective import explicit_objective,reference_objective,metrics,accepted,values,independent_observations
from torch_references import scalar_ad_state


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_hash(a):return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()
def flatten(grads):return [v for pair in grads for v in pair]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['c17_graph','a24_graph','naive_q35_graph'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--expected-gpu-uuid')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-smoke',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    protocol_path=root/'protocol-v4.json';data_path=root/'data-v2.npz'
    protocol=json.loads(protocol_path.read_text())
    assert args.seed in protocol['initialization_seeds'] and args.arm in protocol['arms']
    assert digest(data_path)==protocol['data_file_sha256']
    assert not args.output.exists();args.output.mkdir(parents=True)
    summary={'state':'STARTED','arm':args.arm,'role':protocol['arm_roles'][args.arm],
             'seed':args.seed,'protocol_sha256':digest(protocol_path),'data_sha256':digest(data_path),
             'cpu_smoke_only':args.cpu_smoke,'torch':torch.__version__,'numpy':np.__version__,
             'started_unix':time.time(),'qualification':[],'timing_seconds':{},'closure_evaluations':0}
    def save():
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    save()
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        device=torch.device(args.device)
        assert args.cpu_smoke == (device.type=='cpu')
        if device.type=='cuda':
            assert args.expected_gpu_uuid
            # Multi-GPU node: the driver exposes exactly one GPU by UUID.
            assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.expected_gpu_uuid and torch.cuda.device_count()==1
            uuid=subprocess.check_output(['nvidia-smi','-i',args.expected_gpu_uuid,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            assert uuid==args.expected_gpu_uuid
            summary.update(gpu_uuid=uuid,gpu_name=torch.cuda.get_device_name(0),tf32=False)
        raw=np.load(data_path)
        for key in raw.files:assert array_hash(raw[key])==protocol['data_array_sha256'][key]
        data={k:torch.as_tensor(raw[k],device=device).clone() for k in raw.files if not k[0].isdigit()}
        initial=[torch.as_tensor(raw[f'{args.seed}_{tag}{i}'],device=device).clone()
                 for i in range(3) for tag in ['W','b']]
        flat=[torch.nn.Parameter(v.clone()) for v in initial]
        parameters=[(flat[i],flat[i+1]) for i in range(0,len(flat),2)]
        initial_hash=hashlib.sha256(np.concatenate([v.detach().cpu().numpy().ravel() for v in flat]).tobytes()).hexdigest()
        assert initial_hash==protocol['initial_parameter_sha256'][str(args.seed)]
        summary['initial_parameter_sha256']=initial_hash

        def sync():
            if device.type=='cuda':torch.cuda.synchronize()

        def evaluate(p,d):
            if args.arm=='c17_graph':
                with torch.no_grad():return explicit_objective(p,d,protocol)
            return reference_objective(p,d,protocol,naive=args.arm=='naive_q35_graph')

        qualification_raw={}
        def compare(left,right,label):
            maximum=0.;ratio=0.;entries=0
            for i,(a,b) in enumerate(zip(left,right)):
                aa=a.detach().cpu().numpy().copy();bb=b.detach().cpu().numpy().copy()
                budget=2e-12+2e-11*np.abs(bb)
                assert np.all(np.isfinite(aa)) and np.all(np.isfinite(bb))
                err=np.abs(aa-bb)
                assert np.all(err<=budget),(label,i,float(np.max(err)),float(np.max(err/budget)))
                qualification_raw[f'{label}_{i}_actual']=aa
                qualification_raw[f'{label}_{i}_reference']=bb
                maximum=max(maximum,float(np.max(err)));ratio=max(ratio,float(np.max(err/budget)));entries+=aa.size
            summary['qualification'].append({'label':label,'entries':entries,'max_absolute_error':maximum,'max_error_over_budget':ratio})

        tq=time.perf_counter()
        # New complete objective (IC/BC/gauge included), all4740 derivatives.
        small={k:v[:7] for k,v in data.items()}
        private=[(W.detach().clone(),b.detach().clone()) for W,b in parameters]
        for step in range(3):
            l,g=evaluate(private,small)
            rl,rg=reference_objective(private,small,protocol,scalar_ad=True)
            compare([l]+flatten(g),[rl]+flatten(rg),f'scalar_ad_update{step}')
            for i,(W,b) in enumerate(private):
                qualification_raw[f'scalar_ad_update{step}_W{i}']=W.cpu().numpy().copy()
                qualification_raw[f'scalar_ad_update{step}_b{i}']=b.cpu().numpy().copy()
            private=[(W-1e-4*dW,b-1e-4*db) for (W,b),(dW,db) in zip(private,g)]
        sync();summary['timing_seconds']['qualification_small']=time.perf_counter()-tq

        ts=time.perf_counter()
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=evaluate(parameters,data)
            torch.cuda.current_stream().wait_stream(stream);sync()
            del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):captured_loss,captured_grads=evaluate(parameters,data)
            sync()
            def replay():
                graph.replay()
                return captured_loss,captured_grads
            summary['capture_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            summary['capture_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        else:
            def replay():return evaluate(parameters,data)
        sync();summary['timing_seconds']['setup_capture']=time.perf_counter()-ts

        tq=time.perf_counter()
        # Verify full-data replay consumes new parameter storage on three updates.
        for step in range(3):
            l,g=replay()
            rl,rg=reference_objective(parameters,data,protocol)
            compare([l]+flatten(g),[rl]+flatten(rg),f'full_replay_update{step}')
            for i,(W,b) in enumerate(parameters):
                qualification_raw[f'full_replay_update{step}_W{i}']=W.detach().cpu().numpy().copy()
                qualification_raw[f'full_replay_update{step}_b{i}']=b.detach().cpu().numpy().copy()
            with torch.no_grad():
                for p,delta in zip(flat,flatten(g)):p.add_(delta,alpha=-1e-4)
        with torch.no_grad():
            for p,v in zip(flat,initial):p.copy_(v)
        sync();summary['timing_seconds']['qualification_full_replay']=time.perf_counter()-tq
        np.savez_compressed(args.output/'qualification.npz',**qualification_raw)
        summary['qualification_sha256']=digest(args.output/'qualification.npz')
        summary['qualification_state']='COMPLETE_OBJECTIVE_ALL_PARAMETER_AND_GRAPH_UPDATES_PASS'
        initial_metrics=metrics(parameters,data,protocol,independent=True)
        assert not accepted(initial_metrics,protocol),'Step0 already accepted; not a training solve'
        summary['initial_metrics']=initial_metrics
        save()

        opt=protocol['optimizer'];evals=0;stage='adam';accepted_at=None
        budget_adam=3 if args.cpu_smoke else opt['adam_steps']
        budget_lbfgs=1 if args.cpu_smoke else opt['lbfgs_blocks']
        eval_bound=budget_adam+budget_lbfgs*(1+(2 if args.cpu_smoke else opt['lbfgs_max_iter_per_block'])*26)
        optimization_start=time.perf_counter()
        def closure():
            nonlocal evals
            assert evals<eval_bound,'Frozen algorithmic closure bound exhausted'
            loss,grads=replay();evals+=1
            for p,g in zip(flat,flatten(grads)):p.grad=g.detach()
            # CUDA graph loss/grads are borrowed buffers. LBFGS retains losses;
            # return an owned scalar so the next replay cannot overwrite it.
            return loss.detach().clone()

        def check(iteration):
            nonlocal accepted_at
            with torch.no_grad():m=metrics(parameters,data,protocol)
            assert all(np.isfinite(v) for v in m.values()),'Nonfinite validation'
            entry={'stage':stage,'iteration':iteration,'closure_evaluations':evals,
                   'optimization_seconds':time.perf_counter()-optimization_start,'metrics':m}
            if accepted(m,protocol):
                independent=metrics(parameters,data,protocol,independent=True)
                entry['independent_metrics']=independent
                entry['independent_accepted']=accepted(independent,protocol)
                if entry['independent_accepted']:accepted_at=entry
            with (args.output/'validation-history.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
            print(json.dumps(entry,allow_nan=False),flush=True)
            summary.update(state='OPTIMIZING',closure_evaluations=evals,last_validation=entry)
            save()
            return accepted_at is not None

        adam=torch.optim.Adam(flat,lr=opt['adam_lr'],foreach=False,fused=False)
        for step in range(1,budget_adam+1):
            adam.step(closure)
            if step%protocol['validation_every']['adam_steps']==0 or step==budget_adam:
                if check(step):break
        lbfgs_iterations=0
        if accepted_at is None:
            stage='lbfgs'
            lbfgs=torch.optim.LBFGS(flat,lr=opt['lbfgs_lr'],max_iter=2 if args.cpu_smoke else opt['lbfgs_max_iter_per_block'],
                max_eval=4 if args.cpu_smoke else opt['lbfgs_max_eval_per_block'],history_size=opt['lbfgs_history'],
                tolerance_grad=opt['tolerance_grad'],tolerance_change=opt['tolerance_change'],line_search_fn=opt['line_search'])
            for block in range(1,budget_lbfgs+1):
                lbfgs.step(closure)
                if block%protocol['validation_every']['lbfgs_blocks']==0 or block==budget_lbfgs:
                    if check(block):break
            lbfgs_iterations=int(lbfgs.state[flat[0]].get('n_iter',0))
        sync();summary['timing_seconds']['optimization_including_stop_checks']=time.perf_counter()-optimization_start
        summary.update(closure_evaluations=evals,adam_steps=step,lbfgs_iterations=lbfgs_iterations,accepted_at=accepted_at)

        tf=time.perf_counter()
        final_metrics=metrics(parameters,data,protocol,independent=True)
        final_accepted=accepted(final_metrics,protocol)
        # Preserve final independent physical state and every held-out observable.
        s=scalar_ad_state(parameters,data['validation']);R,RG=independent_observations(s,protocol['parameters']['nu'])
        outputs={'validation_state':s.detach().cpu().numpy(),'residual':R.detach().cpu().numpy(),
                 'residual_gradient':RG.detach().cpu().numpy()}
        for key in ['validation','validation_boundary','validation_initial','validation_gauge']:
            outputs[key+'_prediction']=values(parameters,data[key]).detach().cpu().numpy()
        for i,(W,b) in enumerate(parameters):
            outputs[f'final_W{i}']=W.detach().cpu().numpy();outputs[f'final_b{i}']=b.detach().cpu().numpy()
        np.savez_compressed(args.output/'final-raw.npz',**outputs)
        sync();summary['timing_seconds']['final_independent_acceptance']=time.perf_counter()-tf
        summary.update(state='CPU_SMOKE_COMPLETE' if args.cpu_smoke else ('ACCEPTED' if final_accepted else 'NONCONVERGED'),
            accepted=final_accepted and not args.cpu_smoke,final_metrics=final_metrics,
            final_raw_sha256=digest(args.output/'final-raw.npz'),actual_exit_code=0,finished_unix=time.time())
        summary['timing_seconds']['total_worker']=time.perf_counter()-WORKER_START
        summary['timing_seconds']['solve_setup_plus_optimization']=summary['timing_seconds']['setup_capture']+summary['timing_seconds']['optimization_including_stop_checks']
        summary['performance_claim_eligible']=summary['accepted'] and accepted_at is not None
        save();print(json.dumps(summary,allow_nan=False),flush=True)
        return 0
    except BaseException as exc:
        summary.update(state='RUNTIME_FAILURE',actual_exit_code=1,error=repr(exc),finished_unix=time.time())
        summary['timing_seconds']['total_worker']=time.perf_counter()-WORKER_START
        save();traceback.print_exc();return 1

if __name__=='__main__':sys.exit(main())
