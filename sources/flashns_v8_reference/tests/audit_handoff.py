#!/usr/bin/env python3
"""Recompute statistics from uploaded historical JSON; does NOT rerun a solver."""
import argparse,hashlib,json,statistics
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('handoff_root',type=Path)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'reports/handoff_audit.json')
    a=ap.parse_args();h=a.handoff_root
    sp=h/'flashns/experiments/pinn_solver/artifacts/extension1/suite.json'
    s=json.loads(sp.read_text());k=json.loads((h/'02_KEY_RESULTS.json').read_text())
    rows=s['runs'];groups={}
    for r in rows:groups.setdefault(r['backend'],{})[r['seed']]=r
    stats={};maxdiff=0
    for name,g in groups.items():
        ratios=[r['result']['total_wall_seconds']/groups['HopperTMA'][seed]['result']['total_wall_seconds'] for seed,r in sorted(g.items())]
        val=statistics.median(ratios)
        if name in k['paired_baselines']:
            err=abs(val-k['paired_baselines'][name]['paired_speedup_median']);maxdiff=max(maxdiff,err);assert err<1e-12
        stats[name]={'total_wall_median_s':statistics.median(r['result']['total_wall_seconds'] for r in g.values()),
                     'optimization_wall_median_s':statistics.median(r['result']['optimization_wall_seconds'] for r in g.values()),
                     'paired_total_over_tma_median':val,'paired_total_over_tma_by_seed':dict(zip(map(str,sorted(g)),ratios))}
    files=['00_README_FOR_GPT_PRO.md','01_ITERATION_RFC_V7_ZH.md','02_KEY_RESULTS.json',
           'flashns/docs/pinn-solver-results.md','flashns/docs/cuda-hopper-results.md','flashns/docs/openai-ns-formal-results.md',
           'flashns/experiments/pinn_solver/problem.py','flashns/experiments/pinn_solver/backends.py',
           'flashns/experiments/cuda_jet_h100/common.py','flashns/experiments/cuda_jet_h100/stable_jet.cuh',
           'flashns/src/flashns/jet_stable.py','flashns/experiments/pinn_solver/artifacts/extension1/protocol.json',
           'flashns/experiments/pinn_solver/artifacts/extension1/suite.json']
    sourcehash={f:sha(h/f) for f in files}
    report={'scope':'recomputation of historical uploaded JSON, not independent GPU measurement',
            'final_combinations':len(rows),'final_converged':sum(r['result']['converged'] for r in rows),
            'initial_converged':sum(r['original_converged'] for r in rows),
            'actual_attempts':len(rows)+sum(not r['reused_converged_run'] for r in rows),
            'all_attempts_process_wall_s':sum(r['all_attempts_process_wall_seconds'] for r in rows),
            'max_summary_ratio_discrepancy':maxdiff,'statistics':stats,'source_sha256':sourcehash,
            'new_gpu_measurement':False,'new_lean_replay':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('final_combinations','final_converged','initial_converged','actual_attempts','max_summary_ratio_discrepancy')},indent=2))
if __name__=='__main__':main()
