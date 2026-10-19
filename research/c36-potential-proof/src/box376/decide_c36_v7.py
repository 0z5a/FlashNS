"""Registered decision of the C36 v7 optimizer study (pre-registered with its protocol).

    python -m box376.decide_c36_v7 --reader READER.json --protocol PROTOCOL --output OUT.json

Applies protocol['decision_rule'] to the in-batch reader (box376.read_c36_v7), with the v6
outcomes in the same order over the four registered runs:
- F1_pass: at least one run passes physical acceptance at an observed checkpoint;
- F2_near_miss: no run passes, and some run's complete final observation exceeds no acceptance
  limit by more than the registered factor (box376.decide_c36_v6.over_limit);
- F0_incomplete: no pass and no near miss, and some registered run lacks a complete final
  observation: no scientific reading;
- F3_fail: no pass and no near miss, with complete final observations of all four runs.
Registered secondary readouts, which do not change the outcome: for each fresh run, the first
evaluation at or below each loss threshold against the v5 analytic L-BFGS run of the same seed
(the protocol's constants); for each continue run, the final loss over its start loss.
Standard library only.
"""
import argparse
import hashlib
import json
from pathlib import Path

from box376.decide_c36_v6 import over_limit

PRESSURE = 'normalized_corrected_pressure_rms'


def decide(protocol, reader):
    rule = protocol['decision_rule']
    runs = {}
    for run in reader['runs']:
        key = f"{run['arm']}:{run['seed']}"
        observed = {tag: row for tag, row in run['physical'].items() if row.get('state') == 'COMPLETE'}
        entry = {'arm': run['arm'], 'seed': run['seed'], 'training_state': run['training_state'],
                 'training_gate_reached': run['training_gate_first'] is not None,
                 'observed_tags': sorted(observed),
                 'passing_tags': sorted(t for t, r in observed.items() if r['physical_acceptance_pass'])}
        if 'final' in observed:
            factor, worst, failing = over_limit(observed['final'], protocol)
            entry.update(final_worst_over_limit=factor, final_worst_test=worst, final_failing_tests=failing,
                         final_stratified_pressure_over_limit=observed['final']['datasets']['stratified']['global'][PRESSURE] / 1e-3)
        curve = run.get('loss_curve')
        if curve and run['arm'] == 'fresh':
            reference = rule['v5_analytic_first_evaluation_at_or_below'][str(run['seed'])]
            entry['first_evaluation_versus_v5'] = {
                threshold: {'v7': hit['evaluation'] if hit else None, 'v5': reference.get(threshold),
                            'v7_over_v5': (hit['evaluation'] / reference[threshold]) if hit and reference.get(threshold) else None}
                for threshold, hit in curve['first_evaluation_at_or_below'].items()}
        if curve and run['arm'] == 'continue' and run.get('continuity'):
            entry['final_over_start_loss'] = (run['final_loss'] / run['continuity']['start_loss']) if run.get('final_loss') else None
        runs[key] = entry
    passing = [k for k, r in runs.items() if r['passing_tags']]
    finals = [r['final_worst_over_limit'] for r in runs.values() if 'final_worst_over_limit' in r]
    missing = sorted(k for k in rule['runs'] if k not in runs or 'final_worst_over_limit' not in runs[k])
    if passing:
        outcome = {'key': 'F1_pass', 'runs': passing}
    elif finals and min(finals) <= rule['near_miss_factor']:
        outcome = {'key': 'F2_near_miss', 'best_final_worst_over_limit': min(finals)}
    elif missing:
        outcome = {'key': 'F0_incomplete', 'runs': missing}
    else:
        outcome = {'key': 'F3_fail', 'best_final_worst_over_limit': min(finals)}
    outcome['text'] = rule['outcomes_in_order'][outcome['key']]
    return {'outcome': outcome, 'runs': runs, 'runs_without_complete_final_observation': missing,
            'follow_up': rule['follow_up'][outcome['key']], 'formal_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    reader = json.loads(args.reader.read_text())
    assert reader['protocol_sha256'] == hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    result = {'protocol_sha256': reader['protocol_sha256'],
              'reader_sha256': hashlib.sha256(args.reader.read_bytes()).hexdigest(), **decide(protocol, reader)}
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps(result['outcome']))


if __name__ == '__main__':
    main()
