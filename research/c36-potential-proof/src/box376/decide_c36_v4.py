"""Registered decision of the C36 v4 combination study (pre-registered with its protocol).

    python -m box376.decide_c36_v4 --protocol PROTOCOL --summary observation-summary.json --output OUT.json

Reads the all-planes observation summary of the v4 checkpoints (box376.run_c36_observe)
and applies protocol['decision_rule'] mechanically. Runs are named seed{S}-resume-{start}.
Standard library only.
"""
import argparse
import hashlib
import json
from pathlib import Path

ARMS = ('baseline', 'C2')
PRESSURE = 'normalized_corrected_pressure_rms'


def decide(protocol, summary):
    rule = protocol['decision_rule']
    tags = protocol['physical_observation']['checkpoints_per_run']
    seeds = protocol['seeds']
    rows = {}
    for row in summary['rows']:
        if row.get('observation_state') != 'COMPLETE':
            continue
        for tag in row['tags']:
            run, name = tag.split(':')
            rows[(run, name)] = row
    runs = {}
    for arm in ARMS:
        for seed in seeds:
            run = f'seed{seed}-resume-{arm}'
            observed = {tag: rows[(run, tag)] for tag in tags if (run, tag) in rows}
            passing = [r['LBFGS_calls'] for r in observed.values() if r['physical_acceptance_pass']]
            runs[run] = {'arm': arm, 'seed': seed, 'observed_tags': sorted(observed, key=tags.index),
                         'extra_calls': {t: r['LBFGS_calls'] for t, r in observed.items()},
                         'acceptance': {t: r['physical_acceptance_pass'] for t, r in observed.items()},
                         'stratified_pressure': {t: r['datasets']['stratified']['global'][PRESSURE] for t, r in observed.items()},
                         'stratified_velocity': {t: r['datasets']['stratified']['global']['velocity_relative_l2'] for t, r in observed.items()},
                         'first_pass_extra_calls': min(passing) if passing else None,
                         'missing_tags': [t for t in tags if t not in observed]}
    by = {(r['arm'], r['seed']): r for r in runs.values()}

    def passes(arm):
        return [by[(arm, s)]['first_pass_extra_calls'] is not None for s in seeds]

    def stopped(arm):
        values = []
        for s in seeds:
            p = by[(arm, s)]['stratified_pressure']
            if 'final' not in p or 'LBFGS-call3200' not in p:
                return None
            values.append(p['final'] >= .95 * p['LBFGS-call3200'])
        return all(values)

    stopped_falling = {arm: stopped(arm) for arm in ARMS}
    base, c2 = passes('baseline'), passes('C2')
    if all(base) and all(c2):
        first = {s: (by[('C2', s)]['first_pass_extra_calls'], by[('baseline', s)]['first_pass_extra_calls']) for s in seeds}
        sub = ('C2_sooner' if all(a < b for a, b in first.values()) else
               'baseline_sooner' if all(a > b for a, b in first.values()) else 'same_or_mixed')
        outcome = {'key': 'O1_both_arms_pass', 'sub_result': sub}
    elif all(c2) and not all(base):
        outcome = {'key': 'O2_only_C2_passes'}
    elif all(base) and not all(c2):
        outcome = {'key': 'O3_only_baseline_passes'}
    elif any(base) or any(c2):
        outcome = {'key': 'O4_partial'}
    elif any(v is True for v in stopped_falling.values()):
        outcome = {'key': 'O6_none_stopped'}
    else:
        outcome = {'key': 'O5_none_still_falling',
                   'stopped_falling_evaluable': all(v is not None for v in stopped_falling.values())}
    outcome['text'] = rule['outcomes_in_order'][outcome['key']]
    either_both = all(base) or all(c2)
    secondary = {}
    for tag in tags:
        pairs = {}
        for s in seeds:
            c, b = by[('C2', s)]['stratified_pressure'].get(tag), by[('baseline', s)]['stratified_pressure'].get(tag)
            if c is not None and b is not None:
                pairs[str(s)] = {'C2': c, 'baseline': b, 'C2_over_baseline': c / b,
                                 'interpreted': abs(c / b - 1) >= .32}
        secondary[tag] = pairs
    return {'outcome': outcome, 'stopped_falling': stopped_falling, 'runs': runs,
            'secondary_stratified_pressure': secondary,
            'follow_up': ('freeze a formal protocol with new seeds' if either_both else 'no formal claim'),
            'formal_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    summary = json.loads(args.summary.read_text())
    result = {'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              'summary_sha256': hashlib.sha256(args.summary.read_bytes()).hexdigest(), **decide(protocol, summary)}
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['outcome']))


if __name__ == '__main__':
    main()
