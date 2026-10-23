"""Registered decision of the C36 v5 budget study (pre-registered with its protocol).

    python -m box376.decide_c36_v5 --protocol PROTOCOL --summary observation-summary.json [...] --output OUT.json

Reads one or more all-planes observation summaries (box376.run_c36_observe) of the v5
checkpoints and applies protocol['decision_rule'] mechanically. Run names and tags come
from the protocol. Standard library only.
"""
import argparse
import hashlib
import json
from pathlib import Path

PRESSURE = 'normalized_corrected_pressure_rms'


def decide(protocol, summaries):
    rule = protocol['decision_rule']
    tags = ['LBFGS-call800'] + list(protocol['physical_observation']['checkpoints_per_run'])
    rows = {}
    for summary in summaries:
        for row in summary['rows']:
            if row.get('observation_state') != 'COMPLETE':
                continue
            for tag in row['tags']:
                run, name = tag.split(':')
                rows[(run, name)] = row
    runs, by = {}, {}
    for key, run in rule['runs'].items():
        engine, seed = key.split(':')
        observed = {tag: rows[(run, tag)] for tag in tags if (run, tag) in rows}
        passing = [r['LBFGS_calls'] for r in observed.values() if r['physical_acceptance_pass']]
        runs[run] = by[(engine, int(seed))] = {
            'engine': engine, 'seed': int(seed), 'observed_tags': [t for t in tags if t in observed],
            'LBFGS_calls': {t: r['LBFGS_calls'] for t, r in observed.items()},
            'acceptance': {t: r['physical_acceptance_pass'] for t, r in observed.items()},
            'stratified_pressure': {t: r['datasets']['stratified']['global'][PRESSURE] for t, r in observed.items()},
            'stratified_velocity': {t: r['datasets']['stratified']['global']['velocity_relative_l2'] for t, r in observed.items()},
            'first_pass_LBFGS_calls': min(passing) if passing else None,
            'missing_tags': [t for t in tags[1:] if t not in observed]}
    engines = sorted({e for e, _ in by})
    seeds = sorted({s for _, s in by})

    def passes(engine):
        return [by[(engine, s)]['first_pass_LBFGS_calls'] is not None for s in seeds]

    def stopped(engine):
        values = []
        for s in seeds:
            p = by[(engine, s)]['stratified_pressure']
            if 'final' not in p or 'LBFGS-call4000' not in p:
                return None
            values.append(p['final'] >= .95 * p['LBFGS-call4000'])
        return all(values)

    stopped_falling = {e: stopped(e) for e in engines}
    passing_engines = [e for e in engines if all(passes(e))]
    if len(passing_engines) == len(engines):
        outcome = {'key': 'E1_all_pass'}
    elif len(passing_engines) == 1:
        outcome = {'key': 'E2_one_engine_passes', 'engine': passing_engines[0]}
    elif any(any(passes(e)) for e in engines):
        outcome = {'key': 'E3_partial'}
    elif any(v is True for v in stopped_falling.values()):
        outcome = {'key': 'E5_none_stopped'}
    else:
        outcome = {'key': 'E4_none_still_falling',
                   'stopped_falling_evaluable': all(v is not None for v in stopped_falling.values())}
    outcome['text'] = rule['outcomes_in_order'][outcome['key']]
    return {'outcome': outcome, 'stopped_falling': stopped_falling, 'runs': runs,
            'follow_up': ('freeze a formal protocol with new seeds' if outcome['key'] in ('E1_all_pass', 'E2_one_engine_passes')
                          else 'no formal claim'),
            'formal_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--summary', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    summaries = [json.loads(p.read_text()) for p in args.summary]
    result = {'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              'summary_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in args.summary],
              **decide(protocol, summaries)}
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['outcome']))


if __name__ == '__main__':
    main()
