"""Post-hoc phase split of the Ethier external-baseline batch (not part of the registered analysis).

    python3 posthoc_ext_phase_split.py BATCH_ROOT OUT.json

For every solve, from its validation history: time per closure evaluation in the Adam phase (to the
last Adam stop check) and in the L-BFGS phase (from there to the last check). Both include the
optimizer step and the stop checks of that phase. Then, per external arm, paired geometric means
over seeds of baseline/C17 for each phase. Standard library only; writes no absolute paths.
"""
import json
import math
import sys
from pathlib import Path


def split(directory):
    rows = [json.loads(line) for line in (directory / 'validation-history.jsonl').read_text().splitlines() if line.strip()]
    adam = [r for r in rows if r['stage'] == 'adam']
    out = {'checks': len(rows)}
    if adam:
        a, last = adam[-1], rows[-1]
        out['adam_evaluations'] = a['closure_evaluations']
        out['adam_ms_per_evaluation'] = 1000 * a['optimization_seconds'] / a['closure_evaluations']
        if last['closure_evaluations'] > a['closure_evaluations']:
            out['lbfgs_evaluations'] = last['closure_evaluations'] - a['closure_evaluations']
            out['lbfgs_ms_per_evaluation'] = 1000 * (last['optimization_seconds'] - a['optimization_seconds']) / out['lbfgs_evaluations']
    return out


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    audit = json.loads((root / 'audit-ext.json').read_text())
    solves = {}
    for row in audit['rows']:
        entry = {'arm': row['arm'], 'seed': row['seed'], 'accepted': row['accepted'], **split(root / 'evidence' / row['id'])}
        solves[row['id']] = entry
    by = {(v['seed'], v['arm']): v for v in solves.values()}
    arms = sorted({v['arm'] for v in solves.values()} - {'c17_graph'})
    pairs = {}
    for arm in arms:
        for phase in ('adam_ms_per_evaluation', 'lbfgs_ms_per_evaluation'):
            values = [by[(s, arm)][phase] / by[(s, 'c17_graph')][phase] for s in sorted({v['seed'] for v in solves.values()})
                      if (s, arm) in by and (s, 'c17_graph') in by and by[(s, arm)]['accepted'] and by[(s, 'c17_graph')]['accepted']
                      and phase in by[(s, arm)] and phase in by[(s, 'c17_graph')]]
            if values:
                pairs.setdefault(arm, {})[phase] = {'geometric_mean': math.exp(sum(map(math.log, values)) / len(values)),
                                                    'range': [min(values), max(values)], 'pairs': len(values)}
    ranges = {}
    for arm in sorted({v['arm'] for v in solves.values()}):
        for phase in ('adam_ms_per_evaluation', 'lbfgs_ms_per_evaluation'):
            values = [v[phase] for v in solves.values() if v['arm'] == arm and phase in v]
            ranges.setdefault(arm, {})[phase] = [min(values), max(values)] if values else None
    result = {'scope': 'post hoc; not part of the registered analysis', 'per_arm_ms_per_evaluation': ranges,
              'paired_over_c17': pairs, 'solves': solves}
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'ranges': ranges, 'pairs': {a: {p: round(v['geometric_mean'], 3) for p, v in d.items()} for a, d in pairs.items()}}))


if __name__ == '__main__':
    main()
