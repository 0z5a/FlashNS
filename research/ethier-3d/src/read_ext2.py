"""Registered reader of the compiled Ethier external-baseline batch (protocol-ext2-20261017).

    python3 read_ext2.py --root BATCH_ROOT --output OUT.json

1. Selection: per method, the variant (graph or compiled) with the lower median warm-up replay time
   over the lanes, among variants whose warm-up qualification passed.
2. Primary and secondary statistics: selected external variant over selected C17 variant, paired
   geometric means of setup-plus-optimization and of optimization-only time over jointly accepted
   formal seeds, each classified by the registered decision rule.
3. Supplementary: every arm against c17_graph, the compile factor per method (graph over compiled,
   per evaluation and per warm-up replay), and each formal solve's cache state (first of its arm on
   its lane after the warm-up, and setup against the arm's median warm-up setup).
Reads the audit (audit-ext2.json) for acceptance and timings. Standard library only; no absolute paths.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def gm(values):
    logs = [math.log(v) for v in values]
    return math.exp(sum(logs) / len(logs)) if logs else None


def decide(ratios):
    if len(ratios) < 5:
        return 'not decidable'
    g = gm(ratios)
    if g > 1.03 and sum(r > 1 for r in ratios) >= 5:
        return 'C17 faster'
    if g < 1 / 1.03 and sum(r < 1 for r in ratios) >= 5:
        return 'external faster'
    return 'no decided difference'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    protocol = json.loads((root / 'protocol-ext2-20261017.json').read_text())
    audit = json.loads((root / 'audit-ext2.json').read_text())
    controller = json.loads((root / 'evidence' / 'controller.json').read_text())
    warm = {}
    for case in controller.get('warmups', []):
        path = root / 'evidence' / 'warm' / case['id'] / 'summary.json'
        summary = json.loads(path.read_text()) if path.is_file() else {}
        warm.setdefault(case['arm'], []).append({'lane': case['lane'], 'exit': case.get('actual_exit_code'), 'state': summary.get('state'),
                                                 'replay_ms': summary.get('warmup_replay_ms'),
                                                 'setup_seconds': summary.get('timing_seconds', {}).get('setup_capture')})
    methods = sorted({a.rsplit('_', 1)[0] for a in protocol['arms']})
    selection = {}
    for method in methods:
        candidates = {}
        for variant in ('graph', 'compiled'):
            rows = [w for w in warm.get(f'{method}_{variant}', []) if w['state'] in ('WARM_ONLY_COMPLETE', 'CPU_SMOKE_COMPLETE') and w['replay_ms']]
            if rows:
                candidates[variant] = statistics.median(w['replay_ms'] for w in rows)
        selection[method] = {'replay_ms_median': candidates,
                             'selected': min(candidates, key=candidates.get) if candidates else None}
    rows = {(r['seed'], r['arm']): r for r in audit['rows']}
    seeds = protocol['initialization_seeds']

    def ratios(arm, base, field):
        out = []
        for s in seeds:
            a, b = rows.get((s, arm)), rows.get((s, base))
            if a and b and a['accepted'] and b['accepted']:
                ta = a['timing_seconds'][field] if field != 'per_evaluation' else a['timing_seconds']['optimization_including_stop_checks'] / a['closure_evaluations']
                tb = b['timing_seconds'][field] if field != 'per_evaluation' else b['timing_seconds']['optimization_including_stop_checks'] / b['closure_evaluations']
                out.append(ta / tb)
        return out

    primary = {}
    c17 = selection.get('c17', {}).get('selected')
    for method in methods:
        if method == 'c17':
            continue
        chosen = selection[method]['selected']
        if not (c17 and chosen):
            primary[method] = {'decision': 'not decidable (no qualified variant)'}
            continue
        arm, base = f'{method}_{chosen}', f'c17_{c17}'
        entry = {'external_arm': arm, 'c17_arm': base}
        for name, field in (('setup_plus_optimization', 'solve_setup_plus_optimization'), ('optimization_only', 'optimization_including_stop_checks')):
            r = ratios(arm, base, field)
            entry[name] = {'geometric_mean': gm(r), 'range': [min(r), max(r)] if r else None, 'pairs': len(r),
                           'favour_c17': sum(x > 1 for x in r), 'decision': decide(r)}
        primary[method] = entry
    versus_c17_graph = {}
    for arm in protocol['arms']:
        if arm == 'c17_graph':
            continue
        entry = {}
        for name, field in (('setup_plus_optimization', 'solve_setup_plus_optimization'), ('optimization_only', 'optimization_including_stop_checks'), ('per_evaluation', 'per_evaluation')):
            r = ratios(arm, 'c17_graph', field)
            entry[name] = {'geometric_mean': gm(r), 'range': [min(r), max(r)] if r else None, 'pairs': len(r)}
        versus_c17_graph[arm] = entry
    compile_factor = {}
    for method in methods:
        r = ratios(f'{method}_graph', f'{method}_compiled', 'per_evaluation')
        compile_factor[method] = {'per_evaluation_graph_over_compiled': gm(r), 'pairs': len(r),
                                  'replay_graph_over_compiled': (selection[method]['replay_ms_median'].get('graph', 0) / selection[method]['replay_ms_median']['compiled'])
                                  if selection[method]['replay_ms_median'].get('compiled') else None}
    cache = []
    for lane_order in [sorted([c for c in controller['children'] if c['lane'] == lane], key=lambda c: c['started_unix']) for lane in sorted({c['lane'] for c in controller['children']})]:
        for c in lane_order:
            row = rows.get((c['seed'], c['arm']))
            warm_setups = [w['setup_seconds'] for w in warm.get(c['arm'], []) if w['setup_seconds']]
            cache.append({'id': c['id'], 'lane': c['lane'], 'setup_seconds': row['timing_seconds']['setup_capture'] if row and 'timing_seconds' in row else None,
                          'median_warmup_setup_seconds': statistics.median(warm_setups) if warm_setups else None})
    result = {'protocol_version': protocol['protocol_version'], 'selection': selection, 'warmups': warm, 'primary': primary,
              'versus_c17_graph': versus_c17_graph, 'compile_factor': compile_factor, 'cache_state': cache,
              'accepted': {arm: sum(1 for s in seeds if rows.get((s, arm), {}).get('accepted')) for arm in protocol['arms']}}
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'selection': {m: v['selected'] for m, v in selection.items()},
                      'primary': {m: {k: (v[k]['geometric_mean'], v[k]['decision']) for k in ('setup_plus_optimization', 'optimization_only')} if 'external_arm' in v else v
                                  for m, v in primary.items()}}))


if __name__ == '__main__':
    main()
