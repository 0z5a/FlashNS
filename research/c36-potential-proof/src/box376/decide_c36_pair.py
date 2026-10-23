"""Registered decision of the paired C36 engine cohort (analytic vs parameter-AD engine).

For each seed, the primary ratio is the parameter-AD arm's active training seconds to the loss
threshold over the analytic arm's. A pair counts only when it is jointly accepted: both arms reach
the threshold, both threshold checkpoints pass physical acceptance and both independent audits pass.
"""
import math

MARGIN = 1.03


def pair_entry(seed, arms):
    """arms: {'analytic': audit entry, 'parameter_ad': audit entry}; returns the per-seed record."""
    accepted = all(a is not None and a.get('independent_audit_pass') and a.get('threshold_reached')
                   and a.get('physical_acceptance_pass') for a in (arms.get('analytic'), arms.get('parameter_ad')))
    row = {'seed': seed, 'jointly_accepted': bool(accepted)}
    if accepted:
        t_an, t_ad = arms['analytic']['active_training_seconds'], arms['parameter_ad']['active_training_seconds']
        if not all(isinstance(t, (int, float)) and 0 < t < math.inf for t in (t_an, t_ad)):
            raise ValueError('positive finite training seconds required')
        row.update(analytic_seconds=t_an, parameter_ad_seconds=t_ad, ratio=t_ad / t_an)
    return row


def decide(seeds, entries, minimum_pairs=4, minimum_agreeing=4):
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate registered seeds')
    by_seed = {e['seed']: e for e in entries}
    if set(by_seed) - set(seeds):
        raise ValueError('unregistered seed')
    rows = [by_seed.get(s, {'seed': s, 'jointly_accepted': False}) for s in seeds]
    ratios = [r['ratio'] for r in rows if r['jointly_accepted']]
    result = {'denominator': len(seeds), 'jointly_accepted_pairs': len(ratios), 'rows': rows,
              'margin': MARGIN, 'minimum_pairs': minimum_pairs, 'minimum_agreeing': minimum_agreeing}
    if len(ratios) < minimum_pairs:
        result.update(outcome='INSUFFICIENT_PAIRS', geometric_mean_ratio=None)
        return result
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    faster_an, faster_ad = sum(r > 1 for r in ratios), sum(r < 1 for r in ratios)
    if gm > MARGIN and faster_an >= minimum_agreeing:
        outcome = 'ANALYTIC_FASTER'
    elif gm < 1 / MARGIN and faster_ad >= minimum_agreeing:
        outcome = 'PARAMETER_AD_FASTER'
    else:
        outcome = 'NO_DECIDED_DIFFERENCE'
    result.update(outcome=outcome, geometric_mean_ratio=gm, range=[min(ratios), max(ratios)],
                  pairs_favouring_analytic=faster_an, pairs_favouring_parameter_ad=faster_ad)
    return result
