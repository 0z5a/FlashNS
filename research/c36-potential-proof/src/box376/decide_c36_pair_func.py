"""Registered decision of the analytic vs torch.func paired cohort.

box376.decide_c36_pair applied with the torch.func arm in place of the parameter-AD arm: the
primary ratio is the torch.func arm's active training seconds to the loss threshold over the
analytic arm's, over jointly accepted pairs, with the same margin and counts.
"""
from box376 import decide_c36_pair

OUTCOMES = {'ANALYTIC_FASTER': 'ANALYTIC_FASTER', 'PARAMETER_AD_FASTER': 'TORCH_FUNC_FASTER',
            'NO_DECIDED_DIFFERENCE': 'NO_DECIDED_DIFFERENCE', 'INSUFFICIENT_PAIRS': 'INSUFFICIENT_PAIRS'}


def pair_entry(seed, arms):
    return decide_c36_pair.pair_entry(seed, {'analytic': arms.get('analytic'), 'parameter_ad': arms.get('func_jacfwd')})


def decide(seeds, entries, **kwargs):
    result = decide_c36_pair.decide(seeds, entries, **kwargs)
    result['outcome'] = OUTCOMES[result['outcome']]
    if 'pairs_favouring_parameter_ad' in result:
        result['pairs_favouring_torch_func'] = result.pop('pairs_favouring_parameter_ad')
    return result
