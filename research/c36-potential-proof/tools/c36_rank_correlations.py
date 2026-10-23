"""Rank correlation between log loss and log stratified error along each C36 run.

    python3 c36_rank_correlations.py c36-observations-20261015.json OUT.json

Uses the checkpoints of the original problem (width 1/2) that have a logged loss, as in
Figure 4. Spearman correlation with average ranks for ties. Pure standard library.
"""
import json
import math
import sys
from pathlib import Path

KEYS = ('normalized_corrected_pressure_rms', 'velocity_relative_l2')


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2
        i = j + 1
    return result


def spearman(x, y):
    a, b = ranks(x), ranks(y)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else None


def main():
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    observations = json.loads(source.read_text())['observations']
    result = {'source': source.name, 'definition': __doc__.strip().splitlines()[0], 'runs': {}}
    for run in sorted({r['run'] for r in observations}):
        rows = [r for r in observations if r['run'] == run and r['width'] == .5 and r['loss_at_checkpoint'] is not None]
        if len(rows) < 3:
            continue
        loss = [math.log10(r['loss_at_checkpoint']) for r in rows]
        result['runs'][run] = {'checkpoints': len(rows), **{
            key: spearman(loss, [math.log10(r['planes']['stratified']['global'][key]) for r in rows]) for key in KEYS}}
    out.write_text(json.dumps(result, indent=1) + '\n')
    for run, row in result['runs'].items():
        print(f"{run:26s} n={row['checkpoints']:2d} pressure {row[KEYS[0]]:.3f} velocity {row[KEYS[1]]:.3f}")


if __name__ == '__main__':
    main()
