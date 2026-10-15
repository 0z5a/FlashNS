"""TikZ log-log scatter: stratified corrected pressure (and velocity) versus checkpoint loss.

python3 c36_trend_figure.py c36-observations-20261015.json OUT.tex
Only checkpoints of the original problem (width 1/2) with a logged loss are drawn.
"""
import json
import math
import sys

trends = json.load(open(sys.argv[1]))
rows = [r for r in trends['observations'] if r['loss_at_checkpoint'] is not None and r['width'] == .5]


def arm(run):
    if run.endswith('-analytic'):
        return 'v2 analytic'
    if run.endswith('-parameter_ad'):
        return 'v2 parameter AD'
    return 'C1' if run.endswith('-C1') else 'C2'


MARKS = {'v2 analytic': 'o', 'v2 parameter AD': 'square', 'C1': 'triangle', 'C2': 'diamond'}
COLORS = {'v2 analytic': 'black', 'v2 parameter AD': 'black!55', 'C1': 'blue!70!black', 'C2': 'red!70!black'}
W, H = 6.4, 4.6
xs = [math.log10(r['loss_at_checkpoint']) for r in rows]
x0, x1 = math.floor(min(xs)), math.ceil(max(xs))


def panel(key, label, limit, shift):
    ys = [math.log10(r['planes']['stratified']['global'][key]) for r in rows]
    # Margins keep the limit line and the extreme points off the frame.
    y0, y1 = min(ys + [math.log10(limit)]) - .25, max(ys + [math.log10(limit)]) + .15
    X = lambda v: (v - x0) / (x1 - x0) * W + shift
    Y = lambda v: (v - y0) / (y1 - y0) * H
    out = [f'\\draw[thin] ({X(x0):.3f},0) rectangle ({X(x1):.3f},{H:.3f});']
    for e in range(x0, x1 + 1):
        out.append(f'\\draw ({X(e):.3f},0) -- ({X(e):.3f},.08) node[below=2pt,font=\\scriptsize] {{$10^{{{e}}}$}};')
    for e in range(math.ceil(y0), math.floor(y1) + 1):
        out.append(f'\\draw ({X(x0):.3f},{Y(e):.3f}) -- ({X(x0) + .08:.3f},{Y(e):.3f}) node[left=3pt,font=\\scriptsize] {{$10^{{{e}}}$}};')
    out.append(f'\\draw[dashed] ({X(x0):.3f},{Y(math.log10(limit)):.3f}) -- ({X(x1):.3f},{Y(math.log10(limit)):.3f}) '
               f'node[above left,font=\\scriptsize] {{limit}};')
    for r, x, y in zip(rows, xs, ys):
        a = arm(r['run'])
        out.append(f'\\node[mark size=1.5pt,{COLORS[a]}] at ({X(x):.3f},{Y(y):.3f}) {{\\pgfuseplotmark{{{MARKS[a]}}}}};')
    out.append(f'\\node[font=\\small] at ({X((x0 + x1) / 2):.3f},-.75) {{checkpoint loss}};')
    out.append(f'\\node[font=\\small,rotate=90] at ({X(x0) - 1.0:.3f},{H / 2:.3f}) {{{label}}};')
    return out


lines = ['\\begin{tikzpicture}']
lines += panel('normalized_corrected_pressure_rms', 'stratified pressure', 1e-3, 0)
lines += panel('velocity_relative_l2', 'stratified velocity', 1e-2, W + 2.0)
legend_y = H + .45
for i, (a, m) in enumerate(MARKS.items()):
    x = .3 + i * 3.3
    lines.append(f'\\node[mark size=1.6pt,{COLORS[a]}] at ({x:.2f},{legend_y:.2f}) {{\\pgfuseplotmark{{{m}}}}};')
    lines.append(f'\\node[right,font=\\scriptsize] at ({x + .1:.2f},{legend_y:.2f}) {{{a}}};')
lines.append('\\end{tikzpicture}')
open(sys.argv[2], 'w').write('\n'.join(lines) + '\n')
print(len(rows), 'points', x0, x1)
