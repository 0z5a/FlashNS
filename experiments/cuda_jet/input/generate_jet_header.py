"""Generate straight-line scalar jet primitives; no runtime Bell-table walks."""
from __future__ import annotations
from itertools import product
from pathlib import Path
from collections import Counter


def inds(dim: int) -> list[tuple[int, ...]]:
    return sorted((a for a in product(range(4), repeat=dim) if sum(a) <= 3),
                  key=lambda a: (sum(a),a))


def pair_expr(alpha, aa, left='z', right='z', exclude_zero=False):
    lookup={a:i for i,a in enumerate(aa)}
    pairs=[]
    for j,beta in enumerate(aa):
        if all(b<=a for a,b in zip(alpha,beta)):
            gamma=tuple(a-b for a,b in zip(alpha,beta))
            k=lookup[gamma]
            if exclude_zero and (j==0 or k==0):
                continue
            pairs.append(tuple(sorted((j,k))) if left==right else (j,k))
    count=Counter(pairs)
    return ' + '.join((f'{n}.0 * ' if n!=1 else '')+
                      f'{left}[{j}] * {right}[{k}]'
                      for (j,k),n in sorted(count.items())) or '0.0'


def make() -> str:
    lines=['#pragma once','#include <cmath>',
           '// Generated factorial-normalized 2-D/3-D jets, total degree <=3.',
           '// Host primitives verified. CUDA backend NOT compiled/tested here.',
           '// Inputs and outputs must not alias. FP64 ordinary arithmetic;',
           '// saturation-tail robustness requires separate high-precision tests.',
           '#ifdef __CUDACC__',
           '#define FLASHNS_JET_HD __host__ __device__ __forceinline__',
           '#else','#define FLASHNS_JET_HD inline','#endif',
           'namespace flashns {']
    for dim in (2,3):
        aa=inds(dim); lookup={a:i for i,a in enumerate(aa)}; q=len(aa)
        suffix=f'{dim}d3'
        lines += [f'// Coefficient order {suffix}: '+str(aa),
                  f'FLASHNS_JET_HD void tanh_fwd_{suffix}(const double* z, double* h) {{',
                  '  const double t = ::tanh(z[0]);',
                  '  const double a1 = 1.0 - t*t;',
                  '  const double a2 = -t*a1;',
                  '  const double a3 = a1*(t*t - 1.0/3.0);',
                  '  h[0] = t;']
        for i,a in enumerate(aa):
            if sum(a)>=2:
                lines.append(f'  const double p2_{i} = {pair_expr(a,aa,exclude_zero=True)};')
        for i,a in enumerate(aa[1:],1):
            expr=f'a1*z[{i}]'
            if sum(a)>=2: expr+=f' + a2*p2_{i}'
            if sum(a)==3:
                terms=[]
                for j,b in enumerate(aa):
                    if sum(b)==1 and all(bi<=ai for ai,bi in zip(a,b)):
                        k=lookup[tuple(ai-bi for ai,bi in zip(a,b))]
                        terms.append(f'z[{j}]*p2_{k}')
                expr+=' + a3*('+' + '.join(terms)+')'
            lines.append(f'  h[{i}] = {expr};')
        lines += ['}',f'FLASHNS_JET_HD void tanh_vjp_{suffix}(const double* h, const double* bh, double* bz) {{',
                  '  const double g0 = 1.0 - h[0]*h[0];']
        for i,a in enumerate(aa[1:],1):
            lines.append(f'  const double g{i} = -({pair_expr(a,aa,left="h",right="h")});')
        for j,b in enumerate(aa):
            terms=[]
            for i,a in enumerate(aa):
                if all(bi<=ai for ai,bi in zip(a,b)):
                    k=lookup[tuple(ai-bi for ai,bi in zip(a,b))]
                    terms.append(f'bh[{i}]*g{k}')
            lines.append(f'  bz[{j}] = '+' + '.join(terms)+';')
        lines += ['}']
    lines+=['} // namespace flashns','#undef FLASHNS_JET_HD','']
    return '\n'.join(lines)

if __name__=='__main__':
    path=Path(__file__).with_name('jet_primitives.cuh')
    path.write_text(make())
    print(path)
