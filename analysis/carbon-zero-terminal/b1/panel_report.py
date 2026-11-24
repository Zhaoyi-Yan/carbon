"""Summarize the B1 development panel: margins and paired deltas, bootstrapped over maps.

Usage: python3 panel_report.py [evaluation-dir]
"""
import json
from pathlib import Path
import sys
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name('evaluation'))
RIVALS = ('ZW0', 'ft3-generic-no-search', 'yan-v6-no-search')
NAMES = [n for n in ('ZW0', 'A10', 'B1_10k', 'B1_50k', 'A50', 'C50', 'C100')
         if (OUT / f'{n}.json').exists()]
BLOCKS = {n: json.loads((OUT / f'{n}.json').read_text())['rivals'] for n in NAMES}


def games(name, rival):
    return {(g['seed'], g['seat']): g for g in BLOCKS[name][rival]['games']}


def wdl(rows):
    w = sum(r['margin'] > 0 for r in rows)
    d = sum(r['margin'] == 0 for r in rows)
    return f'{w}-{d}-{len(rows) - w - d}'


def boot(deltas, rep=4000, seed=11):
    seeds = sorted({s for s, _ in deltas})
    by = {s: [v for (ss, _), v in deltas.items() if ss == s] for s in seeds}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(rep):
        pick = rng.choice(seeds, len(seeds), replace=True)
        vals.append(np.mean([np.mean(by[s]) for s in pick]))
    return np.percentile(vals, [2.5, 97.5])


def show():
    for name in NAMES:
        for rival in RIVALS:
            rows = BLOCKS[name][rival]['games']
            line = f'{name:8} {rival:22} {wdl(rows):8} mm={np.mean([g["margin"] for g in rows]):+8.1f}'
            if name != 'ZW0':
                base = games('ZW0', rival)
                d = {k: g['margin'] - base[k]['margin'] for k, g in games(name, rival).items()}
                ci = boot(d)
                line += f' d={np.mean(list(d.values())):+7.1f} CI[{ci[0]:+.0f},{ci[1]:+.0f}]'
            print(line)
    for x, y in (('A10', 'B1_10k'), ('A50', 'B1_50k'), ('B1_10k', 'B1_50k'), ('C50', 'B1_50k')):
        if not all(p in NAMES for p in (x, y)):
            continue
        print(f'\n== {y} - {x} (paired) ==')
        for rival in RIVALS:
            a, b = games(x, rival), games(y, rival)
            d = {k: b[k]['margin'] - a[k]['margin'] for k in a}
            ci = boot(d)
            print(f'  {rival:22} d={np.mean(list(d.values())):+7.1f} CI[{ci[0]:+.0f},{ci[1]:+.0f}]')


if __name__ == '__main__':
    show()
