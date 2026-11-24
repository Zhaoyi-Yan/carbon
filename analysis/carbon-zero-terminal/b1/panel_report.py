"""Summarize the B1 development panel: mixed utility (primary), margin, WDL.

Usage: python3 panel_report.py [evaluation-dir]
Deltas are paired by map/seat against the frozen ZW0 run and bootstrapped over
maps (4000 resamples).  The training objective is U, so U is the primary metric;
the raw margin is kept to expose cases where the two disagree.
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


def paired(a, b, rival, key):
    ga, gb = games(a, rival), games(b, rival)
    return {k: gb[k][key] - ga[k][key] for k in ga}


def line(d, scale=1.0, digits=3):
    ci = boot(d)
    return f'{np.mean(list(d.values())) * scale:+8.{digits}f} [{ci[0] * scale:+8.{digits}f},{ci[1] * scale:+8.{digits}f}]'


def show():
    print('paired deltas vs frozen ZW0: U (primary), margin; bootstrap over maps')
    for name in NAMES:
        for rival in RIVALS:
            rows = BLOCKS[name][rival]['games']
            text = f'{name:8} {rival:22} {wdl(rows):8} U={line(paired("ZW0", name, rival, "utility"), 1, 3)}'
            if name != 'ZW0':
                text += f' margin={line(paired("ZW0", name, rival, "margin"), 1, 0)}'
            print(text)
    for x, y in (('A10', 'B1_10k'), ('A50', 'B1_50k'), ('C50', 'B1_50k'), ('B1_10k', 'B1_50k')):
        if not all(p in NAMES for p in (x, y)):
            continue
        print(f'\n== {y} - {x} (paired) ==')
        for rival in RIVALS:
            print(f'  {rival:22} U={line(paired(x, y, rival, "utility"), 1, 3)}'
                  f' margin={line(paired(x, y, rival, "margin"), 1, 0)}')


if __name__ == '__main__':
    show()
