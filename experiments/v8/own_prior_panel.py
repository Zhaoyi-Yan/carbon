"""Pick fresh screen/confirmation maps disjoint from the prior-fit replays and old panels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark import configure_engine, new_game  # noqa: E402


def fingerprint(seed):
    state, _ = new_game(seed, 300)
    obs = state[0].observation
    players = obs['players']
    payload = json.dumps([obs['full_carbon'],
                          [sorted(players[0][1]), sorted(players[1][1])]], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def main(a):
    configure_engine()
    report = json.loads(a.report.read_text())
    used = {row['layout'] for row in report['sources']}
    old = json.loads((ROOT / 'experiments/v7/panels.json').read_text())
    old_seeds = set(old['development'])
    for fold in ('qualifying', 'confirmation'):
        for batch in old[fold]:
            old_seeds.update(batch)
    panels = {}
    for name, count, namespace in (('screen', 12, 'v8-own-prior-screen-1'),
                                   ('confirm', 12, 'v8-own-prior-confirm-1')):
        seeds, rows = [], []
        index = 0
        while len(seeds) < count:
            seed = int.from_bytes(hashlib.sha256(f'{namespace}/{index}'.encode()).digest()[:4], 'big')
            index += 1
            if seed in old_seeds:
                continue
            layout = fingerprint(seed)
            if layout in used:
                continue
            used.add(layout)
            seeds.append(seed)
            rows.append(dict(seed=seed, layout=layout))
        panels[name] = rows
    out = dict(rule='screen and confirm maps disjoint from prior-fit replay layouts and v7 panels',
               replay_layouts=len(report['sources']), old_panel_seeds=len(old_seeds), panels=panels)
    a.out.write_text(json.dumps(out, indent=2) + '\n')
    for name, rows in panels.items():
        print(name, [row['seed'] for row in rows])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    main(a)
