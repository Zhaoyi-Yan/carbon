"""Freeze screen/confirm maps for the v9 reactive-opening experiment."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark import configure_engine, new_game  # noqa: E402


def layout_of(seed):
    state, env = new_game(seed, 300)
    board = state[0].observation
    layout = [env.configuration.size, board.full_carbon,
              [sorted(p[1].values()) for p in board.players]]
    return hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest()


def replay_layout(path):
    game = json.load(gzip.open(path))
    obs = game['steps'][0][0]['observation']
    layout = [game['configuration']['size'], obs['full_carbon'],
              [sorted(p[1].values()) for p in obs['players']]]
    return hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest()


def main(a):
    configure_engine()
    excluded = set()
    index = json.loads((a.replays / 'index.json').read_text())
    v8_replays = [row for row in index
                  if any(entry.get('id') == '52af7735a7677815efb48cdf94269e29'
                         for entry in row['strategies'].values())]
    for row in v8_replays:
        excluded.add(replay_layout(a.replays / row['file']))
    prior = json.loads(a.prior_panels.read_text())
    prior_seeds = set()
    for rows in prior['panels'].values():
        for row in rows:
            prior_seeds.add(row['seed'])
    excluded.update(layout_of(seed) for seed in prior_seeds)
    v7 = json.loads((ROOT / 'experiments/v7/panels.json').read_text())
    v7_seeds = set(v7['development'])
    for fold in ('qualifying', 'confirmation'):
        for batch in v7[fold]:
            v7_seeds.update(batch)
    excluded.update(layout_of(seed) for seed in v7_seeds)
    manifest = json.loads(a.v9_manifest.read_text())
    excluded.update(row['layout_sha256'] for row in manifest['games'])
    known_seeds = prior_seeds | v7_seeds
    panels = {}
    for name, count, namespace in (('screen', 12, 'v9-reactive-screen-1'),
                                   ('confirm', 12, 'v9-reactive-confirm-1')):
        rows, index_i = [], 0
        while len(rows) < count:
            seed = int.from_bytes(hashlib.sha256(f'{namespace}/{index_i}'.encode()).digest()[:4], 'big')
            index_i += 1
            if seed in known_seeds:
                continue
            layout = layout_of(seed)
            if layout in excluded:
                continue
            excluded.add(layout)
            rows.append(dict(seed=seed, layout=layout))
        panels[name] = rows
    candidate = a.candidate.read_bytes()
    baseline = a.baseline.read_bytes()
    protocol = dict(experiment='v9-reactive-opening', candidate=dict(sha256=hashlib.sha256(candidate).hexdigest(),
                    bytes=len(candidate), change='remove the _v7_early_income call in _act'),
                    baseline=dict(sha256=hashlib.sha256(baseline).hexdigest(), bytes=len(baseline),
                                  source='e97a386b290a1727f26331470094ab57cd452948:yan-v8-own-prior.py'),
                    panels=panels, excluded_layouts=len(excluded),
                    excluded_sources=dict(v8_prior_replays=len(v8_replays), v8_prior_panel_seeds=len(prior_seeds),
                                          v7_panel_seeds=len(v7_seeds), v9_study_games=len(manifest['games'])),
                    rng=20261125, bootstrap=1000000,
                    gates=dict(screen='mean U > 0, normal finishes, no candidate failure',
                               confirm='mean U > 0 and 95% map-bootstrap lower bound > 0'))
    a.out.write_text(json.dumps(protocol, indent=2) + '\n')
    for name, rows in panels.items():
        print(name, [row['seed'] for row in rows], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('replays', type=Path)
    p.add_argument('prior_panels', type=Path)
    p.add_argument('v9_manifest', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('baseline', type=Path)
    p.add_argument('out', type=Path)
    main(p.parse_args())
