"""Fit a cash-value correction for trees within four steps of an opposing planter."""
import argparse
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_replays import load_strategy


def _contest_bin(features):
    # Model the opponent's travel time, which planter is closer and time left to collect.
    enemy = features[:, 11].astype(int)
    friendly = features[:, 10]
    race = np.where(friendly < enemy, 0, np.where(friendly == enemy, 1, 2))
    lifetime = np.searchsorted([6, 12], features[:, 0], side='left')
    return np.minimum(enemy, 4) * 9 + race * 3 + lifetime


def layout(replay):
    obs = replay['steps'][0][0]['observation']
    data = [replay['configuration']['size'], obs['full_carbon'],
            [sorted(p[1].values()) for p in obs['players']]]
    return hashlib.sha256(json.dumps(data, separators=(',', ':')).encode()).hexdigest()


def read_game(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def tree_examples(model, replay, seat):
    steps = replay['steps']
    ledgers = []
    for frame in steps:
        obs = frame[0]['observation']
        ledgers.append({tree: (1 - 2 * side) * obs['trees'][tree][1]
                        for side, p in enumerate(obs['players']) for tree in p[3]})
    tracker = model.BeliefTracker(model._Bel, seat)
    rows = []
    weights = model._TREE_MODEL
    for t, frame in enumerate(steps[:-1]):
        obs = frame[0]['observation']
        players = obs['players']
        estimated, _ = tracker.step(players, t, frame[seat]['observation']['carbon'])
        tracker.set_actions(players, steps[t + 1][seat]['action'] or {})
        if t % 3:
            continue
        keys, owners, features = model._tree_features(players, estimated, obs['trees'], t, replay['configuration'])
        if not keys:
            continue
        active = (features[:, 11] <= 4) & (features[:, 0] > 0)
        if not active.any():
            continue
        x = np.clip((features[active] - weights['mean']) * weights['scale'], -7, 7)
        predicted = np.maximum(x @ weights['w'] + weights['b'], 0) @ weights['out'] + weights['offset']
        indices = np.flatnonzero(active)
        labels = [sum(.96**(h-1) * ledgers[t+h].get(keys[j], 0)
                      for h in range(1, min(19, len(steps)-t))) * (1-2*owners[j]) for j in indices]
        rows.append(np.column_stack((_contest_bin(features[active]), predicted, labels)))
    return np.concatenate(rows) if rows else np.empty((0, 3))


def fit(rows):
    bins = rows[:, 0].astype(int)
    residual = np.clip(rows[:, 2] - rows[:, 1], -200, 200)
    return np.bincount(bins, residual, minlength=45) / (60 + np.bincount(bins, minlength=45))


def metrics(rows, coef):
    error = rows[:, 1] + coef[rows[:, 0].astype(int)] - rows[:, 2]
    return dict(n=len(rows), bias=float(error.mean()), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.mean(error**2))))


def export(template, correction, destination):
    anchor = 'def _future_tree_income(board, me, configuration):'
    line = '    forecast = np.where(features[:, 0] > 0, forecast, 0.)'
    assert template.count(anchor) == template.count(line) == 1
    table = '_CONTEST_CORRECTION = np.array(' + repr(correction.round(6).tolist()) + ')'
    program = template.replace(anchor, inspect.getsource(_contest_bin) + '\n' + table + '\n\n' + anchor)
    program = program.replace(line, '    forecast += np.where(features[:, 11] <= 4, '
                              '_CONTEST_CORRECTION[_contest_bin(features)], 0.)\n' + line)
    compile(program, str(destination), 'exec')
    with destination.open('x') as stream:
        stream.write(program)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='new directory for candidate and metrics')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'v5-tree-value:submission.py'])
    if hashlib.sha256(source).hexdigest() != '0483eca2a5e8031270dfb89b87cd5d901ee81f823383f6094f8ed839dec6a7dd':
        raise ValueError('Wrong baseline')
    baseline = args.output / 'yan-v2.py'
    baseline.write_bytes(source)
    model = load_strategy(baseline)
    apollo = ROOT / 'replays/apollo'
    seen_maps, seen_seeds = set(), set()
    for row in json.loads((apollo / 'manifest.json').read_text())['games']:
        replay = read_game(apollo / Path(row['file']).name)
        seen_maps.add(layout(replay))
        seen_seeds.add(replay['configuration']['randomSeed'])
    root = ROOT / 'replays/top-five'
    groups = {'fit': [], 'holdout': []}
    manifests = {k: [] for k in groups}
    for row in sorted(json.loads((root / 'index.json').read_text()), key=lambda r: r['id']):
        replay = read_game(root / row['file'])
        digest = layout(replay)
        seed = replay['configuration']['randomSeed']
        if digest in seen_maps or seed in seen_seeds:
            continue
        seen_maps.add(digest)
        seen_seeds.add(seed)
        key = 'holdout' if int(digest[:8], 16) % 4 == 0 else 'fit'
        seat = int(digest[-2:], 16) % 2
        rows = tree_examples(model, replay, seat)
        groups[key].append(rows)
        manifests[key].append(dict(id=row['id'], layout=digest, seed=seed, observer=seat, rows=len(rows)))
    fit_rows, holdout = [np.concatenate(groups[k]) for k in ('fit', 'holdout')]
    coef = fit(fit_rows)
    results = dict(maps=manifests, train_rows=len(fit_rows), baseline=metrics(holdout, np.zeros(45)),
                   corrected=metrics(holdout, coef), correction=coef.tolist())
    (args.output / 'heldout.json').write_text(json.dumps(results, indent=2) + '\n')
    export(source.decode(), coef, args.output / 'contested-trees.py')
    print({k:results[k] for k in ('train_rows','baseline','corrected')}, flush=True)


if __name__ == '__main__':
    main()
