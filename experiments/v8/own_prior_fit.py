"""Refit the v8 own-search worker prior from traceable yan-QQ-v8 online replays.

The teacher is the deployed yan-QQ-v8 strategy (replay repository id 52af...);
replays carry no source SHA, so the local frozen file (a87f8e6b...) is the
documented proxy.  Only the per-role linear candidate prior is refitted; the
network, search, decoder and all other v8 logic are untouched.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/v7'))
from benchmark import configure_engine, load_strategy  # noqa: E402
from distill_fit import fit  # noqa: E402

V8_SHA = 'a87f8e6b83656c771cb5a25d6b6f383b245127e7c782e4c6c7c4a0a2cc15d396'
TEACHER_ID = '52af7735a7677815efb48cdf94269e29'
TEACHER_NAME = 'yan-QQ-v8'
ENCODE = ((0, 59), (85, 122), (162, 201), (245, 284))
SAMPLE = ((10, 59), (97, 122), (174, 201), (257, 284))


def load_v8():
    source = (ROOT / 'yan-v8.py').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != V8_SHA:
        raise ValueError(f'yan-v8.py hash {digest} != frozen {V8_SHA}')
    configure_engine()
    folder = Path(tempfile.mkdtemp(prefix='own-prior-v8-'))
    path = folder / 'v8.py'
    path.write_bytes(source)
    return load_strategy('own_prior_v8', path, source)


def fingerprint(game):
    obs = game['steps'][0][0]['observation']
    players = obs['players']
    payload = json.dumps([obs.get('full_carbon'),
                          [sorted(players[0][1]), sorted(players[1][1])]],
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def v8_seat(row):
    slot = next(name for name, entry in row['strategies'].items() if entry.get('id') == TEACHER_ID)
    order = row.get('order') or ['a', 'b']
    return order.index(slot)


def extract(agent, game, seat):
    cfg = game['configuration']
    steps = game['steps']
    agent._ST, agent._FC_HISTORY = {}, {}
    tracker = agent.BeliefTracker(agent._Bel, seat)
    previous = None
    groups = [[], [], [], []]
    for t in range(len(steps) - 1):
        obs = {k: v for k, v in steps[t][0]['observation'].items() if k != 'full_carbon'}
        obs.update(steps[t][seat]['observation'])
        obs.pop('full_carbon', None)
        obs['player'] = seat
        actions = steps[t + 1][seat]['action'] or {}
        ps = obs['players']
        if hasattr(agent, '_fc_track'):
            agent._fc_track(obs)
        est, known = tracker.step(ps, t, obs['carbon'])
        est, known = agent._sense_tree_carbon(tracker.bel, obs, cfg, previous, est, known)
        if any(a <= t < b for a, b in ENCODE):
            x = agent._quant(agent.encode(ps, t, seat, est, known, previous))
            lw, lb = agent._v5_root(obs, est, x, previous)
            lw, _ = agent._v6_response(obs, est, cfg, lw, lb)
            lw = agent._selfcredit_adjust(x, lw)
            if t % 2 == 0 and any(a <= t < b for a, b in SAMPLE):
                ids, roles, z, feat = agent._ds_features(x, ps, seat, previous, lw, None)
                if ids:
                    labels = [agent.WIDX.get(actions.get(uid), 0) for uid in ids]
                    for dst, src in zip(groups, (roles, z, feat, labels), strict=True):
                        dst.append(np.asarray(src))
        tracker.set_actions(ps, actions)
        previous = ps
        agent._ST['prev'] = ps
    if not groups[0]:
        raise ValueError('no labels extracted')
    return {key: np.concatenate(arr) for key, arr in zip(('role', 'z', 'x', 'y'), groups, strict=True)}


def measure(records, scale, weights):
    rows = []
    for meta, data in records:
        roles, z, x, y = (data[k] for k in ('role', 'z', 'x', 'y'))
        naive = z * scale[roles, None]
        adjusted = naive + np.einsum('naf,nf->na', x, weights[roles])
        scores = []
        for policy in (naive, adjusted):
            logp = policy - np.logaddexp.reduce(policy, axis=1, keepdims=True)
            scores.append([float(-logp[np.arange(len(y)), y].mean()),
                           float(np.mean(policy.argmax(1) == y))])
        rows.append(dict(**meta, workers=int(len(y)), calibrated=scores[0], learned=scores[1]))
    return rows


def main(a):
    agent = load_v8()
    index = json.loads((a.replays / 'index.json').read_text())
    entries = [row for row in index
               if any(entry.get('id') == TEACHER_ID for entry in row['strategies'].values())]
    if a.limit:
        entries = entries[:a.limit]
    a.cache.mkdir(parents=True, exist_ok=True)
    records, sources = [], []
    for row in entries:
        path = a.replays / row['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        game = json.load(gzip.open(path))
        seat = v8_seat(row)
        cache = a.cache / f"{digest[:16]}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            layout = fingerprint(game)
        else:
            arrays = extract(agent, game, seat)
            np.savez_compressed(cache, **arrays)
            layout = fingerprint(game)
        meta = dict(id=row['id'], game=row.get('game', 1), file=row['file'], sha256=digest,
                    layout=layout, seat=seat, seed=None)
        records.append((meta, arrays))
        sources.append(dict(meta, workers=int(len(arrays['y']))))
        print(len(records), row['id'], 'seat', seat, 'workers', len(arrays['y']), flush=True)
    layouts = sorted({meta['layout'] for meta, _ in records})
    order = sorted(layouts, key=lambda value: hashlib.sha256(('own-prior-v8' + value).encode()).hexdigest())
    hold = min(len(order) - 1, max(4, len(order) // 3))
    holdout = set(order[:hold])
    for meta, _ in records:
        meta['fold'] = 'holdout' if meta['layout'] in holdout else 'train'
    training = [data for meta, data in records if meta['fold'] == 'train']
    roles, z, x, y = (np.concatenate([d[key] for d in training]) for key in ('role', 'z', 'x', 'y'))
    scales, weights = [], []
    for role in (0, 1):
        use = roles == role
        zz = z[use]
        scale = float(max(.05, fit(zz[:, :, None], y[use], np.zeros_like(zz), .0005)[0]))
        scales.append(scale)
        weights.append(fit(x[use], y[use], zz * scale, .014))
    scales = np.asarray(scales)
    weights = np.asarray(weights)
    held = measure([r for r in records if r[0]['fold'] == 'holdout'], scales, weights)
    model = dict(scale=scales.round(7).tolist(), coefficients=weights.round(7).tolist())
    (a.out / 'own-search-model-v8.json').write_text(json.dumps(model) + '\n')
    report = dict(teacher=dict(id=TEACHER_ID, name=TEACHER_NAME, proxy_sha256=V8_SHA,
                               note='online replays carry no source SHA; the frozen local file is the proxy'),
                  windows=dict(encode=ENCODE, sample=SAMPLE),
                  replay_index_sha256=hashlib.sha256((a.replays / 'index.json').read_bytes()).hexdigest(),
                  teacher_seconds=0.0,
                  train_workers=int(len(y)), train_layouts=len({m['layout'] for m, _ in records if m['fold'] == 'train'}),
                  holdout_layouts=len(holdout), sources=sources, holdout=held,
                  training=[m for m, _ in records if m['fold'] == 'train'],
                  model=model)
    (a.out / 'own-search-fit-v8.json').write_text(json.dumps(report, indent=2) + '\n')
    print('held', np.mean([r['calibrated'] for r in held], axis=0),
          np.mean([r['learned'] for r in held], axis=0), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('replays', type=Path, help='carbon-league-replays checkout')
    p.add_argument('out', type=Path)
    p.add_argument('--cache', type=Path, default=Path('/tmp/own-prior-v8-cache'))
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    main(a)
