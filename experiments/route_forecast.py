"""Learn how loaded enemy collectors move, from the observing seat of public replays.

Run from the repository root: python experiments/route_forecast.py
Optionally export a single-file arena candidate with --candidate /tmp/route-opponent.py.
"""
import argparse
import gzip
from hashlib import sha256
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTION = ('NONE', 'UP', 'RIGHT', 'DOWN', 'LEFT')
MIN_LOAD = 75


def route_rows(players, previous, seat, step, logits, size, minimum_load, distances, destinations):
    """Construct five candidate moves using only the public unit state and observer-side prior."""
    own = players[seat]
    opposing = players[1-seat]
    homes = list(own[1].values())
    if not homes:
        return []
    bank_dist = distances[:, homes].min(1)
    pursuers = [(pos, cargo) for pos, cargo, role in opposing[2].values() if role == 'COLLECTOR']
    enemy_tree = {t[0] for t in opposing[3].values()}
    allies = {w[0] for w in own[2].values()}
    rows = []
    for worker, (pos, load, role) in own[2].items():
        if role != 'COLLECTOR' or load < minimum_load:
            continue
        dest = destinations[pos]
        steps_home = bank_dist[pos] - bank_dist[dest]
        previous_pos = previous[seat][2].get(worker, [pos])[0] if previous is not None else pos
        # If multiple cardinal moves wrap to one destination on a tiny board,
        # the learned model is used only on the standard 15x15 board.
        last = np.flatnonzero(destinations[previous_pos] == pos)
        lighter = [p for p, c in pursuers if c < load]
        danger = distances[np.ix_(dest, lighter)].min(1) if lighter else np.full(5, size)
        tied = [p for p, c in pursuers if c == load]
        tie_risk = (distances[np.ix_(dest, tied)].min(1) <= 1) if tied else np.zeros(5)
        # Decision-time context: at least one off-base possible destination can
        # also be reached by an opposing, lighter collector on this turn.
        contested = bool(np.any((danger <= 1) & (bank_dist[dest] != 0)))
        idle = np.arange(5) == 0
        log_load = min(float(np.log1p(load/minimum_load)), 2.5)
        danger_move = danger <= 1
        prior = logits[pos].astype(np.float64)
        x = np.column_stack((
            prior - prior.max(), idle, steps_home, steps_home * log_load,
            bank_dist[dest] == 0, idle * log_load, danger_move, tie_risk,
            danger_move * steps_home, np.minimum(3, danger)/3.,
            np.arange(5) == (last[0] if len(last) else -1),
            [int(d) in enemy_tree for d in dest],
            [int(d) != pos and int(d) in allies for d in dest],
            steps_home * (step >= 270),
        ))
        rows.append((worker, pos, x, contested, steps_home))
    return rows


def choice_probability(logits):
    a = np.exp(logits-logits.max(axis=-1, keepdims=True))
    return a/a.sum(axis=-1, keepdims=True)


def fit(x, y):
    """Convex multinomial conditional logit, with a small fixed ridge penalty."""
    n, _, width = x.shape
    ridge = .001
    w = np.zeros(width)
    eye = np.eye(width)
    chosen = x[np.arange(n), y]
    for _ in range(40):
        z = x @ w
        prob = choice_probability(z)
        mean = np.einsum('na,naf->nf', prob, x)
        diff = x - mean[:, None, :]
        grad = (mean-chosen).mean(axis=0) + ridge*w
        if np.max(np.abs(grad)) < 1e-7:
            break
        hess = np.einsum('naf,nag->fg', diff*prob[:, :, None], diff)/n + ridge*eye
        step = np.linalg.solve(hess, grad)
        # Backtracking keeps convergence reliable for highly certain neural priors.
        old = -np.log(np.maximum(prob[np.arange(n), y], 1e-240)).mean()+ridge*(w@w)/2
        rate = 1.
        while rate >= 1e-5:
            nxt = w-rate*step
            p = choice_probability(x @ nxt)
            loss = -np.log(np.maximum(p[np.arange(n), y], 1e-240)).mean()+ridge*(nxt@nxt)/2
            if loss <= old:
                w = nxt
                break
            rate *= .5
        else:
            break
    return w


def collect(strategy, replay_dir, target, exclude=()):
    excluded_layouts = {g['map_sha256'] for g in exclude}
    excluded_seeds = {g['seed'] for g in exclude}
    manifest = replay_dir/'manifest.json'
    if manifest.exists():
        catalog = json.loads(manifest.read_text())['games']
        data_dir = replay_dir
    else:
        data_dir = replay_dir/'replays'
        catalog = [g for g in json.loads((replay_dir/'index.json').read_text())
                   if (data_dir/Path(g['file']).name).is_file()]
    games = sorted(catalog, key=lambda g: (g['id'], g['game']))
    dist = strategy.FEAT_DIST
    moves = np.array([[strategy.move(i, a) for a in range(5)] for i in range(strategy.N)])
    maps, feats, acts, flags, routes = [], [], [], [], []
    net = strategy._net()
    if net is None:
        raise RuntimeError('CPU PyTorch is required for the reference opponent forecast')
    game_ids = []
    for game in games:
        sides = [slot for slot,v in game['strategies'].items() if v['name'] == target]
        if len(sides) != 1:
            continue
        seat = game['order'].index(sides[0])
        with gzip.open(data_dir/Path(game['file']).name) as f:
            replay = json.load(f)
        cfg = replay['configuration']
        if cfg['size'] != 15 or cfg['moveCost'] != 0 or cfg['episodeSteps'] != 300:
            raise ValueError('Route model expects default 15x15, 300-step, zero-move-cost game')
        seed = int(cfg['randomSeed'])
        start = replay['steps'][0][0]['observation']
        layout = [cfg['size'], start['full_carbon'], [sorted(p[1].values()) for p in start['players']]]
        sig = sha256(json.dumps(layout,separators=(',', ':')).encode()).hexdigest()
        if sig in excluded_layouts or seed in excluded_seeds:
            continue
        for i, slot in enumerate(game['order']):
            if abs(replay['rewards'][i]-game['scores'][slot]) >= .02:
                raise ValueError(f"Replay seat ordering disagrees with scores for {game['id']}")
        excluded_layouts.add(sig)
        excluded_seeds.add(seed)
        game_ids.append({'id':game['id'],'game':game['game'],'seed':seed,'map_sha256':sig})
        observer = 1-seat
        tracker = strategy.BeliefTracker(strategy._Bel, observer)
        previous = None
        pending = []
        def flush():
            if not pending:
                return
            lw, _, _ = net(np.stack([p[0] for p in pending]), strategy._SID)
            for (_, players, previous, step, labels), logits in zip(pending, lw):
                for worker, _, x, contested, toward in route_rows(
                        players, previous, seat, step, logits, 15, MIN_LOAD, dist, moves):
                    maps.append(seed)
                    feats.append(x)
                    acts.append(ACTION.index(labels.get(worker) or 'NONE'))
                    flags.append(contested)
                    routes.append(toward)
            pending.clear()
        steps = replay['steps']
        for t, (cur, nxt) in enumerate(zip(steps, steps[1:])):
            players = cur[0]['observation']['players']
            est, _ = tracker.step(players,t,cur[observer]['observation']['carbon'])
            # Mirrors the v5 root opponent model; full_carbon and the enemy's
            # private observed carbon never enter the features or reference prior.
            x = strategy._quant(strategy.encode(players,t,seat,est,[1.] * strategy.N,previous))
            pending.append((x,players,previous,t,nxt[seat]['action'] or {}))
            tracker.set_actions(players,nxt[observer]['action'] or {})
            previous = players
            if len(pending) == 16:
                flush()
        flush()
    if not maps:
        raise ValueError('No matching games with eligible loaded-collector actions')
    return game_ids, tuple(np.array(col) for col in (maps,feats,acts,flags,routes))


def metrics(prob, labels, mask):
    if not mask.any():
        return {'decisions': 0}
    p, y = prob[mask], labels[mask]
    return dict(decisions=len(y), top1=float(np.mean(p.argmax(1)==y)),
                log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-240)).mean()))


def score(data, temperature):
    seeds, x, y, contested, toward = data
    # Only map seed sets folds. Every match and both seats sharing a map stay together.
    folds = {seed: i%5 for i,seed in enumerate(sorted(set(seeds)))}
    if len(folds) < 5:
        raise ValueError('Five-fold evaluation requires five unique maps')
    fold = np.array([folds[s] for s in seeds])
    preds = {'v5_raw': choice_probability(x[:,:,0]/temperature),
             'temperature_only': np.empty((len(y),5)), 'tactical': np.empty((len(y),5))}
    for f in range(5):
        test = fold == f
        for key,width in [('temperature_only',1),('tactical',x.shape[-1])]:
            w = fit(x[~test,:,:width], y[~test])
            preds[key][test] = choice_probability(x[test,:,:width] @ w)
    report = {'all':{}, 'contested':{}, 'homeward_share':{}, 'map_top1_delta':{}}
    for name,mask in [('all', np.ones(len(y),dtype=bool)),('contested',contested)]:
        for k,p in preds.items():
            report[name][k] = metrics(p,y,mask)
    for name,mask in [('contested',contested),('other',~contested)]:
        report['homeward_share'][name] = float((toward[np.arange(len(y)),y][mask]>0).mean()) if mask.any() else None
    for seed in sorted(folds):
        mask = (seeds==seed)&contested
        if mask.any():
            t = metrics(preds['tactical'],y,mask)['top1']
            b = metrics(preds['v5_raw'],y,mask)['top1']
            report['map_top1_delta'][str(seed)] = t-b
    return report, fit(x,y)


def export_candidate(template, path, weights):
    if path.exists():
        raise FileExistsError(path)
    source = template.read_text()
    anchor = '    lwo, lbo, _ = net(xo.astype(np.float32), _SID)\n'
    search = 'def _search(obs,'
    if source.count(anchor) != 1 or source.count(search) != 1:
        raise ValueError('Expected the v5 search and root opponent call in the strategy template')
    runtime = (inspect.getsource(route_rows) + '\n_ROUTE_DEST = np.asarray([[move(i,a) for a in range(5)] for i in range(N)])\n'
               + '_ROUTE_WEIGHTS = np.asarray('+repr(weights.tolist())+')\n\n')
    source = source.replace(search, runtime+search,1)
    amendment = '''    if (configuration.get('moveCost', 0) == 0 and configuration.get('episodeSteps', 300) == 300):
        for _, pos, features, _, _ in route_rows(players, prev, op, step, lwo[0], SIZE, 75, FEAT_DIST, _ROUTE_DEST):
            lwo[0, pos] = _TEMP * (features @ _ROUTE_WEIGHTS)
'''
    source = source.replace(anchor,anchor+amendment,1)
    compile(source,str(path),'exec')
    path.write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', type=Path, default=ROOT/'submission.py')
    parser.add_argument('--replays', type=Path, default=ROOT/'replays'/'apollo')
    parser.add_argument('--target', default='CarbonEdge_Apollo')
    parser.add_argument('--external-replays', type=Path, help='Optional replay directory or official snapshot with index.json')
    parser.add_argument('--external-target', help='Another player name for out-of-style testing')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--profile', choices=('tactical', 'temperature'), default='tactical',
                        help='Select which opponent model is embedded into the exported candidate')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('route_reference', args.strategy)
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)
    games,data = collect(strategy,args.replays,args.target)
    results,weights = score(data,strategy._TEMP)
    report = {'target':args.target, 'games':games, 'distinct_seeds':len(set(data[0])),
              'minimum_cargo':MIN_LOAD,'folds':5,'metrics':results}
    if args.external_replays or args.external_target:
        if not (args.external_replays and args.external_target):
            parser.error('--external-replays and --external-target are required together')
        external_games, external = collect(strategy,args.external_replays,args.external_target,exclude=games)
        _, x, y, contested, _ = external
        _, train_x, train_y, _, _ = data
        baseline_scale = fit(train_x[:,:,:1],train_y)
        predictions = {'v5_raw':choice_probability(x[:,:,0]/strategy._TEMP),
                       'temperature_only':choice_probability(x[:,:,:1] @ baseline_scale),
                       'tactical':choice_probability(x @ weights)}
        outside = {name:{k:metrics(prob,y,mask) for k,prob in predictions.items()}
                   for name,mask in [('all',np.ones(len(y),dtype=bool)),('contested',contested)]}
        report['external'] = {'target':args.external_target,'games':external_games,'metrics':outside}
        print('External',json.dumps(outside,indent=2))
    print(json.dumps(report['metrics'],indent=2))
    if args.report:
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    if args.candidate:
        if args.profile == 'temperature':
            _, x, y, _, _ = data
            weights = np.zeros_like(weights)
            weights[:1] = fit(x[:,:,:1], y)
        export_candidate(args.strategy,args.candidate,weights)

if __name__ == '__main__':
    main()
