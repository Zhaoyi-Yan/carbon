"""Extract per-unit frozen-trunk features and future cash labels from top-20 matches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from backbone import load
from features import _v5_observe, _v5_context

HERE = Path(__file__).resolve().parent
TEACHERS = {p['id']: i for i, p in enumerate(json.loads((HERE/'design.json').read_text())['ranking'])}


def initialize():
    global MODEL, NET
    MODEL = load()
    NET = MODEL._net()


def extract(task):
    game, path, dest = task
    if dest.exists():
        return
    replay = json.loads(gzip.decompress(path.read_bytes()))
    frames = replay['steps']
    cfg = replay['configuration']
    last = cfg['episodeSteps'] - 2
    assert cfg['size'] == 15 and last == 298 and len(frames) == 300
    initial = frames[0][0]['observation']
    layout = [initial['full_carbon'], [sorted(p[1].values()) for p in initial['players']]]
    key = hashlib.sha256(json.dumps(layout,separators=(',', ':')).encode()).hexdigest()
    seed = int(cfg['randomSeed'])
    hx, actions, roles, styles, economics, routes, responses, teachers, visible, phase = [], [], [], [], [], [], [], [], [], []
    style_delta = int(key[:8], 16) % 5
    for side, slot in enumerate(game['order']):
        teacher = TEACHERS.get(game['strategies'][slot]['id'])
        if teacher is None:
            continue
        tracker = MODEL.BeliefTracker(MODEL._Bel, side)
        previous = None
        batch, positions = [], []
        def flush():
            if not batch:
                return
            hidden = NET(np.stack(batch), MODEL._SID, hidden=True)
            for h, pos in zip(hidden, positions):
                hx.extend(h[pos].astype(np.float16))
            batch.clear()
            positions.clear()
        for t, (before, after) in enumerate(zip(frames, frames[1:])):
            obs = before[0]['observation']
            players = obs['players']
            carbon, known = tracker.step(players, t, before[side]['observation']['carbon'])
            action = after[side]['action'] or {}
            if t % 5 == style_delta:
                econ, behavior = _v5_observe(players, obs.get('trees', {}), t, last)
                value, route = _v5_context(econ, behavior, players, t, last, side)
                future = frames[min(t + 48, len(frames)-1)][0]['observation']['players']
                change = (future[side][0]-future[1-side][0]) - (players[side][0]-players[1-side][0])
                economics.append(value); routes.append(route); responses.append(change/1000.)
                teachers.append(teacher); visible.append(behavior[side]); phase.append(t//75)
                x = MODEL._quant(MODEL.encode(players,t,side,carbon,known,previous))
                batch.append(x.astype(np.float32))
                pl = players[side]
                cells = []
                for wid, (pos, _, kind) in pl[2].items():
                    cells.append(pos); actions.append(MODEL.WIDX.get(action.get(wid), 0))
                    roles.append(0 if kind=='COLLECTOR' else 1); styles.append(teacher)
                if pl[0] >= 30 and len(pl[2]) < 10:
                    for bid, pos in pl[1].items():
                        cells.append(pos); actions.append(MODEL.BIDX.get(action.get(bid), 0))
                        roles.append(2); styles.append(teacher)
                positions.append(np.array(cells,dtype=int))
                if len(batch) == 12:
                    flush()
            tracker.set_actions(players, action)
            previous = players
        flush()
    assert len(hx) == len(actions)
    temp = dest.with_suffix('.partial.npz')
    np.savez(temp, H=np.asarray(hx,dtype=np.float16), A=np.asarray(actions,dtype=np.uint8),
             K=np.asarray(roles,dtype=np.uint8), T=np.asarray(styles,dtype=np.uint8),
             E=np.asarray(economics,dtype=np.float32), C=np.asarray(routes,dtype=np.float32),
             Y=np.asarray(responses,dtype=np.float32), J=np.asarray(teachers,dtype=np.uint8),
             B=np.asarray(visible,dtype=np.float32), P=np.asarray(phase,dtype=np.uint8),
             layout=key, seed=seed)
    temp.replace(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('snapshots', nargs='+', type=Path)
    p.add_argument('--jobs', type=int, default=3)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    games={}
    for root in args.snapshots:
        for g in json.loads((root/'index.json').read_text()):
            path=root/g['file']
            if (path.is_file() and g['steps']==299 and all(s=='DONE' for s in g['statuses'].values())
                    and any(s['id'] in TEACHERS for s in g['strategies'].values())):
                games.setdefault((g['id'],g['game']), (g,path,args.out/(path.name.split('.')[0]+'.npz')))
    print('complete eligible matches:',len(games),flush=True)
    with ProcessPoolExecutor(args.jobs,initializer=initialize) as pool:
        for i,_ in enumerate(pool.map(extract, games.values()),1):
            if i%30==0:
                print('cached',i,flush=True)


if __name__=='__main__':
    main()
