"""Score raw and online-calibrated pre-coordination opponent moves in recorded yan-v5 games."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np


def score_game(candidate, replay, me):
    tracker = candidate.BeliefTracker(candidate._Bel, me)
    previous = None
    candidate._ST.clear()
    sums = np.zeros((2, 2, 5))
    steps = replay['steps']
    candidate.LAST_TURN = replay['configuration']['episodeSteps'] - 2
    for t, (cur, nxt) in enumerate(zip(steps,steps[1:])):
        full = cur[0]['observation']
        players = full['players']
        own_obs = {**cur[me]['observation'], 'step': t, 'player': me, 'players': players, 'trees': full.get('trees', {})}
        est, known = tracker.step(players, t, own_obs['carbon'])
        x = candidate._quant(candidate.encode(players, t, me, est, known, previous))
        candidate._v5_root(own_obs, est, x, previous)
        state = candidate._ST['v5_history']
        seat, scales = state['opponent_scale']
        assert seat == 1-me
        predicted, _ = state['opponent']
        enemy_action = nxt[seat]['action'] or {}
        for uid, (pos, _, kind) in players[seat][2].items():
            role = int(kind == 'PLANTER')
            raw = predicted[pos].astype(float) / (scales[role] * candidate._TEMP)
            label = candidate.WIDX[enemy_action.get(uid) or 'NONE']
            for i, z in enumerate((raw, raw*scales[role])):
                p = np.exp(z-z.max())
                p /= p.sum()
                sums[i,role] += [1, int(p.argmax()==label), -np.log(max(p[label],1e-100)), p.max(), scales[role]]
        tracker.set_actions(players, nxt[me]['action'] or {})
        previous = players
    return sums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('v6_candidate',args.candidate)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    rows = json.loads((args.data/'index.json').read_text())
    ready = []
    for g in sorted(rows, key=lambda r: (r['id'],r['game'])):
        slots = [k for k,v in g['strategies'].items() if v['name']=='yan-v5']
        if len(slots)==1 and (args.data/g['file']).is_file():
            ready.append((g,slots[0]))
    if args.limit:
        ready = ready[:args.limit]
    if not ready:
        raise ValueError('No local yan-v5 replays')
    with args.output.open('x') as report:
        for g,slot in ready:
            with gzip.open(args.data/g['file']) as f:
                r = json.load(f)
            me = g['order'].index(slot)
            if not all(abs(r['rewards'][i]-g['scores'][s])<.02 for i,s in enumerate(g['order'])):
                raise ValueError('Replay seat/score mismatch')
            stats = score_game(candidate,r,me)
            rec = {'id':g['id'], 'game':g['game'], 'seed':r['configuration']['randomSeed'], 'yan_seat':me,
                   'opponent':g['strategies'][g['order'][1-me]]['name'], 'turns':len(r['steps'])-1,
                   'counts_top1_nll_confidence_scale':stats.tolist()}
            report.write(json.dumps(rec)+'\n')
            report.flush()
            print(g['id'],stats[:,:,2].sum(axis=1)/max(1,stats[0,:,0].sum()),flush=True)


if __name__ == '__main__':
    main()
