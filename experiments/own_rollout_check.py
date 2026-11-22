"""Compare root rollout action priors to the decisions of the archived online yan-v2."""
import argparse
import gzip
import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import numpy as np


YAN_V2 = '03b736af6aaf915fe90a48b0127e5cac'
ROLES = ('COLLECTOR', 'PLANTER', 'BASE')


def add_batch(candidate, rows, metrics):
    if not rows:
        return
    X = np.stack([row[0] for row in rows])
    net = candidate._net()
    for backend, stride in enumerate((0, 1)):
        workers, bases, _ = net(X, candidate._SID, actor_stride=stride)
        for j, (_, player, action) in enumerate(rows):
            for k, role in enumerate(ROLES):
                if role == 'BASE':
                    if len(player[2]) >= 10 or player[0] < 30:
                        continue
                    items = player[1].items()
                    logits, names = bases[j], candidate.BIDX
                else:
                    items = ((uid, pos) for uid, (pos, _, r) in player[2].items() if role == r)
                    logits, names = workers[j], candidate.WIDX
                choices = [(pos, names[action.get(uid) or 'NONE']) for uid, pos in items]
                if not choices:
                    continue
                positions, labels = map(np.array, zip(*choices))
                z = logits[positions] / candidate._TEMP
                z -= z.max(axis=1, keepdims=True)
                nll = np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(labels)), labels]
                metrics[backend, k] += [len(labels), (z.argmax(axis=1) == labels).sum(), nll.sum()]


def audit(candidate, snapshot):
    index = json.loads((snapshot/'index.json').read_text())
    results, seen = [], set()
    for game in sorted(index, key=lambda g: (g['id'], g['game'])):
        slots = [k for k,v in game['strategies'].items() if v['id'] == YAN_V2]
        path = snapshot/game['file']
        if not slots or not path.is_file():
            continue
        seat = game['order'].index(slots[0])
        with gzip.open(path) as f:
            replay = json.load(f)
        steps = replay['steps']
        if len(steps) != 300 or replay['statuses'] != ['DONE', 'DONE']:
            continue
        if any(abs(replay['rewards'][i] - game['scores'][k]) >= .02 for i,k in enumerate(game['order'])):
            raise ValueError(f"Seat/score mismatch for {game['id']}")
        initial = steps[0][0]['observation']
        layout = [15, initial['full_carbon'], [sorted(p[1].values()) for p in initial['players']]]
        digest = sha256(json.dumps(layout, separators=(',', ':')).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        tracker = candidate.BeliefTracker(candidate._Bel, seat)
        previous, pending = None, []
        stats = np.zeros((2, len(ROLES), 3))
        for t, (cur, nxt) in enumerate(zip(steps, steps[1:])):
            players = cur[0]['observation']['players']
            estimate, _ = tracker.step(players, t, cur[seat]['observation']['carbon'])
            action = nxt[seat]['action'] or {}
            if t % 3 == 2:
                # Simulated rollout features assume known carbon at the observer's estimate.
                x = candidate._quant(candidate.encode(players, t, seat, estimate, [1.]*225, previous))
                pending.append((x, players[seat], action))
                if len(pending) == 12:
                    add_batch(candidate, pending, stats)
                    pending.clear()
            tracker.set_actions(players, action)
            previous = players
        add_batch(candidate, pending, stats)
        results.append(dict(id=game['id'], map=digest, seed=replay['configuration']['randomSeed'],
                            seat=seat, metrics=stats.tolist()))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path, help='Directory containing index.json and replays/')
    parser.add_argument('--candidate', required=True, type=Path, help='Output of own_rollout.py')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('rollout_candidate', args.candidate)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    results = audit(candidate, args.snapshot)
    if not results:
        parser.error('No complete yan-v2 games with local full replays in this snapshot')
    total = np.sum([r['metrics'] for r in results], axis=0)
    print('maps', len(results), 'decisions', total[0, :, 0],
          'top1', total[:, :, 1]/total[:, :, 0], 'NLL', total[:, :, 2]/total[:, :, 0])
    args.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
