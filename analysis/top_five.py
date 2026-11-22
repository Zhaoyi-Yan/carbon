"""Measure both seats of completed games in an official Carbon League replay snapshot.

Run with a snapshot directory containing index.json and replays/*.json.gz. The
default IDs match the 2026-11-22 heavy leaderboard used for the published data.
"""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys


LEADERS = (
    'dd1f117a64c676225f0c6333b1cdd1e7',
    '89e3cd6b39bfbed12b30d45b89683945',
    '1839bb05230ab093ee19355a94276150',
    '40a58f89955a8b792d08b4bf5aeb3e53',
    '182aed58db5991bcd0df65cd41a667be',
)


def opening(actions):
    codes = {'RECCOLLECTOR': 'C', 'RECPLANTER': 'P'}
    return ''.join(sorted(codes[v] for v in (actions or {}).values() if v in codes))


def measure(meta, replay):
    """State t's action is the one that produced that state. Credit tree income there."""
    order = meta['order']
    states = replay['steps']
    config = replay['configuration']
    if set(order) != {'a', 'b'}:
        raise ValueError(f"invalid seat order in {meta['id']}")
    if not 2 <= len(states) <= config['episodeSteps']:
        raise ValueError(f"unsupported game length in {meta['id']}")
    start = states[0][0]['observation']
    layout = [config['size'], start['full_carbon'], [sorted(p[1].values()) for p in start['players']]]
    map_hash = sha256(json.dumps(layout, separators=(',', ':')).encode()).hexdigest()
    size = config['size']
    base_zones = []
    for p in start['players']:
        bases = [divmod(pos, size) for pos in p[1].values()]
        home_distances = []
        for pos in range(size * size):
            row, col = divmod(pos, size)
            home_distances.append(min(min(abs(row-r), size-abs(row-r)) + min(abs(col-c), size-abs(col-c))
                                      for r, c in bases))
        base_zones.append(home_distances)
    stats = [Counter(), Counter()]
    for t in range(1, len(states)):
        old = states[t-1][0]['observation']['players']
        obs = states[t][0]['observation']
        new = obs['players']
        for seat, (p, q) in enumerate(zip(old, new)):
            s = stats[seat]
            opponent = old[1-seat]
            s['plants'] += len(q[3].keys() - p[3].keys() - opponent[3].keys())
            s['seizes'] += len(q[3].keys() & opponent[3].keys())
            s['tree_income'] += sum(obs['trees'][key][1] for key in q[3])
            # The player's unavailable cargo immediately before a worker disappears.
            # Some of it can be recovered after a friendly collision.
            s['lost_worker_cargo'] += sum(p[2][key][1] for key in p[2].keys() - q[2].keys())
            pressure = sum(role == 'COLLECTOR' and load < 1 and base_zones[1-seat][pos] <= 2
                           for pos, load, role in q[2].values())
            s['empty_collectors_near_enemy_base'] += pressure
            if t <= 75:
                s['early_empty_collectors_near_enemy_base'] += pressure
            if t == 75:
                s['collectors75'] = sum(w[2] == 'COLLECTOR' for w in q[2].values())
                s['planters75'] = len(q[2]) - s['collectors75']
            if t in (75, 150, 225, 280):
                s[f'cash{t}'] = q[0]
            if t == len(states)-1:
                s['terminal_cargo'] = sum(w[1] for w in q[2].values())
    sides = []
    for i, slot in enumerate(order):
        score = meta['scores'][slot]
        if abs(replay['rewards'][i] - score) > .011:
            raise ValueError(f"seat/score mismatch in {meta['id']}")
        strategy = meta['strategies'][slot]
        sides.append(dict(strategy, score=score, cash=new[i][0], opening=opening(states[1][i]['action']),
                          **{k: round(v, 2) for k, v in stats[i].items()}))
    return dict(match=meta['id'], game=meta['game'], seed=config['randomSeed'],
                turns=len(states)-1, full_length=len(states) == config['episodeSteps'],
                map_sha256=map_hash, seats=sides)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--leaders', nargs='+', default=LEADERS, help='public strategy IDs')
    args = parser.parse_args()
    index = json.loads((args.snapshot/'index.json').read_text())
    seen_games, seen_maps = set(), set()
    counts = Counter()
    for meta in sorted(index, key=lambda m: (m['id'], m['game'])):
        if not any(v['id'] in args.leaders for v in meta['strategies'].values()):
            continue
        key = meta['id'], meta['game']
        if key in seen_games:
            continue
        seen_games.add(key)
        if any(meta['statuses'].get(k) != 'DONE' for k in ('a', 'b')):
            counts['abnormal finish'] += 1
            continue
        with gzip.open(args.snapshot/meta['file']) as f:
            replay = json.load(f)
        result = measure(meta, replay)
        if result['map_sha256'] in seen_maps:
            counts['repeat map'] += 1
            continue
        seen_maps.add(result['map_sha256'])
        print(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
    print(f'Analysed {len(seen_maps)} different layouts from {len(seen_games)} games. Skips {dict(counts)}',
          file=sys.stderr)


if __name__ == '__main__':
    main()
