"""Check exported joint proposals against recorded public states and frozen ally moves."""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import validate_actions


def load_model(candidate):
    spec = importlib.util.spec_from_file_location('checked_tree_guard', candidate)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def check(candidate, replay_paths):
    model = load_model(candidate)
    stats = Counter()
    for path in replay_paths:
        with gzip.open(path) as f:
            replay = json.load(f)
        frames = replay['steps']
        cfg = replay['configuration']
        trackers = [model.BeliefTracker(model._Bel, i) for i in range(2)]
        for t, (frame, after) in enumerate(zip(frames, frames[1:])):
            players = frame[0]['observation']['players']
            for side, tracker in enumerate(trackers):
                own, enemy = players[side], players[1-side]
                carbon, _ = tracker.step(players,t,frame[side]['observation']['carbon'])
                base = after[side]['action'] or {}
                tracker.set_actions(players,base)
                # Freeze a rotating subset, in addition to all laden collector actions.
                fixed = {w:base.get(w) for i,(w,unit) in enumerate(own[2].items())
                         if unit[1] >= 1 or (i+t)%5 == 0}
                proposals = model._tree_guard_moves(players,side,base,fixed,carbon,cfg,t)
                proposals = [(kind, a) for kind,a in proposals if a != base]
                stats['states'] += 1
                stats['states_with_alternative'] += bool(proposals)
                if cfg['episodeSteps']-1-t < 18:
                    assert not proposals
                for kind, action in proposals:
                    validate_actions(action,own)
                    changed = {k for k in set(action)|set(base) if base.get(k)!=action.get(k)}
                    assert changed <= own[2].keys() and changed.isdisjoint(fixed) and 1 <= len(changed) <= 2
                    assert all(action.get(k)==base.get(k) for k in own[1])
                    dest = {wid:model.move(w[0],model.WIDX.get(action.get(wid),0)) for wid,w in own[2].items()}
                    reserved = {pos for k,pos in own[1].items() if k in base}
                    for wid in changed:
                        assert dest[wid] not in reserved
                        assert dest[wid] not in enemy[1].values()
                        assert all(dest[w]!=dest[wid] for w in dest if w!=wid)
                        assert own[2][wid][2]=='PLANTER' or own[2][wid][1]<1
                    stats[kind] += 1
                    stats['changed_two_workers'] += len(changed)==2
    return dict(stats)


def trace_search(candidate, path):
    """Use recorded observations and track which added root moves the critic actually plays."""
    model = load_model(candidate)
    with gzip.open(path) as stream:
        replay = json.load(stream)
    frames = replay['steps']
    side = 0
    tracker = model.BeliefTracker(model._Bel, side)
    rng = model.np.random.RandomState(model._SEED)
    original = model._tree_guard_moves
    proposed = {}

    def tracked(*args):
        moves = original(*args)
        for kind, action in moves:
            proposed[tuple(sorted(action.items()))] = kind
        return moves

    model._tree_guard_moves = tracked
    model._S_REC = []
    stats = Counter(calls=0, offered=0, selected=0)
    for turn in range(200):
        observation = frames[turn][side]['observation']
        players = observation['players']
        if turn >= 50 and turn % 3 == 0:
            model._ST.update(tr=tracker, prev=frames[turn-1][0]['observation']['players'], rng=rng)
            proposed.clear()
            model._S_REC.clear()
            model.agent(observation, replay['configuration'])
            stats['calls'] += 1
            if model._S_REC:
                _, _, candidates, _, _, best = model._S_REC[-1]
                offered = {index: proposed[marker] for index, action in enumerate(candidates)
                           if index and (marker := tuple(sorted(action.items()))) in proposed}
                stats['offered'] += bool(offered)
                if best in offered:
                    stats['selected'] += 1
                    stats[offered[best]] += 1
        else:
            tracker.step(players, turn, observation['carbon'])
        tracker.set_actions(players, frames[turn+1][side]['action'])
    return dict(replay=path.name, seat=side, **stats)


def loss_windows(directory):
    """When did enemy planters first come within two steps of a subsequently lost tree?"""
    rows = [json.loads(line) for line in (ROOT/'analysis'/'yan-v2-games.jsonl').read_text().splitlines()]
    results = {}
    for row in rows:
        path = directory/f"{row['match']}-{row['game']}.json.gz"
        if row['won'] or not path.exists():
            continue
        with gzip.open(path) as stream:
            replay = json.load(stream)
        frames = [f[0]['observation']['players'] for f in replay['steps']]
        size = replay['configuration']['size']
        me = row['seat']

        def near(pos, t):
            for cell, _, role in frames[t][1-me][2].values():
                dr, dc = abs(cell//size-pos//size), abs(cell%size-pos%size)
                if role == 'PLANTER' and min(dr, size-dr) + min(dc, size-dc) <= 2:
                    return True
            return False

        births, seized = {}, set()
        counts = Counter(at_plant=0, young=0, later=0)
        for t, players in enumerate(frames):
            for owner, player in enumerate(players):
                for tree in player[3]:
                    births.setdefault(tree, (owner, t))
            if t == 0:
                continue
            lost = players[1-me][3].keys() & frames[t-1][me][3].keys()
            for tree in lost - seized:
                seized.add(tree)
                origin, birth = births[tree]
                if origin != me or t-birth > 18 or birth > len(frames)-1-18:
                    continue
                pos = players[1-me][3][tree][0]
                if near(pos, birth-1):
                    counts['at_plant'] += 1
                elif any(frames[k][me][3][tree][1] <= 6 and near(pos, k) for k in range(birth,t)):
                    counts['young'] += 1
                else:
                    counts['later'] += 1
        assert sum(counts.values()) == row['early_seized']
        results[row['match']] = dict(counts)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--trace-search', action='store_true',
                        help='Also replay 50 searches on the first game, seat 0.')
    parser.add_argument('--loss-replays', type=Path, help='Directory containing the full yan-v2 game files.')
    args = parser.parse_args()
    paths = sorted((ROOT/'replays'/'apollo').glob('*.json.gz'))[:4]
    if len(paths)!=4: raise ValueError('Four archived Apollo game files required')
    result = dict(replays=[p.name for p in paths], **check(args.candidate,paths))
    if args.trace_search:
        result['search_trace'] = trace_search(args.candidate, paths[0])
    if args.loss_replays:
        result['loss_windows'] = loss_windows(args.loss_replays)
    print(json.dumps(result, indent=2))
