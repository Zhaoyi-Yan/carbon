"""Choose roots from public v7 replays using only information visible at the decision."""
import hashlib
import json

from run import ROOT
from starvation_replay import distance, read_replay

STRATA = ('cash', 'cargo', 'tree', 'ordinary')
WINDOWS = ((45, 110), (110, 190), (190, 270))


def classify(players, seat, cfg):
    own, foe = players[seat], players[1-seat]
    carriers = [p for p in own[2].values() if p[2] == 'COLLECTOR' and p[1] >= 75]
    exposed = [c for pos, c, _ in carriers
               if any(k == 'COLLECTOR' and v < c and distance(pos, q) <= 2
                      for q, v, k in foe[2].values())]
    threatened = [uid for uid, (pos, age) in own[3].items()
                  if age < cfg['treeLifespan'] - 2 and foe[0] >= cfg['seizeCost']
                  and any(k == 'PLANTER' and distance(pos, q) <= 1 for q, _, k in foe[2].values())]
    flags = {'cash': own[0] < cfg['recCollectorCost'], 'cargo': bool(exposed), 'tree': bool(threatened)}
    flags['ordinary'] = not any(flags.values())
    return flags, dict(cash=own[0], margin=round(own[0]-foe[0], 2),
                       exposed_cargo=round(sum(exposed), 2), threatened_trees=threatened)


def select(data):
    slots = [(s, phase, (i+phase)%2) for phase in range(3) for i, s in enumerate(STRATA)]
    choices = {slot: [] for slot in slots}
    for g in json.loads((data/'metrics.json').read_text()):
        if g['group'] != 'v7_top5' or g['me'] != 'yan-QQ-v7':
            continue
        file = f"{g['id']}-{g['game']}.html.gz"
        replay = read_replay(data/file)
        cfg = replay['configuration']
        for category, phase, seat in slots:
            if g['seat'] != seat:
                continue
            starts = []
            window = ((15, 40), (35, 75), (60, 180))[phase] if category == 'cash' else WINDOWS[phase]
            for turn in range(*window):
                ps = replay['steps'][turn][0]['observation']['players']
                flags, _ = classify(ps, seat, cfg)
                if flags[category]:
                    starts.append(turn)
            if not starts:
                continue
            def order(t):
                return hashlib.sha256(f'fork-v2/{category}/{file}/{t}'.encode()).hexdigest()
            starts.sort(key=order)
            choices[category, phase, seat].append((order(starts[0]), dict(
                file=file, layout=g['layout'], opponent=g['opp'], seed=cfg['randomSeed'],
                source_sha256=hashlib.sha256((data/file).read_bytes()).hexdigest(),
                category=category, phase=phase, seat=seat, turns=starts)))
    # Allocate the least flexible slot first, preserving six roots per seat and unique layouts.
    ordered = sorted(slots, key=lambda s: len(choices[s]))
    def assign(index, used):
        if index == len(ordered):
            return {}
        slot = ordered[index]
        for _, case in sorted(choices[slot], key=lambda pair: pair[0]):
            if case['layout'] in used:
                continue
            rest = assign(index+1, used | {case['layout']})
            if rest is not None:
                rest[slot] = case
                return rest
        return None
    allocation = assign(0, set())
    if allocation is None:
        raise ValueError(f'Insufficient distinct maps: {[(s,len(choices[s])) for s in slots]}')
    return [allocation[s] for s in slots]
