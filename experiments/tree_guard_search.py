"""Export a v5 candidate with coordinated alternatives for contested early-life trees."""
import argparse
from hashlib import sha256
import inspect
from pathlib import Path


BASE_SHA256 = '0483eca2a5e8031270dfb89b87cd5d901ee81f823383f6094f8ed839dec6a7dd'


def _tree_guard_moves(players, me, act0, fixed, carbon, config, step):
    """Propose, but do not force, one planter relocation and one empty-collector escort."""
    if config.get('episodeSteps', 300) - 1 - step < 18:
        return []
    own, enemy = players[me], players[1-me]
    rival_planters = [w[0] for w in enemy[2].values() if w[2] == 'PLANTER']
    if not rival_planters:
        return []
    collectors = [(key, w[0]) for key, w in own[2].items()
                  if w[2] == 'COLLECTOR' and w[1] < 1 and key not in fixed]
    planters = {wid:w[0] for wid,w in own[2].items() if w[2] == 'PLANTER' and wid not in fixed}
    bases = set(own[1].values()) | set(enemy[1].values())
    trees = {tree[0] for p in players for tree in p[3].values()}
    reserved_bases = {pos for key, pos in own[1].items() if key in act0}
    planned = {key:move(w[0], WIDX.get(act0.get(key), 0)) for key,w in own[2].items()}
    rival_dist = lambda pos: min(DIST[pos][q] for q in rival_planters)

    def supply(pos):
        return sum(carbon[q] for q in NB[pos] if q not in trees)

    def clear(pos, key):
        return pos not in reserved_bases and all(d != pos for w,d in planned.items() if w != key)

    # Replicate recruitment costs before deciding if a new planting is affordable.
    balance = own[0]
    for role, action, base_cost, increase in (
        ('COLLECTOR', 'RECCOLLECTOR', 'recCollectorCost', 'recCollectorIncreaseCost'),
        ('PLANTER', 'RECPLANTER', 'recPlanterCost', 'recPlanterIncreaseCost')):
        population = sum(w[2] == role for p in players for w in p[2].values())
        balance -= sum(act0.get(b) == action for b in own[1]) * (config.get(base_cost, 30) + population*config.get(increase, 0))
    cost = config.get('plantCost', 20)
    if len(own[3]) > config.get('playerTreeProtectiveNumber', 0):
        alive = sum(age < config.get('treeLifespan', 50) for p in players for _, age in p[3].values())
        cost += config.get('plantCostInflationRatio', 5) * config.get('plantCostInflationBase', 1.235)**alive

    pending = []
    if balance >= cost:
        for key, pos in planters.items():
            if pos not in bases and pos not in trees and rival_dist(pos) <= 2 and clear(pos,key):
                pending.append((key,pos))
    # Re-evaluate recently planted trees even after the original planter has left.
    young = [pos for pos,age in own[3].values() if age <= 6 and rival_dist(pos) <= 2]
    if not pending and not young:
        return []

    candidates = []
    if pending:
        key, pos = max(pending, key=lambda pair:(act0.get(pair[0]) is None, supply(pair[1])))
        safer = [(k, dst) for k, dst in enumerate(NB[pos], 1)
                 if dst not in bases and dst not in trees and clear(dst,key) and rival_dist(dst) > rival_dist(pos)
                 and all(DIST[dst][w[0]] > 1 for w in enemy[2].values() if w[2] == 'COLLECTOR')]
        if safer:
            k, _ = max(safer, key=lambda item:supply(item[1]) + 12 * rival_dist(item[1]))
            candidates.append(('relocate', dict(act0, **{key:WACT[k]})))

    sites = [(key,pos) for key,pos in pending] + [(None,pos) for pos in young]
    # A collector on a tree or between it and an enemy planter can intercept a seize.
    best = None
    for planter, target in sites:
        for wid, start in collectors:
            if DIST[start][target] > 3:
                continue
            for k, dst in enumerate([start]+NB[start]):
                if dst in bases or not clear(dst,wid) or (planter is not None and dst == target):
                    continue
                if DIST[dst][target] > 1 or (k == 0 and act0.get(wid) is None and (planter is None or planter not in act0)):
                    continue
                if any(DIST[dst][w[0]] <= 1 for w in enemy[2].values() if w[2] == 'COLLECTOR' and w[1] <= own[2][wid][1]):
                    continue
                # The v5 planter's previous destination stops being occupied if it now stays.
                defended = dict(act0)
                if planter is not None:
                    defended.pop(planter, None)
                if k:
                    defended[wid] = WACT[k]
                else:
                    defended.pop(wid, None)
                if defended == act0:
                    continue
                quality = (supply(target) - 15*DIST[dst][target] - 20*rival_dist(dst)
                           + 18*(dst in rival_planters) - 8*DIST[start][target])
                if best is None or quality > best[0]:
                    best = quality, defended
    if best is not None:
        candidates.insert(0, ('escort', best[1]))
    return candidates


def export(template, output):
    source = template.read_bytes()
    if sha256(source).hexdigest() != BASE_SHA256:
        raise ValueError('The experiment requires the v5-tree-value template')
    text = source.decode()
    anchor = '    for T, cap in _S_GEN:'
    code = '''    # Preserve the original policy and regular proposals within the measured width.
    if most >= 3:
        limit = min(2, (most-1)//2)
        for _, option in _tree_guard_moves(players, me, act0, fixed or {}, est, configuration, step):
            marker = tuple(sorted(option.items()))
            if marker not in keys:
                keys.add(marker)
                cands.append(option)
                if len(cands) - 1 >= limit:
                    break
'''
    if text.count(anchor) != 1 or text.count('def _search(') != 1:
        raise ValueError('Unexpected v5 search entry')
    text = text.replace('def _search(', inspect.getsource(_tree_guard_moves) + '\n\ndef _search(', 1)
    text = text.replace(anchor, code + anchor, 1)
    compile(text, str(output), 'exec')
    with output.open('x', encoding='utf-8') as destination:
        destination.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parents[1]/'submission.py')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.base, args.output)
