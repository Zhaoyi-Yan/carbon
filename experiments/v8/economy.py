"""Build independent tree-payback and extra-harvest policy variants from frozen v7."""
import argparse
import hashlib
import inspect
from pathlib import Path

BASELINE = "4894fe44c601b7360d37de2eed928802c73884f0929a5897604d174e333e02e6"


def _v8_tree_payback(obs, cfg, stock, logits):
    side = obs['player']
    own, foe = obs['players'][side], obs['players'][1-side]
    planters = [p for p, _, role in own[2].values() if role == 'PLANTER']
    if not planters:
        return
    turns = cfg['episodeSteps'] - 1 - obs['step']
    lifespan = cfg['treeLifespan']
    forest = {p: (s, age) for s, player in enumerate(obs['players'])
              for p, age in player[3].values() if age < lifespan}
    depots = set(own[1].values()) | set(foe[1].values())
    rivals = list(foe[2].values())
    thieves = [p for p, _, role in rivals if role == 'PLANTER']
    guards = [p for p, load, role in own[2].values() if role == 'COLLECTOR' and load < 35]
    price = cfg['plantCost']
    if len(own[3]) > cfg['playerTreeProtectiveNumber']:
        price += cfg['plantCostInflationRatio'] * cfg['plantCostInflationBase'] ** len(forest)
    cache = {}

    def value(p, delay):
        if (p, delay) in cache:
            return cache[p, delay]
        owner, age = forest.get(p, (None, 0))
        if p in depots or owner == side:
            return 0.
        taking = owner is not None
        cost = cfg['seizeCost'] if taking else price
        if cost > own[0]:
            return 0.
        life = min(turns - delay, lifespan - age - delay if taking else lifespan)
        thief = min((_V5_DIST[p][q] for q in thieves), default=15)
        guard = min((_V5_DIST[p][q] for q in guards), default=15)
        # Payback ends early if a rival planter can arrive before our guard.
        horizon = max(0, min(life - int(not taking), 30, 6 + 3 * thief + (8 if guard < thief else 0)))
        income = 0.
        for q in NB[p]:
            if q in forest or q in depots:
                continue
            trees = 1 + sum(n != p and n in forest for n in NB[q])
            retention = (1 - cfg['cellAbsorptionRate'] * trees) * .985
            income += max(0., float(stock[q])) * cfg['cellAbsorptionRate'] * (1-retention**horizon) / (1-retention)
        tax = sum(min(cargo, 200.) * cfg['collectorAbsorptionRate']
                  for q, cargo, role in rivals if role == 'COLLECTOR' and (q == p or q in NB8[p]))
        income += min(2., horizon) * tax
        # A seizure also denies the previous owner's next receipts.
        net = (income * (1.65 if taking else 1.) - cost) / (1 + .25 * delay)
        cache[p, delay] = net
        return net

    for p in planters:
        if p in depots or p in forest and forest[p][0] == side:
            stay = 0.
        else:
            stay = value(p, 0)
        logits[p, 0] += max(-2.6, min(1.5, stay / 40.))
        for action, q in enumerate(NB[p], 1):
            # Moving costs a turn before the planter can acquire the target.
            net = value(q, 1)
            if net > 0 and all(_V5_DIST[q][z] > 1 for z, _, k in rivals if k == 'COLLECTOR'):
                logits[p, action] += min(.85, net / 75.)


def _v8_extra_harvest(obs, cfg, stock, logits):
    side, step = obs['player'], obs['step']
    us, foe = obs['players'][side], obs['players'][1-side]
    homes = set(us[1].values())
    if not homes or us[0] < cfg['recCollectorCost']:
        return
    collectors = [(q, w) for q, w, k in us[2].values() if k == 'COLLECTOR']
    rate = max(0., cfg['initialCollectRate'] - len(collectors) * cfg['collectDecreaseRate'])
    home = _fc_home(tuple(sorted(homes)))
    rival = [(p, w) for p, w, k in foe[2].values() if k == 'COLLECTOR']
    trees = {p for pl in obs['players'] for p, age in pl[3].values() if age < cfg['treeLifespan']}
    tax_trees = {p for p, age in foe[3].values() if age < cfg['treeLifespan']}
    blocked = trees | homes | set(foe[1].values())
    remaining = cfg['episodeSteps'] - 1 - step
    for p, cargo in collectors:
        d = home[p]
        if p in blocked or not 1 <= d <= 5 or cargo < 24 or remaining < d + 3:
            continue
        crop = rate * max(0., float(stock[p]))
        if crop < 6:
            continue
        lethal = [q for q, v in rival if v <= cargo + crop]
        if any(_V5_DIST[p][q] <= 2 for q in lethal):
            continue
        # Require a safe first exit and a second shortening step after the harvest.
        exits = [q for q in NB[p] if home[q] == d-1 and
                 (q in homes or all(_V5_DIST[q][z] > 2 for z in lethal))]
        if not any(q in homes or any(home[n] < home[q] and
                   (n in homes or all(_V5_DIST[n][z] > 2 for z in lethal))
                   for n in NB[q]) for q in exits):
            continue
        if any(q != p and _V5_DIST[p][q] == 1 and v > cargo + 25 for q, v in rival):
            continue
        tax = cfg['collectorAbsorptionRate'] * sum(q in tax_trees for q in (p, *NB8[p]))
        # Price the extra turn by the cargo already in transit and the best next harvest.
        next_crop = max((rate * max(0., float(stock[q])) for q in exits if q not in blocked), default=0.)
        net = crop - tax * (cargo + crop) - cargo / (7 + 2*d) - .22 * next_crop
        if net > 3:
            logits[p, 0] += min(2., (net-3) / 8.)


def build(source, arm):
    if hashlib.sha256(source.encode()).hexdigest() != BASELINE:
        raise ValueError("Expected the frozen yan-v7 release")
    names = {'trees': [_v8_tree_payback], 'cargo': [_v8_extra_harvest],
             'combined': [_v8_tree_payback, _v8_extra_harvest]}[arm]
    anchor = 'def _act(obs, configuration=None, t0=None):'
    hook = '        lw=_selfcredit_adjust(x,lw)'
    if source.count(hook) != 1 or source.count(anchor) != 1:
        raise ValueError("Missing unique search hook")
    text = source.replace(anchor, '\n\n'.join(inspect.getsource(fn) for fn in names) + '\n\n' + anchor)
    calls = ['        '+fn.__name__+'(obs, configuration, est, lw)' for fn in names]
    return text.replace(hook, '\n'.join([hook, *calls]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=('trees', 'cargo', 'combined'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    candidate = build((root / 'yan-v7.py').read_text(), args.arm)
    compile(candidate, str(args.output), 'exec')
    if len(candidate.encode()) > 10_000_000:
        raise ValueError("Candidate exceeds competition size")
    args.output.write_text(candidate)
