"""Add a coordinated planting and escort proposal using only the current team's estimates."""
def _v6_initiative(obs, carbon, configuration, workers, bases):
    players, me, step = obs['players'], obs['player'], obs['step']
    own, rival = players[me], players[1-me]
    left = max(0, LAST_TURN + 1 - step)
    revised = workers.copy()
    if left < 4 or not own[2]:
        return revised, bases
    stock = np.maximum(np.asarray(carbon, dtype=float), 0.)
    life = configuration.get('treeLifespan', 50)
    absorb = configuration.get('cellAbsorptionRate', .0375)
    tax = configuration.get('seizeCost', 20.)
    own_bases = tuple(own[1].values())
    enemy_bases = tuple(rival[1].values())
    home = np.asarray(_v5_home(own_bases))
    enemy_home = np.asarray(_v5_home(enemy_bases))
    enemies = [(p, v) for p, v, role in rival[2].values() if role == 'COLLECTOR']
    planters = [p for p, _, role in own[2].values() if role == 'PLANTER']
    enemy_planters = [p for p, _, role in rival[2].values() if role == 'PLANTER']
    geom = _V5_GEOMETRY
    def nearest(sites):
        return geom[sites].min(0) if sites else np.full(225, 16)
    risk = nearest([p for p, v in enemies])
    active = sum(age < life for pl in players for _, age in pl[3].values())
    cost = configuration.get('plantCost', 20.)
    if len(own[3]) > configuration.get('playerTreeProtectiveNumber', 0):
        cost += configuration.get('plantCostInflationRatio', 0.) * configuration.get('plantCostInflationBase', 1.)**active
    neighbors = _V5_ADJ
    cover = np.zeros(225)
    for pl in players:
        for p, age in pl[3].values():
            cover[neighbors[p]] += min(1., (life-age)/12.)
    # Existing trees drain adjacent cells, so discount the carbon shared with them.
    harvest = (stock / (1. + .65*cover))[neighbors].sum(1) * absorb
    potential = np.maximum(0., harvest * min(20., left*.72) - cost)
    if own[0] < cost:
        potential *= .22
    potential[list(own_bases + enemy_bases)] = 0.
    for pos, age in own[3].values():
        potential[pos] = 0.
    for key, (pos, age) in rival[3].items():
        rate = obs.get('trees', {}).get(key, (None,0.))[1]
        remaining = min(left, max(0., life-age))
        # Transfer of a producing tree changes both teams' income.
        potential[pos] = max(0., 2.*max(harvest[pos], min(rate,15.)) * min(remaining, 22.) / (1.+.026*min(remaining,22.)) - tax)
        if own[0] < tax:
            potential[pos] *= .22
    potential *= np.clip(.38+.19*risk, .38, 1.)
    occupied_targets = set()
    for pos in sorted(planters, key=lambda q:-float(potential[q])):
        local = potential.copy()
        if occupied_targets:
            local[list(occupied_targets)] *= .40
        destinations = np.array(_ROLLOUT_LANDING[pos])
        reach = geom[destinations]
        opportunities = local[None,:] / (1. + .52*reach)**1.5
        best = opportunities.max(1)
        # Stationary worker can plant or seize immediately; moving onto a target takes an additional turn.
        best[0] = max(best[0]*.72, local[pos]*1.13)
        best -= np.maximum(0.,2.-risk[destinations])*12.
        adjusted = np.clip((best-best.mean())/62., -.90, .90)
        revised[pos] += adjusted
        target = int(np.argmax(local / (1.+.52*geom[pos])**1.5))
        if geom[pos,target] <= 3 and local[target] > 30.:
            occupied_targets.add(target)
    for pos, cargo, role in own[2].values():
        if role != 'COLLECTOR':
            continue
        dest = np.array(_ROLLOUT_LANDING[pos])
        if cargo <= 12. and enemy_planters:
            # An empty collector can support its planter while contesting an approaching rival.
            targets = [(q, 1. if any(geom[p,q]<=3 for p in planters) else .43) for q in enemy_planters]
            pressure = np.max([w*np.maximum(0,4-geom[dest,q]) for q,w in targets],axis=0)
            revised[pos] += np.clip(.26*(pressure-pressure.mean()),-.5,.5)
        if cargo > 20.:
            revised[pos,enemy_home[dest]==0] -= min(2.5,cargo/48.)
            if own[0] < 65. and home[pos]<=5:
                revised[pos] += .32*np.sign(home[pos]-home[dest])
    return revised, bases
