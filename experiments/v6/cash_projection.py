"""Forecast realizable deposits and tree income from the visible positions of both teams."""
def _v6_assets(players, tree_meta, side, turns):
    own, rival = players[side], players[1-side]
    distance = _V5_DIST
    home = _v5_home(tuple(own[1].values()))
    away = _v5_home(tuple(rival[1].values()))
    enemy_collectors = [(p, v) for p, v, k in rival[2].values() if k == 'COLLECTOR']
    enemy_planters = [p for p, _, k in rival[2].values() if k == 'PLANTER']
    collectors = [(p, v) for p, v, k in own[2].values() if k == 'COLLECTOR']
    planters = [p for p, _, k in own[2].values() if k == 'PLANTER']
    light = [p for p, v in collectors if v < 25]
    tree_yield = secure_yield = stealable_yield = near_end_yield = 0.
    for key, (pos, age) in own[3].items():
        life = max(0, min(turns, 50 - age))
        rate = min(20., tree_meta.get(key, (None, 0.))[1])
        # A tree depletes its neighboring cells. One high-yield turn does not last 50 turns.
        projected = rate * min(life, 24.) / (1. + .024 * min(life, 24.))
        challenger = min((distance[pos][q] for q in enemy_planters), default=16)
        guard = min((distance[pos][q] for q in light), default=16)
        tree_yield += projected
        secure_yield += projected * min(1., challenger / 4.)
        stealable_yield += projected * (challenger <= 2 and challenger < guard)
        near_end_yield += rate * min(8, life)
    cargo = safe_cargo = exposed_cargo = hunter_pressure = escorts = ready = 0.
    for pos, load in collectors:
        reach = home[pos]
        danger = min((distance[pos][q] for q, v in enemy_collectors if v < load), default=16)
        if reach <= turns:
            cargo += load
            safe_cargo += load / (1. + .12 * reach) * min(1., .35 + .20 * danger)
            exposed_cargo += load * (danger <= 2)
        if load < 25:
            hunter_pressure += max(0., 4. - away[pos])
            escorts += sum(distance[pos][p] <= 2 for p in planters)
        ready += min(1., load / 70.) / (1. + reach)
    pressure = sum(min((distance[p][q] for p, v in enemy_collectors if v < 25), default=16) <= 2
                   for q in own[1].values())
    front = sum(max(0., 6. - away[p]) for p in planters)
    return np.array([tree_yield, secure_yield, stealable_yield, near_end_yield, cargo, safe_cargo,
                     exposed_cargo, len(own[3]), len(collectors), len(planters), hunter_pressure,
                     escorts, ready, pressure, front, min(own[0], 90.), min(own[0], 30.),
                     float(not collectors), float(not planters)], dtype=float)


def _v6_position(players, trees, step, side):
    turns = max(0, LAST_TURN + 1 - step)
    diff = _v6_assets(players, trees, side, turns) - _v6_assets(players, trees, 1-side, turns)
    early = max(0., 1. - step / 115.)
    late = max(0., 1. - turns / 75.)
    return np.concatenate((diff, diff * early, diff * late))


def _v6_projection(board, me):
    state = board.get('_st')
    trees = state.trees_meta if state is not None else board.get('trees', {})
    players, step = board['players'], board['step']
    funds = players[me][0] - players[1-me][0]
    change = _v6_position(players, trees, step, me) @ _V6_INCOME
    turns = max(0, LAST_TURN + 1 - step)
    return (funds + min(1., turns / 6.) * change) / 3000.
