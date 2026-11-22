"""Protect a sole worker when losing it would cause immediate elimination."""
def _v6_last_worker(obs, config, action, logits):
    me = obs['player']
    own, rival = obs['players'][me], obs['players'][1-me]
    floor = min(config.get('recCollectorCost',30),config.get('recPlanterCost',30))
    if own[0] >= floor or own[3] or len(own[2]) != 1:
        return action
    uid, (pos, cargo, role) = next(iter(own[2].items()))
    choices = _ROLLOUT_LANDING[pos]
    rival_home = set(rival[1].values())
    own_home = set(own[1].values())
    # A landing is safe if every legal opposing movement or recruitment still leaves our unit alive.
    threats = [0]*5
    movement = 1. - config.get('moveCost', 0.)
    for a, target in enumerate(choices):
        load = cargo*(movement if a else 1.)
        if role=='COLLECTOR' and target in rival_home and load <= config.get('smeltCost',0):
            threats[a] += 1
        if target in rival_home and rival[0]>=config.get('recCollectorCost',30) and len(rival[2])<config.get('workerLimit',10):
            threats[a] += 1
        for start, value, kind in rival[2].values():
            if target not in _ROLLOUT_LANDING[start]:
                continue
            # A bait planter is harmless to our collector. Equal-weight collectors both die.
            if role=='PLANTER' or (kind=='COLLECTOR' and value*(movement if target!=start else 1.)<=load):
                threats[a] += 1
    original = WIDX.get(action.get(uid),0)
    if not threats[original] or all(threats):
        return action
    safe = [a for a in range(5) if threats[a]==0]
    chosen = max(safe,key=lambda a:float(logits[pos,a]) + (1.5 if cargo>=floor and choices[a] in own_home else 0.))
    revised = action.copy()
    if chosen:
        revised[uid] = WACT[chosen]
    else:
        revised.pop(uid,None)
    return revised
