"""React to sustained loss of tree income or an unfunded fleet carrying a possible deposit."""
def _v6_response(obs, est, configuration, workers, bases):
    players, step, side = obs['players'], obs['step'], obs['player']
    own, rival = players[side], players[1-side]
    trees = obs.get('trees', {})
    revenue = [sum(trees.get(key, (None,0.))[1] for key in p[3]) for p in players]
    history = _ST['v5_history']
    gap = revenue[1-side]-revenue[side]
    history['tree_income_gap'] = .8*history.get('tree_income_gap',gap)+.2*gap
    deficit = len(rival[3])-len(own[3])
    opening = 22<=step<=94 and deficit>=2 and history['tree_income_gap']>4.
    liquidity = own[0]<configuration.get('recCollectorCost',30) and revenue[side]<2.
    if not opening and not liquidity:
        return workers, bases
    if opening:
        guided, _ = _v6_initiative(obs,est,configuration,workers,bases)
    else:
        guided = workers.copy()
    if liquidity:
        collectors = [(p,v) for p,v,role in own[2].values() if role=='COLLECTOR']
        rivals = [(p,v) for p,v,role in rival[2].values() if role=='COLLECTOR']
        home = list(own[1].values())
        for pos,cargo in collectors:
            if cargo < 65.:
                continue
            lighter = [p for p,v in rivals if v <= cargo]
            threat = _V5_GEOMETRY[lighter].min(0) if lighter else np.full(225,16)
            # Compare an open depot with a nearer one that an opposing empty collector covers.
            dest = np.asarray(_ROLLOUT_LANDING[pos])
            route = np.min(_V5_GEOMETRY[dest][:,home] +
                           1.4*np.maximum(0.,3.-threat[home]),axis=1)
            merit = -.55*(route-route.mean()) - .45*np.maximum(0.,2.-threat[dest])
            merit += np.isin(dest, home) * 1.4
            guided[pos] += np.clip(merit,-1.1,1.1)
            if min(_V5_DIST[pos][h] for h in home) <= 5:
                for escort,load in collectors:
                    if load>=18. or _V5_DIST[escort][pos]>4:
                        continue
                    land=np.asarray(_ROLLOUT_LANDING[escort])
                    blockers=[p for p,v in rivals if v<=cargo and min(_V5_DIST[p][h] for h in home)<=2]
                    if blockers:
                        distance=_V5_GEOMETRY[land][:,blockers].min(1)
                        guided[escort] += .28*np.clip(distance.mean()-distance,-1.,1.)
    history['alternatives'].insert(0,(guided,bases))
    return workers+.35*(guided-workers), bases
