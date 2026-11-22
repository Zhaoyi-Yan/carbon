"""Look for coordinated one-turn alternatives before expanding the regular search."""
def _v5_countermove(root, config, mine, fixed, worker_logits, base_logits, seed, t0):
    players, side = root['players'], root['player']
    rival = 1-side
    cash, bases, units, _ = players[side]
    if not units:
        return None
    rng = np.random.RandomState(seed+9341)
    ow, ob = _ST['v5_history']['opponent']
    replies = [_sample(players,rival,ow,ob,rng,root['step']) for _ in range(2)]
    if replies[0] == replies[1]:
        replies.pop()
    def score(act):
        values=[]
        for response in replies:
            aa=[act,response] if side==0 else [response,act]
            values.append(_v5_value(_turn(root,config,aa),side))
        return .7*sum(values)/len(values)+.3*min(values)
    def landing(wid,cmd):
        return _ROLLOUT_LANDING[units[wid][0]][WIDX.get(cmd,0)]
    def change(act, uid, cmd):
        updated=act.copy()
        if cmd:
            updated[uid]=cmd
        else:
            updated.pop(uid,None)
        return updated
    base_score=score(mine)
    # Include single-unit alternatives even if other units have more uncertain priors.
    options=[]
    for uid,(pos,load,kind) in units.items():
        if uid in fixed:
            continue
        initial=WIDX.get(mine.get(uid),0)
        rivals=[w for w in players[rival][2].values() if DIST[pos][w[0]]<=2]
        impact=max([0.]+[max(load,v[1])/100. for v in rivals])
        priority=(1. if kind=='PLANTER' else 0.)+min(3.,impact)
        for k in np.argsort(-worker_logits[pos]):
            if k==initial:
                continue
            diff=float(worker_logits[pos,k]-worker_logits[pos,initial])
            options.append((priority+.3*max(-10.,diff), uid, WACT[k] if k else None, max(-8.,diff)))
    # Spending and second-base recruitment can change a whole game's economy.
    for uid,pos in bases.items():
        initial=BIDX.get(mine.get(uid),0)
        for k in range(3):
            if k!=initial and (k==0 or cash>=30 and len(units)<config.get('workerLimit',10)):
                diff=float(base_logits[pos,k]-base_logits[pos,initial])
                options.append((1.+.15*max(-10.,diff),uid,BACT[k] if k else None,max(-8.,diff)))
    options.sort(key=lambda row:-row[0])
    reserved={pos for bid,pos in bases.items() if bid in mine}
    destination={uid:landing(uid,mine.get(uid)) for uid in units}
    best,best_score=None,base_score
    start=_time.perf_counter()
    for _,uid,cmd,prior in options[:36]:
        if _time.perf_counter()-start>.060 or _time.perf_counter()-t0>.27:
            break
        if uid in units:
            dest=landing(uid,cmd)
            if dest in reserved or any(p==dest for other,p in destination.items() if other!=uid):
                continue
        elif cmd and bases[uid] in destination.values():
            continue
        action=change(mine,uid,cmd)
        estimate=score(action)+.0025*prior
        if estimate>best_score+.0003:
            best,best_score=action,estimate
    return best
