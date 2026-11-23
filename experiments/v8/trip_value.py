"""Compare cash from returning today with one more harvest and the trips still possible."""
import hashlib
import inspect

from economy import BASELINE
from trip_history import _v8_observe_trips


def _v8_after_deposit(trips, port, remaining, step):
    # With a long horizon the marginal value approaches carbon per worker-turn.
    # Near the finish it jumps at specific achievable delivery times.
    samples = list(_V8_TRIP_PRIOR)
    recent = [(duration, credit) for t, origin, duration, credit in trips[-36:]
              if t >= step-85 and origin == port]
    samples.extend(recent[-12:])
    value = [0.] * (remaining+1)
    for h in range(1, remaining+1):
        value[h] = sum(credit + value[h-duration] for duration, credit in samples
                       if 0 < duration <= h) / len(samples)
    return value


def _v8_patch_trip(value, horizon, patches, current, picked, rate):
    # Spend the available time on a nearby known patch, then resume ordinary trips.
    # Both travel legs, integer harvest turns and returning by the deadline are required.
    best = value[horizon]
    for q, distance, stock in patches:
        budget = horizon-2*distance
        carbon = stock-picked if q==current else stock
        if budget<=0 or carbon<=0:
            continue
        harvested = 0.
        for turns in range(1,min(5,budget)+1):
            harvested += (carbon-harvested)*rate
            best = max(best, harvested + value[budget-turns])
    return best


def _v8_trip_offer(obs, cfg, stock, base, logits, locked, trips):
    side, t = obs['player'], obs['step']
    if t < 45:
        return None
    us, foe = obs['players'][side], obs['players'][1-side]
    ports = tuple(us[1].values())
    if not ports:
        return None
    remain = cfg['episodeSteps'] - 1 - t
    carriers = [(uid,p,load) for uid,(p,load,k) in us[2].items() if k=='COLLECTOR']
    rate = max(0.,cfg['initialCollectRate']-len(carriers)*cfg['collectDecreaseRate'])
    trees = {q for pl in (us,foe) for q,age in pl[3].values() if age<cfg['treeLifespan']}
    foreign = {q for q,age in foe[3].values() if age<cfg['treeLifespan']}
    barred = trees | set(ports) | set(foe[1].values())
    enemies = [(q,load) for q,load,k in foe[2].values() if k=='COLLECTOR']
    busy = {uid:move(p,WIDX.get(base.get(uid),0)) for uid,(p,_,_) in us[2].items()}
    hiring = {q for uid,q in us[1].items() if base.get(uid)}
    home = _fc_home(tuple(sorted(ports)))
    cached = {}
    best_gain, edit = 0., None
    for uid,p,load in carriers:
        if uid in locked or p in barred or load<24 or home[p]>5:
            continue
        orig = WIDX.get(base.get(uid),0)
        if orig and home[busy[uid]]>=home[p]:
            continue
        reserved = hiring | {q for u,q in busy.items() if u!=uid}
        crop = rate*max(0.,float(stock[p]))
        threat = [q for q,v in enemies if v<=load+crop]
        routes = []
        for a,q in enumerate(NB[p],1):
            if orig and a!=orig or q in reserved or home[q] >= home[p]:
                continue
            if q not in ports and any(_V5_DIST[q][z]<=1 for z in threat):
                continue
            for port in ports:
                d = _V5_DIST[p][port]
                if d!=home[p] or _V5_DIST[q][port]!=d-1:
                    continue
                if q!=port and not any(_V5_DIST[n][port]==d-2 and n not in foe[1].values() and
                                        (n==port or all(_V5_DIST[n][z]>1 for z in threat)) for n in NB[q]):
                    continue
                routes.append((a,q,port,d))
        if not routes:
            continue
        # Keep tactical chases and contested friendly-unit landings under v7 control.
        if p in reserved or any(_V5_DIST[p][z]<=1 for z,_ in enemies):
            continue
        best_now = best_later = -1.
        return_move = None
        for a,q,port,d in routes:
            if remain<d:
                continue
            if port not in cached:
                # Occupancy is checked per carrier below so its own current patch remains eligible.
                patches = [(n,_V5_DIST[port][n],max(0.,float(stock[n]))) for n in range(225)
                           if n not in barred and 1<=_V5_DIST[port][n]<=4 and stock[n]>0 and
                           all(_V5_DIST[n][z]>1 for z,v in enemies if v<40)]
                cached[port] = (_v8_after_deposit(trips,port,remain,t),patches)
            value, patches = cached[port]
            occupied = {pos for other,pos,_ in carriers if other!=uid}
            patches_free = [item for item in patches if item[0] not in occupied]
            left = remain-d
            now = load + _v8_patch_trip(value,left,patches_free,p,0.,rate)
            if now>best_now:
                best_now,return_move = now,a
            if left==0 or us[0]<cfg['recCollectorCost'] or any(_V5_DIST[p][z]<=2 for z in threat):
                continue
            tax = cfg['collectorAbsorptionRate']*sum(z in foreign for z in (p,*NB8[p]))
            later = (load+crop)*(1-tax) + _v8_patch_trip(value,left-1,patches_free,p,crop,rate)
            best_later = max(best_later,later)
        if return_move is None:
            continue
        if orig:
            # Disallow a proposed wait when we cannot also deliver after waiting.
            if best_later<0:
                continue
            alt = 0
            advantage = best_later-best_now
        else:
            alt = return_move
            # When waiting makes the first delivery impossible, its bank value is zero.
            if best_later<0 and remain>home[p]:
                continue
            advantage = best_now-max(0.,best_later)
        advantage -= 2*max(0.,float(logits[p,orig]-logits[p,alt]))
        if advantage>max(8.,best_gain):
            best_gain,edit=advantage,(uid,alt)
    if edit is None:
        return None
    uid, action = edit
    result = dict(base)
    if action:
        result[uid] = WACT[action]
    else:
        result.pop(uid,None)
    return result


def build(source, prior):
    if hashlib.sha256(source.encode()).hexdigest()!=BASELINE:
        raise ValueError('Expected frozen yan-v7')
    fn='def _act(obs, configuration=None, t0=None):'
    obs_hook="    rng = st['rng']\n    est, known = st['tr'].step"
    search_hook='    if most>=3:\n        tree_action=_r6_first_tree'
    if any(source.count(anchor)!=1 for anchor in (fn,obs_hook,search_hook)):
        raise ValueError('Unexpected v7 entry point')
    fns=(_v8_observe_trips,_v8_after_deposit,_v8_patch_trip,_v8_trip_offer)
    code=source.replace(fn, f'_V8_TRIP_PRIOR={prior!r}\n'+ '\n\n'.join(inspect.getsource(f) for f in fns)+'\n\n'+fn)
    observe="""    prev = st.get('prev')
    _v8_observe_trips(st.setdefault('trip_history',{}), players[me], prev[me] if prev else None, step)
"""
    compare="""    if most>=4 and len(cands)<most:
        trip=_v8_trip_offer(obs,configuration,est,act0,lw0,fixed or {},_ST['trip_history']['completed'])
        if trip is not None:
            trip_key=tuple(sorted(trip.items()))
            if trip_key not in keys:
                keys.add(trip_key)
                cands.append(trip)
"""
    return code.replace(obs_hook,observe+obs_hook).replace(search_hook,compare+search_hook)


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    prior = json.loads((root/'analysis/v8/trip-prior.json').read_text())['prior']
    candidate = build((root/'yan-v7.py').read_text(), prior)
    compile(candidate, str(args.output),'exec')
    with args.output.open('x') as dest:
        dest.write(candidate)
