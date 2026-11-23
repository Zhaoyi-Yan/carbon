"""Offer one economic worker deviation to v7 without changing its normal move distribution."""
import inspect
import json
from pathlib import Path
from economy import _v8_extra_harvest, BASELINE
import hashlib


def _v8_tree_features(obs, cfg, stock, side, pos, age, delay):
    own, foe = obs['players'][side], obs['players'][1-side]
    trees = {q for pl in obs['players'] for q,a in pl[3].values() if a < cfg['treeLifespan']}
    depots = set(own[1].values()) | set(foe[1].values())
    life = max(0, min(cfg['treeLifespan']-age-delay, cfg['episodeSteps']-1-obs['step']-delay))
    soil = busy = 0.
    for q in NB[pos]:
        if q in trees or q in depots:
            continue
        rivals = sum(n != pos and n in trees for n in NB[q])
        rate = cfg['cellAbsorptionRate']
        potential = max(0., float(stock[q])) * (1-(1-rate*(1+rivals))**life) / (1+rivals)
        soil += potential
        busy += potential * (rivals > 0)
    thief = min((_V5_DIST[pos][q] for q,_,role in foe[2].values() if role == 'PLANTER'), default=15)
    guard = min((_V5_DIST[pos][q] for q,load,role in own[2].values() if role == 'COLLECTOR' and load < 40), default=15)
    enemy_depot = min(_V5_DIST[pos][q] for q in foe[1].values())
    load_near = sum(min(250., load) for q,load,k in foe[2].values()
                    if k=='COLLECTOR' and (q==pos or q in NB8[pos])) * cfg['collectorAbsorptionRate']
    floor = min(life/12.,1.)
    cover = min(thief,8)/8.
    return [floor, life/50., soil/100., busy/100., soil*cover/100., floor*cover,
            floor*min(guard,8)/8., floor*min(enemy_depot,8)/8., min(load_near,100.)/25.,
            floor*min(load_near,100.)/25., life*(enemy_depot<=3)/50.]


def _v8_root_option(obs, cfg, stock, base, logits, locks):
    side, t = obs['player'], obs['step']
    if t < 45:
        return None
    own, foe = obs['players'][side], obs['players'][1-side]
    depots = set(own[1].values()) | set(foe[1].values())
    occupied = {uid:move(pos,WIDX.get(base.get(uid),0)) for uid,(pos,_,_) in own[2].items()}
    recruits = {q for uid,q in own[1].items() if base.get(uid)}
    best, choice = 0., None
    if _V8_TREE:
        woods = {q:(i,age) for i,pl in enumerate(obs['players'])
                 for q,age in pl[3].values() if age<cfg['treeLifespan']}
        cost = cfg['plantCost']
        if len(own[3]) > cfg['playerTreeProtectiveNumber']:
            cost += cfg['plantCostInflationRatio'] * cfg['plantCostInflationBase']**len(woods)
        killers=[q for q,_,role in foe[2].values() if role=='COLLECTOR']
        def payoff(q, delay):
            owner,age = woods.get(q, (None,0))
            if owner==side or q in depots:
                return 0.
            price = cost if owner is None else cfg['seizeCost']
            if own[0]<price:
                return 0.
            spans = min(cfg['episodeSteps']-1-t-delay, cfg['treeLifespan']-age-delay)
            if spans<=0:
                return -price
            x = _v8_tree_features(obs,cfg,stock,side,q,age,delay)
            income = max(0., sum(a*b for a,b in zip(x,_V8_PAYBACK[int(owner is not None)])))
            return ((1.45 if owner is not None else 1.) * income-price) / (1+.35*delay)
        for uid,(p,_,role) in own[2].items():
            if role!='PLANTER' or uid in locks:
                continue
            original = WIDX.get(base.get(uid),0)
            # The alternative either vetoes a costly purchase here or takes one v7 skipped.
            if p in depots or woods.get(p,(None,))[0]==side:
                continue
            here = payoff(p,0)
            if original and here<25 or not original and here>-18:
                continue
            reserved = {dest for u,dest in occupied.items() if u!=uid} | recruits
            for a,q in enumerate([p,*NB[p]]):
                if a==original or (a and original) or q in reserved or any(_V5_DIST[q][z]<=1 for z in killers):
                    continue
                there = payoff(q,1) if a else here
                baseline = payoff(occupied[uid],1) if original else here
                gain = there-baseline - 6*max(0.,float(logits[p,original]-logits[p,a]))
                if gain>max(18.,best):
                    best,choice = gain,(uid,a)
    if _V8_CARGO and t>=125:
        # Compare a single extra collection only when the incumbent starts its return trip.
        changes = np.zeros_like(logits)
        _v8_extra_harvest(obs,cfg,stock,changes)
        home = _fc_home(tuple(sorted(own[1].values())))
        for uid,(p,load,role) in own[2].items():
            dest = occupied[uid]
            if role!='COLLECTOR' or uid in locks or home[dest]>=home[p] or changes[p,0]<=0:
                continue
            if p in recruits or any(q==p for u,q in occupied.items() if u!=uid):
                continue
            orig = WIDX[base[uid]]
            gain = 15*float(changes[p,0]) - 2*max(0.,float(logits[p,orig]-logits[p,0]))
            if gain>max(8.,best):
                best,choice = gain,(uid,0)
    if choice is not None:
        out=dict(base)
        uid,a=choice
        if a:
            out[uid]=WACT[a]
        else:
            out.pop(uid,None)
        return out
    return None


def build(source, arm):
    if hashlib.sha256(source.encode()).hexdigest()!=BASELINE:
        raise ValueError("Expected frozen yan-v7")
    weights=json.loads(Path(__file__).with_name('tree-fit.json').read_text())['coefficients']
    fn='def _act(obs, configuration=None, t0=None):'
    marker='    if most>=3:\n        tree_action=_r6_first_tree'
    if source.count(marker)!=1 or source.count(fn)!=1:
        raise ValueError("Unexpected search layout")
    const=f"_V8_TREE={arm in ('trees','combined')}\n_V8_CARGO={arm in ('cargo','combined')}\n_V8_PAYBACK={weights!r}\n"
    defs='\n\n'.join(inspect.getsource(f) for f in (_v8_tree_features,_v8_extra_harvest,_v8_root_option))
    add="""    if most>=4 and len(cands)<most:
        trial=_v8_root_option(obs,configuration,est,act0,lw0,fixed or {})
        if trial is not None:
            trial_key=tuple(sorted(trial.items()))
            if trial_key not in keys:
                keys.add(trial_key)
                cands.append(trial)
"""
    return source.replace(fn,const+defs+'\n\n'+fn).replace(marker,add+marker)
