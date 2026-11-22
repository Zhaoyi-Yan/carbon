"""Offer a coordinated first-tree route, retaining cash when it enables immediate planting."""
import argparse
import inspect
from pathlib import Path


def _r6_first_tree(obs, config, carbon, act, worker_logits):
    pl=obs['players'];me=obs['player'];own,rival=pl[me],pl[1-me]
    if obs['step']>70 or own[3] or own[0]<config.get('plantCost',20.):return None
    planters=[(uid,p) for uid,(p,_,k) in own[2].items() if k=='PLANTER']
    if not planters:return None
    grid=_V5_GEOMETRY;adj=_V5_ADJ;soil=np.asarray(carbon)
    near=lambda sites:grid[sites].min(0) if sites else np.full(N,15)
    dangerous=near([p for p,c,k in rival[2].values() if k=='COLLECTOR'])
    challenge=near([p for p,c,k in rival[2].values() if k=='PLANTER'])
    competing=np.zeros(N)
    for pos,age in rival[3].values():
        competing[adj[pos]]+=min(1.,(50-age)/15.)
    usable=soil/(1.+.58*competing)
    # Value of the first twenty discounted receipts from the current soil reserve.
    income=(usable[adj].sum(1)*config.get('cellAbsorptionRate',.0375)*15.5)
    for pos,load,k in rival[2].values():
        if k=='COLLECTOR' and load>25:
            income+=np.maximum(0.,2.-grid[pos])*min(load,180.)*.055
    price=config.get('plantCost',20.)
    benefit=income-price
    for key,(pos,age) in rival[3].items():
        rate=obs.get('trees',{}).get(key,(None,0))[1]
        benefit[pos]=2.*max(rate,1.)*min(18,50-age)-config.get('seizeCost',20.)
    benefit*=np.clip(.38+.16*challenge,.38,1.)
    benefit*=np.clip(.25+.18*dangerous,.25,1.)
    benefit[list(own[1].values())+list(rival[1].values())]=-1000.
    land=_V5_REACH
    units={uid:land[p,WIDX.get(act.get(uid),0)] for uid,(p,_,_) in own[2].items()}
    bids=[key for key in own[1] if key in act]
    best=None
    for uid,pos in planters:
        reserved={q for worker,q in units.items() if worker!=uid}
        occupied_bases={own[1][key] for key in bids}
        for target in np.argsort(-benefit/(1.+grid[pos]*.6)**1.4)[:5]:
            if benefit[target]<20. or grid[pos,target]>5:continue
            for a,q in enumerate(land[pos]):
                if q in reserved or q in occupied_bases or dangerous[q]<=1:continue
                if (target==pos and a!=0) or (target!=pos and grid[q,target]>=grid[pos,target]):continue
                # Keep moves consistent with existing unit control when several routes tie.
                val=benefit[target]/(1.+.65*grid[pos,target])**1.4+.6*np.clip(worker_logits[pos,a],-6,6)
                if best is None or val>best[0]:best=val,uid,a,target
    if best is None:return None
    _,uid,a,target=best;out=act.copy()
    if a:out[uid]=WACT[a]
    else:
        out.pop(uid,None)
        fee=config.get('seizeCost',20.) if any(q==target for q,_ in rival[3].values()) else price
        def cost(key):
            kind=out[key]
            return config.get('recCollectorCost' if kind=='RECCOLLECTOR' else 'recPlanterCost',30.)
        money=own[0]-sum(cost(key) for key in bids)
        for key in reversed(bids):
            if money>=fee:break
            money+=cost(key);out.pop(key)
    return out if out!=act else None


def build(code):
    start='def _search(obs,';spot='    if most >= 3:\n        for aw, ab'
    if code.count(start)!=1 or code.count(spot)!=1:raise ValueError('expected dual-style root')
    code=code.replace(start,inspect.getsource(_r6_first_tree)+'\n\n'+start)
    insertion="""    if most>=3:
        tree_action=_r6_first_tree(obs,configuration,est,act0,lw0)
        if tree_action:
            tree_key=tuple(sorted(tree_action.items()))
            if tree_key not in keys:
                keys.add(tree_key);cands.append(tree_action)
"""
    code=code.replace(spot,insertion+spot)
    code=code.replace("if marker not in keys:\n                keys.add(marker)","if len(cands)<most and marker not in keys:\n                keys.add(marker)")
    compile(code,'<first-tree>','exec');return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(build(a.source.read_text()))
