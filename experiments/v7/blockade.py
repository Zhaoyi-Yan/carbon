"""Consider one light collector denying access to an illiquid rival's depot approaches."""
import inspect
from pathlib import Path
import numpy as np


def _v7_blockade(obs, cfg, estimate, planned):
    side=obs['player'];t=obs['step']
    own,foe=obs['players'][side],obs['players'][1-side]
    price=cfg.get('recCollectorCost',30)
    if t<18 or t>LAST_TURN-12 or own[0]<3*price or foe[0]>=1.5*price or len(own[2])<4:
        return planned
    tree_yield=sum(obs.get('trees',{}).get(k,(None,0.))[1] for k in foe[3])
    if tree_yield>=price/3:
        return planned
    D,L=_V5_GEOMETRY,_V5_REACH
    homes=list(foe[1].values());depots=list(own[1].values())
    if not homes or not depots:
        return planned
    reach=D[homes].min(0)
    carriers=[(p,c) for p,c,role in foe[2].values() if role=='COLLECTOR' and c>=1.65*price and 2<=reach[p]<=8]
    if not carriers:
        return planned
    convoy=sum(c for _,c in carriers)
    if convoy<3*price:
        return planned
    ends={u:int(L[p,WIDX.get(planned.get(u),0)]) for u,(p,_,_) in own[2].items()}
    recruits={p for u,p in own[1].items() if u in planned}
    guards=[(u,p,c) for u,(p,c,role) in own[2].items() if role=='COLLECTOR' and c<=min(.45*price, min(k[1] for k in carriers)-1) and reach[p]<=3]
    if not guards:return planned
    all_light=[(p,c,u) for u,(p,c,r) in own[2].items() if r=='COLLECTOR' and c<min(k[1] for k in carriers)]
    soil=np.asarray(estimate)
    cover=np.ones(225)
    for tree,(_,age) in own[3].items():
        if age>=50:continue
        pos=own[3][tree][0];y,x=divmod(pos,15)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                cover[((y+dy)%15)*15+(x+dx)%15]*=.90
    moves=L[:,1:]
    best=None
    for uid,pos,cargo in guards:
        chosen=ends[uid]
        occupied={q for u,q in ends.items() if u!=uid}|recruits
        lighter=[q for q,c,role in foe[2].values() if role=='COLLECTOR' and c<=cargo]
        options=[int(q) for q in L[pos] if q not in occupied and q not in homes and q not in depots and
                 (q==chosen or (reach[q]<=2 and not any(D[q,k]<=1 for k in lighter)))]
        if chosen not in options or len(options)<2:continue
        # Estimate the best discounted receipt the rival could achieve at either depot.
        # A collision ON its own depot still credits the carrier. Only pre-depot cells can deny cash.
        existing=[q for q,_,u in all_light if u!=uid]
        guard_distance=D[existing].min(0) if existing else np.full(225,15)
        field=np.where(guard_distance==0,.42,np.where(guard_distance==1,.72,.98))
        spacing=D[options]
        risk=np.where(spacing==0,.21,np.where(spacing==1,.58,np.where(spacing==2,.91,1.)))
        attenuation=.90*cover*field*risk
        attenuation[:,homes]=1.
        arrival=np.zeros_like(attenuation);arrival[:,homes]=1.
        for _ in range(8):
            arrival=(arrival*attenuation)[:,moves].max(2)
            arrival[:,homes]=1.
        # The controller has committed to a concrete first step and cannot occupy a different square.
        now=.94*cover*np.where(spacing==0,0.,1.)
        now[:,homes]=1.
        first=(arrival*now)[:,L]
        expected=sum(min(c,350.)*first[:,p,:].max(1) for p,c in carriers)
        baseline=options.index(chosen)
        for j,target in enumerate(options):
            if j==baseline:continue
            # A useful stationary harvest is an immediate opportunity cost. Penalize occupying rich cells as a guard.
            foregone=cfg.get('initialCollectRate',.3)*max(0.,float(soil[pos])) if chosen==pos and target!=pos else 0.
            opportunity=2.+.04*max(0.,float(soil[chosen])-float(soil[target]))+.6*foregone
            gain=expected[baseline]-expected[j]-opportunity
            if gain>4. and (best is None or gain>best[0]):
                best=(float(gain),uid,pos,target)
    if best is None:return planned
    _,uid,pos,target=best
    result=dict(planned);direction=list(L[pos]).index(target)
    if direction:result[uid]=WACT[direction]
    else:result.pop(uid,None)
    return result


def build(code):
    marker='def _act(obs, configuration=None, t0=None):'
    at="    st['tr'].set_actions(players, act)"
    if code.count(marker)!=1 or code.count(at)!=1:
        raise ValueError('Expected pristine v6 action entry and final tracker update')
    code=code.replace(marker,inspect.getsource(_v7_blockade)+'\n\n'+marker)
    return code.replace(at,"    if configuration is not None and _SEARCH_OK:\n        act=_v7_blockade(obs,configuration,est,act)\n"+at)
