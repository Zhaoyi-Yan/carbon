"""Guard a productive owned tree when an opposing planter can capture it soon."""
import hashlib
import inspect
from economy import BASELINE


def _grove_threats(obs,dist):
    side=obs['player']
    own,foe=obs['players'][side],obs['players'][1-side]
    thieves=[p for p,_,k in foe[2].values() if k=='PLANTER']
    for tid,(q,age) in own[3].items():
        rate=obs['trees'][tid][1]
        if age>43 or rate<1. or foe[0]<20:
            continue
        for z in thieves:
            if dist[q][z]<=2:
                yield tid,q,age,rate,z


def _grove_guard(obs,config,base):
    side=obs['player']
    own,foe=obs['players'][side],obs['players'][1-side]
    reserved={p for u,p in own[1].items() if base.get(u)}
    ends={u:move(p,WIDX.get(base.get(u),0)) for u,(p,_,_) in own[2].items()}
    rivals=[(p,v) for p,v,k in foe[2].values() if k=='COLLECTOR']
    best,choice=0.,None
    for tid,q,age,income,z in _grove_threats(obs,_V5_DIST):
        life=min(config['episodeSteps']-1-obs['step'],config['treeLifespan']-age)
        for uid,(p,load,kind) in own[2].items():
            if kind!='COLLECTOR' or load>32 or _V5_DIST[p][q]>2:
                continue
            for a,n in enumerate((p,*NB[p])):
                if n in reserved or n in foe[1].values() or any(c==n for u,c in ends.items() if u!=uid):
                    continue
                if any(_V5_DIST[n][enemy]<=1 and v<=load for enemy,v in rivals):
                    continue
                if n not in (z,q) and not (_V5_DIST[n][q]<_V5_DIST[p][q] and _V5_DIST[z][q]>=1):
                    continue
                value=income*min(life,12)*1.4/(1+_V5_DIST[z][q]+_V5_DIST[n][q])
                value*=1.3 if n==z else 1.
                value-=8+.55*load
                if value>best:
                    best,choice=value,(uid,a)
    if choice is None:
        return base
    uid,a=choice
    result=dict(base)
    if a:result[uid]=WACT[a]
    else:result.pop(uid,None)
    return result


def build(source):
    assert hashlib.sha256(source.encode()).hexdigest()==BASELINE
    anchor='def _act(obs, configuration=None, t0=None):'
    hook="        st['liquidity_action']=dict(act)"
    assert source.count(anchor)==source.count(hook)==1
    return source.replace(anchor,inspect.getsource(_grove_threats)+'\n'+inspect.getsource(_grove_guard)+'\n'+anchor).replace(hook,"        act=_grove_guard(obs,configuration,act)\n"+hook)
