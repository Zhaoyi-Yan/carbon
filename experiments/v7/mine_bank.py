"""Offer one currently visible harvest, conditional on a safe, short trip to a friendly depot."""
import numpy as np

_IDX = np.arange(225)
_Y,_X = np.divmod(_IDX,15)
D = np.minimum(abs(_Y[:,None]-_Y),15-abs(_Y[:,None]-_Y))+np.minimum(abs(_X[:,None]-_X),15-abs(_X[:,None]-_X))
L = np.stack((_IDX,((_Y-1)%15)*15+_X,_Y*15+(_X+1)%15,((_Y+1)%15)*15+_X,_Y*15+(_X-1)%15),axis=1)
ACT=(None,'UP','RIGHT','DOWN','LEFT')


def _land(start, command):
    return int(L[start,ACT.index(command)])


def _tax(q,enemy,cfg):
    row,col=divmod(q,15);n=0
    for u,age in enemy[3].values():
        y,x=divmod(u,15);dy=abs(y-row);dx=abs(x-col)
        n+=age<cfg.get('treeLifespan',50)-1 and min(dy,15-dy)<=1 and min(dx,15-dx)<=1
    return min(1.,n*cfg.get('collectorAbsorptionRate',.1))


def _routes(obs,cfg,base,uid,hold):
    side=obs['player'];us,them=obs['players'][side],obs['players'][1-side];q,load,_=us[2][uid]
    homes=set(us[1].values());other_depots=set(them[1].values())
    own_landing={_land(p,base.get(w)) for w,(p,c,k) in us[2].items() if w!=uid}
    own_hires={p for b,p in us[1].items() if b in base}
    collectors=[(p,c) for p,c,k in them[2].values() if k=='COLLECTOR']
    own_loaded=[(p,c) for w,(p,c,k) in us[2].items() if w!=uid and k=='COLLECTOR' and c>=22.]
    role_count=sum(k=='COLLECTOR' for p,c,k in us[2].values())
    rate=max(0.,cfg.get('initialCollectRate',.3)-role_count*cfg.get('collectDecreaseRate',0.))
    gross=rate*float(obs['carbon'][q]) if hold else 0.
    if hold and (q in own_landing or any(D[p,q]<=1 and c<=load for p,c in collectors)):return []
    carried=(load+gross)*(1.-_tax(q,them,cfg)) if hold else load
    paths=[]
    for home in sorted(homes):
        d=int(D[q,home])
        if d not in (1,2):continue
        if home in own_hires or home in own_landing:continue
        if any(D[p,home]<=d+int(hold) and c>max(35.,carried) for p,c in own_loaded):continue
        options=[(home,)] if d==1 else [(int(z),home) for z in L[q,1:] if D[z,home]==1 and z not in homes|other_depots]
        for path in options:
            if path[0] in own_landing:continue
            if any(D[p,z]<=i+1+int(hold) and c<=carried for i,z in enumerate(path) for p,c in collectors):continue
            payout=carried
            for z in path[:-1]:payout*=1.-_tax(z,them,cfg)
            paths.append((float(payout),path,float(gross)))
    return paths


def propose(obs,cfg,base,minimum=6.):
    s=obs['player'];us,them=obs['players'][s],obs['players'][1-s];t=obs['step']
    if not us[1] or t+3>=cfg.get('episodeSteps',300):return base,None
    homes=set(us[1].values());other_homes=set(them[1].values());trees={v[0] for p in (us,them) for v in p[3].values()}
    candidates=[]
    for uid,(q,load,k) in us[2].items():
        cmd=base.get(uid)
        if k!='COLLECTOR' or not cmd or load>36 or q in homes|other_homes|trees:continue
        dst=_land(q,cmd);dist=min(int(D[q,h]) for h in homes)
        if not 1<=dist<=2 or dst in homes|other_homes:continue
        # Preserve imminent captures, planter interceptions and current deliveries.
        if any(D[p,dst]<=1 and (role=='PLANTER' or c>=max(35.,load+15.)) for p,c,role in them[2].values()):continue
        if float(obs['carbon'][q])<=0:continue
        for payout,path,gross in _routes(obs,cfg,base,uid,True):
            if gross<7.:continue
            retreat=min(D[dst,h] for h in homes)<dist
            score=payout-load-1.5*(len(path)-1)-float(retreat)*(1.+.20*load)
            if score>=minimum:candidates.append((float(score),uid,payout,gross,path))
    if not candidates:return base,None
    score,uid,payout,gross,path=max(candidates);a=dict(base);a.pop(uid)
    return a,dict(uid=uid,start=t,home=path[-1],dist=len(path),path=list(path),gross=round(gross,3),
                  load=us[2][uid][1],score=round(score,3),estimated_payout=round(payout,3),cash=us[0])


def follow(obs,cfg,base,intent):
    us=obs['players'][obs['player']];uid=intent['uid']
    if uid not in us[2]:return base,'lost'
    q,load,_=us[2][uid]
    if q in us[1].values():return base,'banked' if load==0 else 'at_depot'
    elapsed=obs['step']-intent['start']
    if elapsed not in (1,2) or load<=0:return base,'expired'
    routes=[(pay,path) for pay,path,_ in _routes(obs,cfg,base,uid,False) if path[-1]==intent['home'] and len(path)<=3-elapsed]
    if not routes:return base,'released'
    _,path=max(routes,key=lambda v:(v[0],base.get(uid)==ACT[list(L[q]).index(v[1][0])]))
    result=dict(base);result[uid]=ACT[list(L[q]).index(path[0])]
    return result,'return'
