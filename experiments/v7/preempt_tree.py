"""Anticipate an approaching planter and choose one intercept or a compensating seizure."""
import inspect
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
MODEL=HERE/'preempt-tree-model.json'
GRID=np.arange(225);Y,X=np.divmod(GRID,15)
_V7_PT_D=np.minimum(abs(Y[:,None]-Y),15-abs(Y[:,None]-Y))+np.minimum(abs(X[:,None]-X),15-abs(X[:,None]-X))
_V7_PT_L=np.stack((GRID,((Y-1)%15)*15+X,Y*15+(X+1)%15,((Y+1)%15)*15+X,Y*15+(X-1)%15),axis=1)
_CMDS=(None,'UP','RIGHT','DOWN','LEFT')


def _v7_pt_profile(obs,prior,side):
    """Every feature is observable before the chosen action; hostile occupation is excluded."""
    us,them=obs['players'][side],obs['players'][1-side];t=obs['step']
    if not us[3] or them[0]<20:return []
    plan=[(u,p) for u,(p,_,k) in them[2].items() if k=='PLANTER']
    if not plan:return []
    homes=tuple(us[1].values());enemyhomes=tuple(them[1].values())
    ownc=[p for p,v,k in us[2].values() if k=='COLLECTOR'];their_c=[p for p,v,k in them[2].values() if k=='COLLECTOR']
    ownp=[p for p,v,k in us[2].values() if k=='PLANTER']
    out=[]
    for tree,(q,age) in us[3].items():
        d=_V7_PT_D[q];home=min((d[p] for p in homes),default=15)
        rate=float(obs['trees'].get(tree,(None,0.))[1])
        if home>6 or age>43 or rate<.6:continue
        dist,who,z=min((int(d[z]),uid,z) for uid,z in plan)
        if not 1<=dist<=4:continue
        old=prior.get(who);closing=0 if old is None else int(d[old[0]]-dist)
        near=lambda pts:min((int(d[p]) for p in pts),default=12)
        guard=near(ownc);opguard=near(their_c);reclaim=near(ownp)
        x=[1.,*(float(dist==k) for k in (1,2,3)),float(closing>0),float(closing<0),float(closing>0 and dist<=2),
           min(6,guard)/4.,float(guard<dist),min(6,opguard)/4.,min(6,reclaim)/4.,min(20,rate)/6.,age/50.,
           min(home,6)/5.,min(near(enemyhomes),9)/6.,min(them[0],150)/100.,min(5,len(plan))/3.,
           sum(d[p]<=3 for _,p in plan)/2.,t/299.]
        out.append((tree,q,age,rate,who,z,dist,closing,x))
    return out


def _v7_pt_options(obs,cfg,base,previous,coef):
    s=obs['player'];us,them=obs['players'][s],obs['players'][1-s];t=obs['step']
    if not 20<=t<=cfg.get('episodeSteps',300)-15:return []
    prior=previous[1-s][2] if previous else {}
    watch=_v7_pt_profile(obs,prior,s)
    if not watch:return []
    homes=set(us[1].values());depots=homes|set(them[1].values());occupied={p for b,p in us[1].items() if b in base}
    original={u:_CMDS.index(base.get(u)) for u in us[2]}
    land={u:int(_V7_PT_L[p,original[u]]) for u,(p,v,k) in us[2].items()}
    foe_c=[(p,v) for p,v,k in them[2].values() if k=='COLLECTOR']
    ranked=[]
    for tree,q,age,rate,thief,z,d,closing,x in watch:
        if closing<=0 or d<2:continue
        risk=1./(1.+np.exp(-np.clip(np.dot(x,coef),-30.,30.)))
        remaining=min(18,50-age,cfg.get('episodeSteps',300)-1-t)
        exposure=risk*2*rate*max(0,remaining-d)
        if exposure<15.:continue
        # Choose an approach square on the thief's shortest route, *before* it reaches the tree.
        # A worker holding the tree is already covering it, so leave the baseline alone.
        if q in land.values():continue
        goals=[(int(v),'intercept') for v in _V7_PT_L[z,1:] if _V7_PT_D[q,v]<d]
        goals.append((int(q),'cover'))
        for uid,(p,load,k) in us[2].items():
            if k!='COLLECTOR' or load>32. or land[uid] in homes and load>=12.:continue
            if any(_V7_PT_D[land[uid],f]<=1 and v>=max(45,load+20) for f,v in foe_c):continue
            blocked=occupied|{v for a,v in land.items() if a!=uid}|depots
            for dest,kind in goals:
                travel=int(_V7_PT_D[p,dest]);window=1 if kind=='intercept' else d
                if travel>window or dest in blocked:continue
                for a,first in enumerate(_V7_PT_L[p]):
                    if (travel and _V7_PT_D[first,dest]>=travel) or (not travel and a):continue
                    if first in blocked or any(_V7_PT_D[f,first]<=1 and v<=load+1 for f,v in foe_c):continue
                    benefit=exposure/(1+.36*travel)*(.68 if kind=='cover' else 1.)-.5*load-3.0*travel
                    if benefit>=10.:ranked.append((float(benefit),uid,a,kind,tree,int(dest),float(risk),thief))
        # An opposing planter commits to our home grove, leaving a valuable, safely
        # reachable opposing tree available. A single own planter can exchange ownership.
        if us[0]<cfg.get('seizeCost',20):continue
        for eid,(dst,enemyage) in them[3].items():
            er=float(obs['trees'].get(eid,(None,0))[1]);life=min(18,50-enemyage,remaining)
            if er<1. or life<4 or _V7_PT_D[dst,z]<3:continue
            for uid,(p,load,k) in us[2].items():
                if k!='PLANTER' or _V7_PT_D[p,dst]>3:continue
                travel=int(_V7_PT_D[p,dst]);blocked=occupied|{v for a,v in land.items() if a!=uid}|depots
                if dst in blocked or any(_V7_PT_D[f,dst]<=2 for f,v in foe_c):continue
                for a,first in enumerate(_V7_PT_L[p]):
                    if (travel and _V7_PT_D[first,dst]>=travel) or (not travel and a):continue
                    if first in blocked or any(_V7_PT_D[f,first]<=1 for f,v in foe_c):continue
                    value=min(1.,.30+risk)*min(exposure+20.,2.*er*(life-travel)-cfg.get('seizeCost',20))/(1+.4*travel)-4.*travel
                    if value>=10.:ranked.append((float(value),uid,a,'exchange',tree,int(dst),float(risk),thief))
    return sorted(ranked,reverse=True)


def plan(obs,cfg,base,previous,coef):
    options=_v7_pt_options(obs,cfg,base,previous,coef)
    if not options:return base,None
    benefit,u,a,kind,tree,goal,risk,thief=options[0]
    command=_CMDS[a]
    if base.get(u)==command:return base,None
    act=dict(base)
    if command:act[u]=command
    else:act.pop(u,None)
    return act,dict(mode=kind,worker=u,tree=tree,goal=goal,attacker=thief,risk=round(risk,4),expected=round(benefit,2))


def follow(obs,cfg,base,intent):
    s=obs['player'];us,foe=obs['players'][s],obs['players'][1-s]
    u=intent['worker'];tree=intent['tree'];goal=intent['goal'];kind=intent['mode']
    if u not in us[2]:return base,None
    p,load,role=us[2][u];cmd=_CMDS.index(base.get(u))
    if kind=='exchange':
        if us[0]<cfg.get('seizeCost',20) or not any(q==goal for q,age in foe[3].values()):return base,None
    else:
        if tree not in us[3] or intent['attacker'] not in foe[2] or load>32:return base,None
        q=us[3][tree][0];z=foe[2][intent['attacker']][0]
        if _V7_PT_D[q,z]>3:return base,None
        if kind=='intercept' and p==goal:
            goal=z if _V7_PT_D[p,z]<=1 else q
    homes=set(us[1].values());depots=homes|set(foe[1].values())
    if _V7_PT_L[p,cmd] in homes and load>=12:return base,None
    block=depots|{int(_V7_PT_L[v[0],_CMDS.index(base.get(k))]) for k,v in us[2].items() if k!=u}
    enemy=[z for z,c,k in foe[2].values() if k=='COLLECTOR' and (role=='PLANTER' or c<=load+1)]
    d=int(_V7_PT_D[p,goal]);opts=[]
    for a,z in enumerate(_V7_PT_L[p]):
        if (d and _V7_PT_D[z,goal]>=d) or (not d and a) or z in block:continue
        danger=min((int(_V7_PT_D[x,z]) for x in enemy),default=12)
        if danger>1:opts.append((.14*(cmd==a)+.04*min(6,danger),a))
    if not opts:return base,None
    a=max(opts)[1];out=dict(base)
    if a:out[u]=_CMDS[a]
    else:out.pop(u,None)
    return out,intent
