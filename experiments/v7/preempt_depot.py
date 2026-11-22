"""Reassign a planned collector hire to a threatened near-home grove for at most four decisions."""
import numpy as np
from preempt_tree import _v7_pt_profile,_V7_PT_D as D,_V7_PT_L as LAND,_CMDS,follow as worker_follow


def plan(obs,cfg,base,previous,coef):
    side=obs['player'];us,them=obs['players'][side],obs['players'][1-side]
    if len(us[2])>=cfg.get('workerLimit',10) or us[0]<cfg.get('recCollectorCost',30) or len(us[1])<2:return base,None
    hires=[(b,h) for b,h in us[1].items() if base.get(b)=='RECCOLLECTOR']
    if not hires:return base,None
    old=previous[1-side][2] if previous else {}
    watch=_v7_pt_profile(obs,old,side)
    if not watch:return base,None
    ends={int(LAND[q,_CMDS.index(base.get(k))]) for k,(q,_,_) in us[2].items()}
    candidates=[]
    for uid,q,age,rate,thief,z,d,closing,x in watch:
        if not 2<=d<=4 or closing<=0 or q in ends:continue
        likelihood=1./(1.+np.exp(-np.clip(np.dot(coef,x),-30.,30.)))
        gross=2*rate*min(16.,50-age,cfg.get('episodeSteps',300)-1-obs['step'])*likelihood
        if gross<14.:continue
        for nb,np_ in us[1].items():
            dist=int(D[q,np_])
            if not 1<=dist<=min(3,d) or nb in base or np_ in ends:continue
            if any(D[np_,loc]<=1 and load>=35 for loc,load,k in us[2].values() if k=='COLLECTOR'):continue
            if any(D[np_,loc]<=1 for loc,load,k in them[2].values() if k=='COLLECTOR' and load<=2):continue
            # A new birth loses one move. Do not redirect the hire when an available
            # existing light collector has an equally quick approach to the tree.
            if any(D[q,loc]<=dist and load<=20. for loc,load,k in us[2].values() if k=='COLLECTOR'):continue
            for ob,op in hires:
                if ob==nb or D[q,op]-dist<2:continue
                value=gross/(1.+.25*dist)
                candidates.append((float(value),uid,thief,q,nb,np_,ob,float(likelihood)))
    if not candidates:return base,None
    merit,tree,attacker,goal,nb,near,ob,risk=max(candidates)
    act=dict(base);act.pop(ob);act[nb]='RECCOLLECTOR'
    return act,dict(mode='redeploy_hire',old_base=ob,new_base=nb,home=int(near),worker=None,old_workers=sorted(us[2]),
                    tree=tree,goal=int(goal),attacker=attacker,risk=round(risk,4),expected=round(merit,2))


def follow(obs,cfg,base,intent):
    own=obs['players'][obs['player']]
    if intent['worker'] is None:
        born=[u for u,(p,_,k) in own[2].items() if k=='COLLECTOR' and p==intent['home'] and u not in intent['old_workers']]
        if len(born)!=1:return base,None
        intent=dict(intent,worker=born[0])
    return worker_follow(obs,cfg,base,intent)
