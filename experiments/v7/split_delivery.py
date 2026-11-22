"""Move two stranded couriers toward different depots when both need a long return."""
import inspect
import numpy as np


def _v7_future_guard(q,carry,history,foe_logits,our_loads,homes,horizon):
    # The first movement distribution uses the observed rival policy. For later steps,
    # mix continued patrolling, persistence and a possible pursuit of our visible convoy.
    reach=_V7_L
    actions=np.arange(5);foe_dist=_V7_D
    previous=history if history is not None else q
    prior_direction=next((a for a,n in enumerate(reach[previous]) if n==q),0)
    z=foe_logits[q].astype(float);z-=z.max();v=np.exp(z/.85);v/=v.sum()
    mass=np.zeros((225,5));mass[reach[q],actions]=v
    occ=np.zeros((horizon,225));occ[0]=mass.sum(1)
    prey=[p for p,load in our_loads if load>=carry]
    prey_distance=_v7_nearest(prey)
    home_distance=_v7_nearest(homes)
    delta=prey_distance[:,None]-prey_distance[reach]
    block=home_distance[:,None]-home_distance[reach]
    src=foe_dist[q]
    score=np.exp(.24*delta+.19*block+.16*(src[:,None]-src[reach]))
    score[:,0]*=1.08
    for t in range(1,horizon):
        new=np.zeros((225,5))
        for last in range(5):
            probs=score.copy();probs[:,last]*=1.65 if last else 1.2
            if t==1:probs[:,prior_direction]*=1.16
            probs/=probs.sum(1,keepdims=True)
            new[reach,actions]+=mass[:,last,None]*probs
        mass=new;occ[t]=mass.sum(1)
    return occ


def _v7_split_delivery(obs,cfg,base,drought,enemy_logits,past):
    s=obs['player'];us,enemy=obs['players'][s],obs['players'][1-s]
    fee=cfg.get('recCollectorCost',30)
    if us[0]>=fee or us[3] or drought<18:return base,None
    homes=tuple(us[1].values());eh=tuple(enemy[1].values())
    if len(homes)!=2:return base,None
    fleet=[(uid,p,load) for uid,(p,load,k) in us[2].items() if k=='COLLECTOR' and load>=fee]
    if len(fleet)<2 or min(min(_V7_D[p,h] for h in homes) for uid,p,load in fleet)<4:return base,None
    h=min(13,LAST_TURN+1-obs['step'])
    if h<4:return base,None
    previous=past[1-s][2] if past else {}
    threats=[]
    for uid,(q,carry,k) in enemy[2].items():
        if k=='COLLECTOR' and any(carry<=load for _,_,load in fleet):
            threats.append((q,carry,_v7_future_guard(q,carry,previous.get(uid,(q,))[0],enemy_logits,[(p,v) for _,p,v in fleet],homes,h)))
    taxed=np.ones(225)
    for p,age in enemy[3].values():
        if age<49:
            local=(_V7_D[p]<=1)|((_V7_D[p]==2)&(np.arange(225)//15!=p//15)&(np.arange(225)%15!=p%15))
            taxed[local]*=1.-cfg.get('collectorAbsorptionRate',.1)
    occupied={int(_V7_L[p,WIDX.get(base.get(u),0)]) for u,(p,_,k) in us[2].items() if not any(u==v[0] for v in fleet)}
    choices=[]
    for uid,p,load in fleet:
        safe=np.ones((h,225));min_distance=_v7_nearest([q for q,c,_ in threats if c<=load])
        for _,c,occ in threats:
            if c<=load:safe*=1.-np.minimum(1.,occ*1.3)
        # Short-range forced deposits still pass through the established dispatcher.
        safe[0,min_distance<=1]=0.
        if eh:safe[:,list(eh)]=0.
        for home in homes:
            value=np.zeros(225)
            other=homes[1] if home==homes[0] else homes[0]
            for t in range(h-1,0,-1):
                worth=safe[t]*taxed*value*.92;worth[home]=1.;worth[other]=0.
                nxt=worth[_V7_L];nxt[:,0]*=.80
                value=nxt.max(1)
            dest=_V7_L[p];scores=safe[0,dest]*taxed[dest]*value[dest]*.92
            scores[dest==home]=1.;scores[dest==other]=0.;scores[0]*=.80
            for act,q in enumerate(dest):
                if q in occupied or scores[act]<.09:continue
                amount=min(load,2.5*fee)+.20*load
                choices.append((uid,int(q),home,act,amount*scores[act]+.05*int(act==WIDX.get(base.get(uid),0))))
    pairs=[]
    for i,(u,q,a,act,score) in enumerate(choices):
        for v,r,b,other,val in choices[i+1:]:
            if u==v or q==r or a==b:continue
            pairs.append((score+val,u,act,v,other))
    if not pairs:return base,None
    _,u,move,v,other=max(pairs)
    out=dict(base)
    for uid,act in ((u,move),(v,other)):
        if act:out[uid]=WACT[act]
        else:out.pop(uid,None)
    return out,(u,v)


def build(code):
    anchor='def _act(obs, configuration=None, t0=None):'
    old='        act,_=_v7_corridor(obs,configuration,act,drought,st)'
    if code.count(anchor)!=1 or code.count(old)!=1:raise ValueError('expected verified short corridor')
    src=inspect.getsource(_v7_future_guard)+'\n\n'+inspect.getsource(_v7_split_delivery)
    hook=old.replace('act,_=','act,route=')+"\n        if route is None and not st.get('corridor_track') and not st.get('v7_next_depot'):\n            act,_=_v7_split_delivery(obs,configuration,act,drought,st['v5_history']['opponent'][0],st.get('prev'))"
    return code.replace(anchor,src+'\n\n'+anchor).replace(old,hook)
