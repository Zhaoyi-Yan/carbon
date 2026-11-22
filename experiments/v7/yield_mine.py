"""Allow one safe early collector to take a profitable extra harvest before moving."""
import inspect


def _v7_yield_mine(obs,cfg,stock,base,logits,memory):
    t=obs['step'];side=obs['player']
    memory['event']=None
    if t<4 or t>=75:return base
    own,enemy=obs['players'][side],obs['players'][1-side]
    own_homes=tuple(own[1].values());all_homes=set(own_homes)|set(enemy[1].values())
    if not own_homes:return base
    camps={q for p in obs['players'] for q,_ in p[3].values()}
    rival=[(q,c) for q,c,k in enemy[2].values() if k=='COLLECTOR']
    nc=sum(k=='COLLECTOR' for _,_,k in own[2].values())
    rate=max(0.,cfg.get('initialCollectRate',.3)-nc*cfg.get('collectDecreaseRate',0.))
    reserved={u:move(q,WIDX.get(base.get(u),0)) for u,(q,_,_) in own[2].items()}
    recent=memory.setdefault('extra',{})
    best=None
    for uid,(q,load,role) in own[2].items():
        orig=WIDX.get(base.get(uid),0);dst=reserved[uid]
        if role!='COLLECTOR' or not orig or load>42 or t-recent.get(uid,-10)<3:continue
        if q in camps or q in all_homes or dst in own_homes:continue
        if any(o!=uid and target==q for o,target in reserved.items()):continue
        h=_v5_home(own_homes);d=h[q]
        if not 1<=d<=5:continue
        # A collector sees the exact stock under itself. Adjacent stock is its legal belief.
        gross=rate*obs['carbon'][q]
        if gross<12:continue
        if any(_V5_DIST[q][p]<=2 and c<=load+gross for p,c in rival):continue
        if any(_V5_DIST[dst][p]<=1 and c>=max(45.,load+20.) for p,c in rival):continue
        if h[dst]<d and load+own[0]>=cfg.get('recCollectorCost',30) and own[0]<cfg.get('recCollectorCost',30):continue
        tax=0.
        for p,age in enemy[3].values():
            if age>=49:continue
            dy=abs(q//15-p//15);dx=abs(q%15-p%15)
            if min(dx,15-dx)<=1 and min(dy,15-dy)<=1:tax+=cfg.get('collectorAbsorptionRate',.1)
        # Charge for the delayed trip or for forgoing a richer next cell, and use the
        # network's move preference as an opportunity cost if it strongly wants to go.
        next_crop=0. if dst in camps|all_homes else rate*max(0.,stock[dst])
        advantage=gross-(load+gross)*min(1.,tax)-.45*max(0.,next_crop-.70*gross)
        advantage-=.08*load+2.5*(h[dst]<d)+1.35*min(5.,max(0.,logits[q,orig]-logits[q,0]))
        if advantage>=7. and (best is None or advantage>best[0]):best=(advantage,uid,gross,d,load)
    if best is None:return base
    net,uid,gross,d,load=best
    recent[uid]=t;memory['event']=('extra_stay',uid,[round(float(gross),3),round(float(net),3),int(d),round(float(load),2)])
    result=dict(base);result.pop(uid)
    return result


def build(code):
    root='def _act(obs, configuration=None, t0=None):'
    spot="    st['tr'].set_actions(players, act)"
    if code.count(root)!=1 or code.count(spot)!=1 or 'def _v7_yield_mine(' in code:raise ValueError('Expected compatible root')
    return code.replace(root,inspect.getsource(_v7_yield_mine)+'\n\n'+root).replace(
        spot,"    if configuration is not None:act=_v7_yield_mine(obs,configuration,est,act,lw,st.setdefault('yield_mine',{}))\n"+spot)
