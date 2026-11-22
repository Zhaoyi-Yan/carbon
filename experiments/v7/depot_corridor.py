"""Advance a courier along a short collision-free route when working capital is scarce."""
import inspect


def _v7_follow_corridor(obs,action,uid,tiles):
    us,them=obs['players'][obs['player']],obs['players'][1-obs['player']]
    if uid not in us[2] or not tiles:return None
    pos,load,role=us[2][uid];q=tiles[0]
    if role!='COLLECTOR' or load<=0 or q not in _V7_L[pos,1:]:return None
    homes=set(us[1].values());foe_bases=set(them[1].values())
    threats=[(z,c) for z,c,k in them[2].values() if k=='COLLECTOR']
    if q not in homes and any(c<=load and _V7_D[z,q]<=1 for z,c in threats):return None
    out=dict(action);land={u:int(_V7_L[p,WIDX.get(out.get(u),0)]) for u,(p,c,k) in us[2].items() if u!=uid}
    blocked={val for key,val in us[1].items() if key in out}
    for other,arrival in list(land.items()):
        if arrival!=q:continue
        p,c,kind=us[2][other];others={v for k,v in land.items() if k!=other}
        choices=[]
        for a,z in enumerate(_V7_L[p]):
            z=int(z)
            if z==q or z in others|blocked|foe_bases:continue
            if any(_V7_D[z,w]<=1 and (kind=='PLANTER' or v<=c) for w,v in threats):continue
            choices.append((int(a==0),-min(_V7_D[z,b] for b in homes),-a,z))
        if not choices:
            if q in homes:continue
            return None
        _,_,negative,dest=max(choices);a=-negative;land[other]=dest
        if a:out[other]=WACT[a]
        else:out.pop(other,None)
    for key,p in us[1].items():
        if p==q:out.pop(key,None)
    out[uid]=WACT[list(_V7_L[pos]).index(q)]
    return out


def _v7_corridor(obs,config,action,drought,memory=None):
    side=obs['player'];us,them=obs['players'][side],obs['players'][1-side]
    fee=config.get('recCollectorCost',30)
    protected=set()
    if memory is not None:
        prior=memory.pop('corridor_track',[])
        for turn,uid,tiles in prior:
            if turn!=obs['step']:continue
            revised=_v7_follow_corridor(obs,action,uid,tiles)
            if revised is None or any(revised.get(u)!=action.get(u) for u in protected):continue
            action=revised;protected.add(uid)
            if len(tiles)>1:memory.setdefault('corridor_track',[]).append((turn+1,uid,tiles[1:]))
        if len(protected)>=2:return action,None
    if us[0]>=2*fee or drought<7 or obs['step']<18:return action,None
    homes=set(us[1].values());rival_bases=set(them[1].values())
    if not homes or us[0]>=fee and len(us[2])+1>=len(them[2]):return action,None
    live={p for p,v,k in them[2].values() if k=='COLLECTOR'}
    land={uid:int(_V7_L[p,WIDX.get(action.get(uid),0)]) for uid,(p,_,_) in us[2].items()}
    taxes={}
    for uid,(q,age) in them[3].items():
        y,x=divmod(q,15)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                z=((y+dy)%15)*15+(x+dx)%15
                taxes[z]=taxes.get(z,0.)+config.get('collectorAbsorptionRate',.1)
    offers=[]
    for uid,(p,load,kind) in us[2].items():
        if uid in protected or kind!='COLLECTOR' or load<fee or min(_V7_D[p,b] for b in homes)<=2:continue
        reserved={z for w,z in land.items() if w!=uid}
        for home in homes:
            length=int(_V7_D[p,home])
            if length not in (3,4) or length>LAST_TURN+1-obs['step']:continue
            paths=[(p,load,())]
            for tick in range(1,length):
                front=[]
                for prev,cargo,trace in paths:
                    for j,q in enumerate(_V7_L[prev,1:],1):
                        q=int(q)
                        if _V7_D[q,home]!=length-tick or q in rival_bases:continue
                        if tick==1 and q in reserved:continue
                        # Existing enemies can move once per turn; a new recruit starts at its depot.
                        if any(_V7_D[q,z]<=tick for z in live) or any(_V7_D[q,z]<tick for z in rival_bases):continue
                        net=cargo*max(0.,1.-taxes.get(q,0.))
                        if net>=fee:front.append((q,net,trace+(q,)))
                paths=front
            for q,net,trace in paths:
                offers.append((net/(1+.22*length),int(_V7_L[p,WIDX.get(action.get(uid),0)])==trace[0],uid,trace+(home,),home,length,net))
    if not offers:return action,None
    _,_,uid,tiles,home,length,net=max(offers)
    out=_v7_follow_corridor(obs,action,uid,tiles)
    if out is None or any(out.get(u)!=action.get(u) for u in protected):return action,None
    if memory is not None:memory.setdefault('corridor_track',[]).append((obs['step']+1,uid,tiles[1:]))
    return out,(uid,home,length,net)


def build(code):
    marker='def _act(obs, configuration=None, t0=None):'
    spot='        act=_v7_short_deposit(obs,configuration,act,drought)'
    if code.count(marker)!=1 or code.count(spot)!=1:raise ValueError('unknown depot chain')
    return code.replace(marker,inspect.getsource(_v7_follow_corridor)+'\n\n'+inspect.getsource(_v7_corridor)+'\n\n'+marker).replace(spot,
        spot+'\n        act,_=_v7_corridor(obs,configuration,act,drought,st)')
