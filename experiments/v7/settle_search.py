"""Resolve close final positions using paired cash-at-terminal rollouts for both teams."""
import inspect


def _v7_finish_delivery(players,side,now,act,force):
    left=LAST_TURN+1-now;us,them=players[side],players[1-side]
    home=list(us[1].values());h=_V7_D[home].min(0);foe_homes=set(them[1].values())
    out=dict(act);booked={p for bid,p in us[1].items() if bid in out}
    if left<=2:
        for bid in us[1]:out.pop(bid,None)
        booked=set()
    hunters=[(p,load) for p,load,k in them[2].values() if k=='COLLECTOR']
    units=sorted(us[2].items(),key=lambda row:-row[1][1])
    for uid,(p,load,k) in units:
        a=WIDX.get(out.get(uid),0);q=int(_V7_L[p,a]);dist=h[p]
        if k!='COLLECTOR' or not load or not dist or dist>left or (left>dist+force and not (dist==1 and load>=70)):
            if q not in home:booked.add(q)
            continue
        options=[]
        for m,z in enumerate(_V7_L[p,1:],1):
            if h[z]>=dist or z in foe_homes or z in booked and z not in home:continue
            risk=sum(v<=load and _V7_D[z,w]<=1 for w,v in hunters) if z not in home else 0
            options.append((-risk,int(m==a),m))
        if not options:continue
        _,_,best=max(options);q=int(_V7_L[p,best]);out[uid]=WACT[best]
        if q not in home:booked.add(q)
        elif q in booked:
            for bid,pos in us[1].items():
                if pos==q:out.pop(bid,None)
    return out


def _v7_settle_end(obs,cfg,stock,base,lw,lb,t0,fixed=False):
    side=obs['player'];other=1-side;t=obs['step'];left=LAST_TURN+1-t;pl=obs['players']
    if left<1 or left>7:return None
    if abs(pl[side][0]-pl[other][0])>300+sum(c for team in pl for _,c,k in team[2].values()):return None
    org=_v7_finish_delivery(pl,side,t,base,0);sprint=_v7_finish_delivery(pl,side,t,base,3)
    actions=[];keys=set()
    def offer(a):
        if a is None:return
        k=tuple(sorted(a.items()))
        if k not in keys:keys.add(k);actions.append(a)
    offer(base);offer(org);offer(sprint)
    offer(_v7_plan(obs,cfg,stock,org,'intercept'))
    if any(b in org for b in pl[side][1]):offer({u:c for u,c in org.items() if u not in pl[side][1]})
    rng=np.random.RandomState(19731+t)
    for a in _deviations(pl,side,lw,org,{},3):offer(a)
    if len(actions)<2:return None
    actions=actions[:(4 if left>=5 else 6)]
    w,b=_ST['v5_history']['opponent'];replies=[]
    for s in (0,1):
        a=_sample(pl,other,w,b,np.random.RandomState(8127+t+13*s),t)
        replies.append(a if s==0 else _v7_finish_delivery(pl,other,t,a,3))
    board={'players':pl,'step':t,'player':side,'carbon':list(stock),'trees':obs.get('trees',{})}
    jobs=[]
    for ours in actions:
        for opp in replies:
            pair=[ours,opp] if side==0 else [opp,ours];jobs.append([_turn(board,cfg,pair),pl])
    moving=.035
    ids=_v7_rollout_ids([i for _ in jobs for i in (0,1)])
    for tick in range(1,left):
        start=_time.perf_counter()
        if not fixed and start-t0+max(.035,1.35*moving)>.75:return None
        views=[]
        for o,prev in jobs:views.extend(_v5_pair(o['players'],o['step'],o['carbon'],prev))
        wa,ba,_=_V6R_TINY(np.array(views),ids)
        for j,(o,prev) in enumerate(jobs):
            future=o['players'];tt=t+tick;moves=[]
            for i in (0,1):
                rng=np.random.RandomState(19731+tick*149+17*i+23*(j%2))
                a=_sample(future,i,wa[2*j+i],ba[2*j+i],rng,tt)
                moves.append(_v7_finish_delivery(future,i,tt,a,(j%2)*2))
            jobs[j]=[_turn(o,cfg,moves),future]
        moving=.55*moving+.45*(_time.perf_counter()-start)
    outcomes=[]
    for o,_ in jobs:
        cash=o['players'];sign=o.get('_terminal_sign')
        value=(cash[side][0]-cash[other][0]) if sign is None else sign*(1-2*side)*10000.
        outcomes.append(value)
    out=np.asarray(outcomes).reshape(-1,2)
    relative=out-out[0]
    merit=.60*relative.mean(axis=1)+.40*relative.min(axis=1)
    merit[1:]-=7.+.64*max(0,left-2)
    best=int(merit.argmax())
    return actions[best]


def build(code):
    anchor='def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):'
    start='    net = _net()\n    M = _S_M'
    if code.count(anchor)!=1 or code.count(start)!=1:raise ValueError('unknown search')
    helpers='\n\n'.join(inspect.getsource(f) for f in (_v7_finish_delivery,_v7_settle_end))
    return code.replace(anchor,helpers+'\n\n'+anchor).replace(start,
        "    if LAST_TURN+1-step<=7:\n        settlement=_v7_settle_end(obs,configuration,est,act0,lw0,lb0,t0)\n        if settlement is not None:\n            _ST['v7_terminal_step']=step\n            return settlement\n"+start).replace(
        'if _SEARCH_OK and _RX and t0 is not None:',"if _SEARCH_OK and _RX and t0 is not None and st.get('v7_terminal_step')!=step:")
