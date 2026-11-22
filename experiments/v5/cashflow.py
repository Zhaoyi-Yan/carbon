"""Value the ability to keep earning during a fight, including cash needed to restart."""
def _v5_funding(players,trees,side,step,last):
    pl=players[side];enemy=players[1-side]
    left=max(0,last+1-step);cash=pl[0]
    collectors=[(p,v) for p,v,r in pl[2].values() if r=='COLLECTOR']
    planters=len(pl[2])-len(collectors)
    hunters=[(p,v) for p,v,r in enemy[2].values() if r=='COLLECTOR']
    home=_v5_home(tuple(pl[1].values()))
    receipts=standing=securable=0.
    for uid,(p,age) in pl[3].items():
        life=max(0,50-age)
        rate=trees.get(uid,(None,0))[1]
        receipts+=rate*min(1.,life/7.)
        standing+=min(1.,life/14.)
    dependable=0.;trapped=0.;survivors=0
    for p,v in collectors:
        conflict=any(h<v and _V5_DIST[p][q]<=2 for q,h in hunters)
        survivors+=not conflict
        if home[p]<=min(9,left):
            dependable+=v/(1.+.15*home[p])*(.35 if conflict else 1.)
        if conflict:trapped+=v
    for p in pl[1].values():
        securable+=any(_V5_DIST[q][p]<=2 and h<30 for q,h in hunters)
    bare=float(not pl[3] and not collectors and cash<20.)
    empty=float(not pl[3] and not collectors and not planters and cash<30.)
    reserve=min(1.,cash/30.)
    ready=min(1.,(cash+4.*receipts+.70*dependable)/30.)
    shortage=1.-ready
    return np.array([bare,empty,1.-reserve,shortage,shortage/(1.+len(collectors)),
                     1./(1.+len(collectors)),1./(1.+survivors),1./(1.+standing),
                     1./(1.+receipts/4.),min(3.,dependable/100.),min(3.,trapped/100.),
                     securable/max(1,len(pl[1])),float(planters==0),float(cash<30.)/(1.+receipts),
                     min(2.,receipts/12.)/(1.+len(collectors))])


def _v5_funding_features(pl,trees,step,last,side):
    delta=_v5_funding(pl,trees,side,step,last)-_v5_funding(pl,trees,1-side,step,last)
    left=max(0,last+1-step)
    return np.r_[delta,delta*min(1.,left/50.),delta*(step/max(1,last))]


def _v5_cashflow_eval(board,me):
    step=board['step'];pl=board['players'];st=board.get('_st')
    meta=st.trees_meta if st is not None else board.get('trees',{})
    delta=_v5_funding_features(pl,meta,step,LAST_TURN,me)
    horizon=min(1.,max(0.,LAST_TURN+1-step)/14.)
    correction=np.clip(delta@_V5['cashflow'],-.80,.80)/3.
    return _v5_territory_value(board,me)+horizon*correction


def _v5_income_move(root,config,act0,fixed,worker_logits,seed):
    pl=root['players'];me=root['player'];op=1-me;step=root['step']
    meta=root.get('trees',{})
    def pressured(side):
        p=pl[side];rate=sum(meta.get(uid,(None,0))[1] for uid in p[3])
        nc=sum(k=='COLLECTOR' for _,_,k in p[2].values())
        return (p[0]<125 and rate<8. and (nc<=4 or len(p[3])<=2))
    vulnerable=pressured(me);enemy_weak=pressured(op)
    if not vulnerable and not enemy_weak:return None
    ow,ob=_ST['v5_history']['opponent']
    reply=_sample(pl,op,ow/.65,ob/.65,np.random.RandomState(seed+47),step)
    if not any(pl[side][2] for side in (0,1)):return None
    options=[];home=_v5_home(tuple(pl[me][1].values()))
    own=pl[me];rival=pl[op]
    enemy_homes=set(rival[1].values())
    treemap={p for p,age in rival[3].values()}
    for uid,(p,load,role) in own[2].items():
        if uid in fixed:continue
        old=WIDX.get(act0.get(uid),0)
        for k,dest in enumerate(_ROLLOUT_LANDING[p]):
            if k==old:continue
            merit=-2.
            if vulnerable and role=='COLLECTOR' and load>=20 and home[dest]<home[p]:merit=1.+min(2.,load/80.)
            if enemy_weak and role=='PLANTER' and p in treemap and k==0:merit=2.4
            if enemy_weak and role=='COLLECTOR':
                for eid,(q,v,kind) in rival[2].items():
                    landing=_ROLLOUT_LANDING[q][WIDX.get(reply.get(eid),0)]
                    if dest==landing and (kind=='PLANTER' or load<v):merit=max(merit,2.+min(2.,v/80.))
            if (enemy_weak and role=='COLLECTOR' and load>=20 and _ROLLOUT_LANDING[p][old] in enemy_homes
                    and dest not in enemy_homes):
                merit=max(merit,3.+min(2.,load/100.))
            if merit>0:
                options.append((merit+.12*float(worker_logits[p,k]-worker_logits[p,old]),uid,k))
    if vulnerable and own[0]>=30 and not any(k=='COLLECTOR' for _,_,k in own[2].values()):
        for uid in own[1]:options.append((6.,uid,1))
    if not options:return None
    def score(actions):
        pair=[actions,reply] if me==0 else [reply,actions]
        return _v5_cashflow_eval(_turn(root,config,pair),me)
    best=None;value=score(act0)
    for _,uid,k in sorted(options,reverse=True)[:8]:
        action=act0.copy()
        if k:action[uid]=BACT[k] if uid in own[1] else WACT[k]
        else:action.pop(uid,None)
        s=score(action)
        if s>value+.0006:best,value=action,s
    return best
