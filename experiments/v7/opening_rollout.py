"""Compare first-turn recruiting plans using paired legal-view continuations."""
import inspect


def _v7_launch(obs,config,stock,base,t0,fixed=False):
    side=obs['player'];other=1-side;pl=obs['players']
    if obs['step'] or any(p[2] or len(p[1])!=2 for p in pl):return base,None
    us=list(pl[side][1]);them=list(pl[other][1]);default=[base.get(u) for u in us]
    if set(default)!={'RECCOLLECTOR','RECPLANTER'}:return base,None
    choices=[base,dict(zip(us,default[::-1])),dict.fromkeys(us,'RECCOLLECTOR'),{}]
    opponents=[{},dict.fromkeys(them,'RECCOLLECTOR')]
    root={'step':0,'player':side,'players':pl,'trees':obs.get('trees',{}),'carbon':list(stock)}
    jobs=[];history=[];last_cost=.045
    for own in choices:
        for theirs in opponents:
            a=[own,theirs] if side==0 else [theirs,own]
            jobs.append([_turn(root,config,a),pl])
    ids=np.array([22 if s==side else 20 for _ in jobs for s in (0,1)])
    for tick in range(1,14):
        start=_time.perf_counter()
        if not fixed and start-t0+max(.045,last_cost*1.6)>.72:break
        views=[]
        for b,prev in jobs:views.extend(_v5_pair(b['players'],b['step'],b['carbon'],prev))
        w,h,_=_V6R_TINY(np.asarray(views),ids)
        for i,(b,prev) in enumerate(jobs):
            players=b['players'];moves=[]
            for s in (0,1):
                rng=np.random.RandomState(2453+23*tick+11*s)
                action=_sample(players,s,w[2*i+s],h[2*i+s],rng,tick)
                if tick==1 and not players[s][2]:
                    action=_r6_opening({'step':tick,'players':players,'player':s},config,h[2*i+s],action)
                moves.append(action)
            jobs[i]=[_turn(b,config,moves),players]
        if tick in (3,6,9,12,13):
            values=np.array([_r6_asset_score(b,side) for b,_ in jobs]).reshape(len(choices),len(opponents))
            history.append((tick,values))
        last_cost=.55*last_cost+.45*(_time.perf_counter()-start)
    if not history or history[-1][0]<6:return base,None
    # Only value improvements durable across adjacent rollout horizons. Both opening opponents
    # share random numbers for every comparison. A marginal forecast keeps the established plan.
    mature=[x for _,x in history[-2:]]
    values=np.mean(mature,axis=0)
    relative=values-values[0]
    scores=.70*relative.mean(axis=1)+.30*relative.min(axis=1)
    scores[1:]-=.011
    scores[2:]-=.011
    best=int(scores.argmax())
    return choices[best],dict(choice=best,ticks=history[-1][0],margin=[round(float(v)*3000,2) for v in scores])


def build(code):
    marker='def _act(obs, configuration=None, t0=None):'
    anchor='    if configuration is not None:act=_v7_early_income(obs,configuration,lb,act)'
    search='    if not _SEARCH_OK or step < _S_START or step > LAST_TURN:'
    if any(code.count(s)!=1 for s in (marker,anchor,search)):raise ValueError('unrecognized opening')
    code=code.replace(marker,inspect.getsource(_v7_launch)+'\n\n'+marker)
    code=code.replace(search,'    if step==0:return act0\n'+search)
    return code.replace(anchor,anchor+"\n    if not step and _SEARCH_OK and configuration is not None:\n        act,st['opening_comparison']=_v7_launch(obs,configuration,est,act,t0)")
