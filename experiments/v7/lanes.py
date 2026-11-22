"""Public-board alternatives for tree expansion, banking and intercepting a shipment."""
import math
import numpy as np

_V7_ACTIONS = (None, 'UP', 'RIGHT', 'DOWN', 'LEFT')
_V7_GRID = np.arange(225)
_v7_r, _v7_c = np.divmod(_V7_GRID, 15)
_v7_dy = abs(_v7_r[:, None] - _v7_r)
_v7_dx = abs(_v7_c[:, None] - _v7_c)
_V7_D = np.minimum(_v7_dx, 15-_v7_dx) + np.minimum(_v7_dy, 15-_v7_dy)
_V7_L = np.column_stack((_V7_GRID, ((_v7_r-1)%15)*15+_v7_c,
                         _v7_r*15+(_v7_c+1)%15, ((_v7_r+1)%15)*15+_v7_c, _v7_r*15+(_v7_c-1)%15))


def _v7_nearest(sites):
    return _V7_D[list(sites)].min(axis=0) if len(sites) else np.full(225, 16)


def _v7_safe_route(pos, target, original, blocked, danger, avoid):
    distance = _V7_D[target]
    score = -distance[_V7_L[pos]].astype(float)
    score -= np.minimum(2., np.maximum(0,3-danger[_V7_L[pos]]))*.53
    score -= avoid[_V7_L[pos]]
    score[original] += .21
    for a,q in enumerate(_V7_L[pos]):
        if q in blocked or danger[q]<=1 or (pos!=target and a==0):
            score[a] = -100.
    if pos==target and pos not in blocked and danger[pos]>1:
        return 0
    best = int(score.argmax())
    if score[best]< -50 or distance[_V7_L[pos,best]]>=distance[pos]:
        return None
    return best


def _v7_plan(obs, config, stock, base, mode):
    if mode not in ('tree','bank','intercept'):return dict(base)
    side=obs['player'];own,enemy=obs['players'][side],obs['players'][1-side]
    last=config.get('episodeSteps',300)-1;left=last-obs['step']
    if left<1:return dict(base)
    homes=list(own[1].values());enemy_homes=list(enemy[1].values())
    oh=_v7_nearest(homes);eh=_v7_nearest(enemy_homes)
    rival=[(p,v) for p,v,k in enemy[2].values() if k=='COLLECTOR']
    enemy_wood=[p for p,age in enemy[3].values() if age<50]
    tax=np.zeros(225)
    for p in enemy_wood:
        rr,cc=divmod(p,15)
        for dy in (-1,0,1):
            for dx in (-1,0,1):tax[((rr+dy)%15)*15+(cc+dx)%15]+=.12
    resource=np.asarray(stock,dtype=float)
    original={u:_V7_ACTIONS.index(base.get(u)) for u in own[2]}
    ends={u:_V7_L[p,original[u]] for u,(p,_,_) in own[2].items()}
    hires={p for u,p in own[1].items() if u in base}
    options=[]
    for uid,(pos,load,role) in own[2].items():
        if (mode=='tree') != (role=='PLANTER'):continue
        if mode=='bank' and (load<45 or oh[pos]>left):continue
        if mode=='intercept' and load>40:continue
        bad=[p for p,v in rival if role=='PLANTER' or v<=load]
        danger=_v7_nearest(bad)
        blocked={q for u,q in ends.items() if u!=uid}|hires|set(enemy_homes)
        if mode=='bank':
            for target in homes:
                d=_V7_D[pos,target]
                if not 0<d<=min(10,left):continue
                action=_v7_safe_route(pos,target,original[uid],blocked,danger,tax)
                if action is None:continue
                merit=load/(1.+.20*d)*(.65 if danger[target]<=2 else 1.)
                options.append((merit,uid,action,target))
        elif mode=='intercept':
            for start,prize in rival:
                ticks=int(eh[start]);cutoff=min(ticks,left,9)
                if not 2<=ticks<=10 or prize<max(60.,load+30):continue
                # Route probability at each approach tile, conditional on returning via a shortest path.
                frontier={start:1.}
                for t in range(1,cutoff+1):
                    nxt={}
                    for q,prob in frontier.items():
                        qs=[n for n in _V7_L[q,1:] if eh[n]==eh[q]-1]
                        for n in qs:nxt[n]=nxt.get(n,0.)+prob/len(qs)
                    frontier=nxt
                    for q,prob in frontier.items():
                        travel=_V7_D[pos,q]
                        if eh[q]<1 or travel>t or q in enemy_homes:continue
                        action=_v7_safe_route(pos,q,original[uid],blocked,danger,tax*.4)
                        if action is None:continue
                        merit=prize*prob*(.91**t)/(1+.27*travel+.10*(t-travel))-.18*load
                        if merit>=13:options.append((merit,uid,action,int(q)))
        else:
            enemies_on_foot=_v7_nearest([p for p,_,k in enemy[2].values() if k=='PLANTER'])
            live_trees=[(p,age) for pl in obs['players'] for p,age in pl[3].values() if age<50]
            adj=_V7_L[:,1:];overlap=np.zeros(225)
            for p,age in live_trees:overlap[adj[p]]+=min(1.,(50-age)/18.)
            reserve=own[0]-sum(config.get('recCollectorCost',30) if base[b]=='RECCOLLECTOR' else config.get('recPlanterCost',30) for b in own[1] if b in base)
            cost=config.get('plantCost',20)
            if len(own[3])>config.get('playerTreeProtectiveNumber',0):
                cost+=config.get('plantCostInflationRatio',5)*config.get('plantCostInflationBase',1.235)**len(live_trees)
            income=(resource/(1+.63*overlap))[adj].sum(1)*config.get('cellAbsorptionRate',.0375)*min(19,left*.65)
            merit=income-cost
            if reserve<cost:merit[:]=-1000.
            for key,(p,age) in enemy[3].items():
                rate=obs.get('trees',{}).get(key,(None,0.))[1]
                if reserve>=config.get('seizeCost',20):
                    merit[p]=2.*min(50-age,20,left)*rate-config.get('seizeCost',20)
            merit[list(own[1].values())+enemy_homes+[p for p,_ in own[3].values()]]=-1000.
            merit*=np.clip(.45+.14*enemies_on_foot,.45,1.)*np.clip(.32+.19*danger,.32,1.)
            merit/=1.+.41*_V7_D[pos]
            for target in np.argsort(-merit)[:6]:
                if merit[target]<10. or _V7_D[pos,target]>5:continue
                action=_v7_safe_route(pos,target,original[uid],blocked,danger,tax*0.)
                if action is not None:options.append((merit[target],uid,action,int(target)))
    result=dict(base);used=set();goals=set();arrivals=set()
    for merit,uid,action,target in sorted(options,reverse=True):
        q=_V7_L[own[2][uid][0],action]
        if uid in used or q in arrivals or (mode!='bank' and target in goals):continue
        used.add(uid);goals.add(target);arrivals.add(q)
        if action:result[uid]=_V7_ACTIONS[action]
        else:result.pop(uid,None)
        if len(used)==2:break
    return result


plan = _v7_plan
