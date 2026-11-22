"""Cover a productive tree when a rival planter is already standing on it."""
import inspect


def _v7_cover_tree(obs,config,action):
    side=obs['player'];us,them=obs['players'][side],obs['players'][1-side]
    if not us[3] or them[0]<config.get('seizeCost',20):return action
    intruders={q for q,_,k in them[2].values() if k=='PLANTER'}
    trees=[(uid,q,age) for uid,(q,age) in us[3].items() if q in intruders]
    if not trees:return action
    threatened={z for q,_,k in them[2].values() if k=='COLLECTOR' for z in _V7_L[q]}
    homes=set(us[1].values());enemyhomes=set(them[1].values());meta=obs.get('trees') or {}
    land={uid:int(_V7_L[p,WIDX.get(action.get(uid),0)]) for uid,(p,_,_) in us[2].items()}
    candidates=[]
    for uid,q,age in trees:
        if q in threatened or q in land.values():continue
        horizon=min(50-age,LAST_TURN+1-obs['step'],20)
        benefit=2.*max(0,horizon)*min(19.,max(0.,meta.get(uid,(None,0))[1]))
        for wid,(p,load,kind) in us[2].items():
            if kind!='COLLECTOR' or _V7_D[p,q]!=1:continue
            target=land[wid]
            if target in homes and load>=15 or target in enemyhomes:continue
            # Preserve an immediately available enemy shipment and a productive unload.
            caught=max((v for z,v,k in them[2].values() if k=='COLLECTOR' and z==target and load<v),default=0.)
            if caught>40 or benefit<max(48.,.85*load+22.):continue
            move=next(i for i,z in enumerate(_V7_L[p]) if z==q)
            candidates.append((benefit-.85*load,-load,wid,move))
    if not candidates:return action
    _,_,wid,move=max(candidates);out=dict(action);out[wid]=WACT[move]
    return out


def build(code):
    spot='        act=_v7_cash_route(obs,configuration,act,lw,enemy,drought)'
    marker='def _act(obs, configuration=None, t0=None):'
    if code.count(spot)!=1 or code.count(marker)!=1:raise ValueError('unexpected liquidity action path')
    return code.replace(marker,inspect.getsource(_v7_cover_tree)+'\n\n'+marker).replace(spot,'        act=_v7_cover_tree(obs,configuration,act)\n'+spot)
