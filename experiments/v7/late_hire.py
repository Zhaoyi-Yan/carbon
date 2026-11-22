"""Price the time available for a last collector to make a delivery after leaving its depot."""
import inspect


def _v7_hire_features(obs,cfg,stock,side,gate):
    us,them=obs['players'][side],obs['players'][1-side];base=us[1][gate];remaining=LAST_TURN+1-obs['step']
    distance=_V7_D[base];home=_V7_D[list(us[1].values())].min(0)
    blocked={p for pl in obs['players'] for p in pl[1].values()}
    blocked.update(q for pl in obs['players'] for q,age in pl[3].values())
    field=np.maximum(0.,np.asarray(stock,dtype=float));field=field.copy();field[list(blocked)]=0.
    # The recruit is created stationary on its depot. At least one turn is spent on each outward
    # and return step. A nearby rich square can still fund a late recruit.
    harvests=np.maximum(0,remaining-1-distance-home)
    potential=np.minimum(100.,field*1.03**np.minimum(12,distance))*(1.-.7**harvests)
    crowd=np.zeros(225)
    for p,c,k in us[2].values():
        if k=='COLLECTOR':crowd+=(_V7_D[p]<=np.minimum(distance,2))
    attainable=float(np.max(potential/(1.+.48*crowd)))
    enemies=[(p,v) for p,v,k in them[2].values() if k=='COLLECTOR']
    nearby=max((v/(1.+.30*max(0.,distance[p]-1)) for p,v in enemies if distance[p]<=max(1,remaining-1)),default=0.)
    adjacent=max((v for p,v in enemies if distance[p]<=2),default=0.)
    allied=max((v for p,v,k in us[2].values() if k=='COLLECTOR' and distance[p]<=2),default=0.)
    return [remaining,attainable,float(nearby),float(adjacent),float(allied),len(us[2]),len(them[2])]


def _v7_hire_screen(obs,cfg,stock,action):
    side=obs['player'];us=obs['players'][side];rem=LAST_TURN+1-obs['step']
    if rem>9 or len(us[2])<2:return action
    out=dict(action)
    for bid in us[1]:
        if action.get(bid)!='RECCOLLECTOR':continue
        left,ore,traffic,near,allied,_,_=_v7_hire_features(obs,cfg,stock,side,bid)
        if left<=3 and near<35 or ore<35 and traffic<85 and near<45:out.pop(bid)
    return out


def build(code):
    marker='def _act(obs, configuration=None, t0=None):';spot="    st['tr'].set_actions(players, act)"
    if code.count(marker)!=1 or code.count(spot)!=1:raise ValueError('unrecognized action return')
    return code.replace(marker,inspect.getsource(_v7_hire_features)+'\n\n'+inspect.getsource(_v7_hire_screen)+'\n\n'+marker).replace(spot,'    if configuration is not None:act=_v7_hire_screen(obs,configuration,est,act)\n'+spot)
