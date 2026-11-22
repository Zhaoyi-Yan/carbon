"""Hold an inflated planting fee when a late tree has little soil or passing freight left to tax."""
import inspect


def _v7_late_forest(obs,cfg,stock,action):
    s=obs['player'];us,them=obs['players'][s],obs['players'][1-s]
    left=LAST_TURN+1-obs['step']
    if left>12:return action
    homes={p for team in obs['players'] for p in team[1].values()}
    trees=[(q,a) for team in obs['players'] for q,a in team[3].values()]
    alive={q for q,a in trees if a<cfg.get('treeLifespan',50)}
    occupied={q for q,_ in trees}|homes
    cost=cfg.get('plantCost',20)
    if len(us[3])>cfg.get('playerTreeProtectiveNumber',0):
        cost+=cfg.get('plantCostInflationRatio',5)*cfg.get('plantCostInflationBase',1.235)**len(alive)
    balance=us[0]-sum(cfg.get('recCollectorCost',30) if action[k]=='RECCOLLECTOR' else cfg.get('recPlanterCost',30) for k in us[1] if k in action)
    if balance<cost:return action
    hunters=[(q,v) for q,v,k in them[2].values() if k=='COLLECTOR']
    sites={u:int(_V7_L[q,WIDX.get(action.get(u),0)]) for u,(q,c,k) in us[2].items()}
    reserved={q for u,q in us[1].items() if u in action}
    out=action
    for u,(q,_,kind) in us[2].items():
        if kind!='PLANTER' or u in action or q in occupied:continue
        tax=cfg.get('cellAbsorptionRate',.0375)
        soil=sum(max(0.,float(stock[z])) for z in NB[q] if z not in alive)
        ground=soil*(1.-(1-tax)**max(0,left-1))
        freight=0.
        for p,v in hunters:
            dist=_V7_D[q,p]
            if dist<=left+1:freight+=min(v,180.)*.1*min(3,left-1)*(.63**max(0,dist-1))
        if ground+freight+max(4.,.20*cost)>=cost:continue
        blocked={z for a,z in sites.items() if a!=u}|homes|reserved
        options=[]
        for a,z in enumerate(_V7_L[q,1:],1):
            if z in blocked:continue
            hostile=min((_V7_D[z,w] for w,_ in hunters),default=16)
            if hostile<=1 and left>=2:continue
            gain=0.
            for tid,(p,age) in them[3].items():
                pay=obs.get('trees',{}).get(tid,(None,0))[1]
                gain=max(gain,pay*min(50-age,max(0,left-1))*max(0.,1.-_V7_D[z,p]/4.))
            options.append((gain+.04*min(hostile,6),-a,int(z)))
        if options:
            _,na,destination=max(options)
            if out is action:out=dict(action)
            out[u]=WACT[-na];sites[u]=destination
    return out


def build(code):
    marker='def _act(obs, configuration=None, t0=None):';spot="    st['tr'].set_actions(players, act)"
    if code.count(marker)!=1 or code.count(spot)!=1:raise ValueError('unexpected action return')
    return code.replace(marker,inspect.getsource(_v7_late_forest)+'\n\n'+marker).replace(spot,'    if configuration is not None:act=_v7_late_forest(obs,configuration,est,act)\n'+spot)
