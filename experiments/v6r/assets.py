"""Price currently carried carbon and each existing tree from the public board."""
def _r6_assets(players, trees, side, step, last):
    own, rival = players[side], players[1-side]
    turns=max(0,last+1-step)
    home=_v5_home(tuple(own[1].values()))
    enemy=[(p,v) for p,v,k in rival[2].values() if k=='COLLECTOR']
    light=[p for p,v,k in own[2].values() if k=='COLLECTOR' and v<25]
    enplan=[p for p,v,k in rival[2].values() if k=='PLANTER']
    ownplan=[p for p,v,k in own[2].values() if k=='PLANTER']
    def near(ds):return min(ds, default=15)
    cargo={};woods={}
    for uid,(p,v,k) in own[2].items():
        if k!='COLLECTOR' or v<=2.:continue
        killers=[q for q,w in enemy if w<=v]
        threat=near(_V5_DIST[p][q] for q in killers);distance=home[p]
        protected=near(_V5_DIST[p][q] for q in light if q!=p)
        # Distance to a depot penalized when a lighter opponent camps the entrance.
        route=near(_V5_DIST[p][b]+max(0,3-near(_V5_DIST[b][q] for q in killers)) for b in own[1].values())
        h=min(1.,turns/28.);risk=min(1.,max(0.,3-threat)/3.)
        cargo[uid]=[1.,distance/10.,route/10.,min(1.,threat/5.),min(1.,protected/5.),risk,
                    float(distance<=1),float(distance<=3),float(route<=4),float(v>110),
                    min(v,800.)/300.,float(own[0]<35),h,float(distance>turns),
                    risk*min(v,300.)/200.,max(0.,distance+2-turns)/12.,min(1.,len(enemy)/7.)]
    for uid,(p,age) in own[3].items():
        life=min(max(0,50-age),turns)
        rate=min(24.,trees.get(uid,(None,0))[1])
        challenger=near(_V5_DIST[p][q] for q in enplan)
        guard=near(_V5_DIST[p][q] for q in light)
        friend=near(_V5_DIST[p][q] for q in ownplan)
        horizon=min(life,36)/36.;risk=max(0.,1.-challenger/4.)
        woods[uid]=[horizon,min(life,12.)/12.,horizon*rate,min(life,12.)/12.*rate,
                    float(age<=2)*horizon,horizon*risk,horizon*rate*risk,
                    horizon*min(1.,guard/4.),horizon*min(1.,friend/4.),
                    horizon*float(guard<challenger),horizon*home[p]/10.,horizon*min(1.,len(enplan)/3.),
                    horizon*float(rival[0]<20),horizon*max(0.,1-rate/3.)]
    return cargo,woods


def _r6_asset_score(board,side):
    players=board['players'];t=board['step'];state=board.get('_st')
    meta=state.trees_meta if state is not None else board.get('trees',{})
    valuations=[]
    for me in (side,1-side):
        p=players[me];car,woods=_r6_assets(players,meta,me,t,LAST_TURN)
        c=sum(p[2][uid][1]*min(1.,max(-.45,row@_R6_CARGO)) for uid,row in car.items())
        tree=sum(float(np.clip(row@_R6_TREE,-90,300.)) for row in woods.values())
        # Marginal replacement value falls away as the scoring deadline approaches.
        horizon=min(1.,max(0,LAST_TURN+1-t)/24.)
        collectors=sum(k=='COLLECTOR' for _,_,k in p[2].values())
        workforce=34.*min(7,collectors)+46.*min(3,len(p[2])-collectors)
        valuations.append(p[0]+c+tree+horizon*workforce)
    return (valuations[0]-valuations[1])/3000.
