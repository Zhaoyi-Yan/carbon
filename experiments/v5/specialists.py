"""Select a specialist for a collector or planter from its local assignment."""
def _v5_assign_features(players,carbon,step,last,side,uid):
    pos,load,kind=players[side][2][uid]
    other=1-side;pl=players[side]
    home=_v5_home(tuple(pl[1].values()))[pos]
    away=_v5_home(tuple(players[other][1].values()))[pos]
    rivals=players[other][2].values()
    contest=sum(_V5_DIST[pos][q]<=2 for q,v,k in rivals if k=='COLLECTOR')
    light=min([v for q,v,k in rivals if k=='COLLECTOR' and _V5_DIST[pos][q]<=2]+[500.])
    enemy_tree=min([_V5_DIST[pos][q] for q,age in players[other][3].values()]+[9])
    stock=(carbon[pos]+sum(carbon[q] for q in _V5_NEAR_FOUR[pos])*.5)/100.
    left=last+1-step
    return np.array([1.,step/max(1,last),max(0.,1.-left/50.),min(load/150.,3.),home/8.,away/8.,
                     (away-home)/8.,min(3,contest)/3.,float(light<load),float(enemy_tree<=2),
                     min(9,enemy_tree)/9.,stock,np.clip((pl[0]-players[other][0])/1000.,-3.,3.),
                     len(pl[3])/12.,float(left<=home+8)])


_V5_NEAR_FOUR=[[(p//15+dr)%15*15+(p%15+dc)%15 for dr,dc in ((-1,0),(0,1),(1,0),(0,-1))] for p in range(225)]


def _v5_specialist_mix(obs,carbon,x,style,own,base):
    me=obs['player'];players=obs['players'];pl=players[me];state=_ST['v5_history']
    workers,recruit=base
    votes={};choices={}
    previous=state.get('assignments',{})
    for uid,(pos,_,kind) in pl[2].items():
        role=int(kind=='PLANTER')
        context=_v5_assign_features(players,carbon,obs['step'],LAST_TURN,me,uid)
        merit=(_V5['unit_route'][role]@context-.70*_V5['unit_center'][role]
               -.024*np.log1p(np.arange(20)))
        merit[style]+=.018
        if uid in previous:merit[previous[uid]]+=.015
        expert=int(merit.argmax())
        advantage=max(0.,merit[expert]-merit[style])
        strength=min(.56,.18+advantage*1.6) if expert!=style else 0.
        choices[uid]=expert,strength
        if strength:votes[expert]=votes.get(expert,0.)+strength
    state['assignments']={uid:t for uid,(t,s) in choices.items()}
    batch=sorted(votes,key=votes.get,reverse=True)[:3]
    if not batch:return workers,recruit
    hidden=_net()(np.repeat(x[None],len(batch),axis=0),np.array(batch),hidden=True)
    w=_POLICY.p['hw.w'].reshape(5,-1);b=_POLICY.p['hw.b']
    out=workers.copy()
    for uid,(expert,strength) in choices.items():
        pos=pl[2][uid][0]
        if expert in batch:
            logits=hidden[batch.index(expert),pos]@w.T+b
            out[pos]=(1.-strength)*out[pos]+strength*logits
    return out,recruit
