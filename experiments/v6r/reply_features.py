"""Directional contact features observable by either player before simultaneous moves."""
import numpy as np

def _reply_features(players, side, uid, previous, geometry, landing):
    own, rival=players[side],players[1-side]
    pos,cargo,role=own[2][uid]
    rivals=[(q,v,k) for q,v,k in rival[2].values() if geometry[pos,q]<=2]
    if not rivals:
        return None
    cells=np.array(landing[pos])
    ownhome=geometry[cells][:,list(own[1].values())].min(1)
    enemyhome=geometry[cells][:,list(rival[1].values())].min(1)
    load=min(cargo,240.)/100.
    collect=role=='COLLECTOR'
    prior=previous[side][2].get(uid) if previous else None
    stay=(cells==pos).astype(float)
    momentum=np.zeros(5)
    if prior and pos in landing[prior[0]]:
        momentum[landing[prior[0]].index(pos)]=1.
    distances=geometry[cells][:,[r[0] for r in rivals]]
    loads=np.array([v for q,v,k in rivals])
    collectors=np.array([k=='COLLECTOR' for q,v,k in rivals])
    threat=collectors & ((loads<cargo) if collect else True)
    trade=collectors & (loads==cargo) & collect
    profit=(~collectors | (loads>cargo)) & collect
    useful_trees={p for p,a in rival[3].values()}
    pals=[q for i,(q,v,k) in own[2].items() if i!=uid]
    crowded=np.isin(cells,pals)
    columns=[stay,stay*load,stay*(not collect),momentum,
             (ownhome[0]-ownhome)*load,(ownhome==0)*load,enemyhome==0,crowded,
             np.isin(cells,list(useful_trees))*(not collect)]
    for category in (threat,trade,profit,~collectors & collect):
        active=distances[:,category]
        reach=(active<=1).any(1)
        here=(active==0).any(1)
        near=(active<=2).any(1)
        columns.extend([here,reach,near,(reach*load)])
    prizes=np.log1p(loads/30.)*profit
    columns.append(np.max((distances<=1)*prizes,axis=1))
    return np.stack(columns,axis=1).astype(np.float32)
