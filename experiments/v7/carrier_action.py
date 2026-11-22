"""Conditional own-worker receipts after one candidate move; fitted only on observed commands."""
import base64
import inspect
import io
from pathlib import Path
import numpy as np

NAMES=('NONE','UP','RIGHT','DOWN','LEFT')
LAND=tuple(tuple(((p//15+dy)%15)*15+(p%15+dx)%15 for dy,dx in ((0,0),(-1,0),(0,1),(1,0),(0,-1))) for p in range(225))
DIST=tuple(tuple(min(abs(p//15-q//15),15-abs(p//15-q//15))+min(abs(p%15-q%15),15-abs(p%15-q%15)) for q in range(225)) for p in range(225))


def _ca_board(obs):
    s=obs['player'];pl=obs['players'];us,op=pl[s],pl[1-s]
    visible=set()
    for p,v,k in us[2].values():
        visible.update([p] if k=='COLLECTOR' else _CA_LAND[p][1:])
    homes=tuple(us[1].values());foes=tuple(op[1].values())
    return (us,op,homes,foes,[(p,v) for p,v,k in op[2].values() if k=='COLLECTOR'],
            {p for p,a in op[3].values()},visible,obs['carbon'],obs['step'])


def _ca_inputs(ctx,uid,moves):
    us,op,homes,foes,enemy,trees,vis,ore,t=ctx
    q,v,_=us[2][uid];d=_CA_DIST[q];near=min(d[b] for b in homes)
    allied=tuple(p for k,(p,n,r) in us[2].items() if k!=uid)
    enemy_light=[p for p,n in enemy if n<=v];enemy_heavy=[(p,n) for p,n in enemy if n>v]
    prior=[min(v,1200)/100,near,min(9,min((d[p] for p in enemy_light),default=12)),ore[q]/100,
           max(d[b] for b in homes),v/max(1,sum(w[1] for w in us[2].values())),sum(d[p]<=2 for p,n in enemy),
           min(us[0],600)/100,len(us[2]),len(op[2]),len(us[3]),len(op[3]),
           min(20,299-t),t/299,min((d[b] for b in foes),default=12),sum(d[p]<=2 for p in allied)]
    rows=[];keys=[]
    for move in moves:
        z=_CA_LAND[q][move];dz=_CA_DIST[z];h=min(homes,key=dz.__getitem__);dist=dz[h];change=near-dist
        risk=min((dz[p] for p in enemy_light),default=12)
        prey=min((dz[p] for p,n in enemy_heavy),default=12)
        path=min((dz[p]+_CA_DIST[p][h]-dist for p in enemy_light),default=12)
        block=min((_CA_DIST[p][h] for p in enemy_light),default=12)
        rich=max((n-v for p,n in enemy_heavy if dz[p]<=1),default=0.)
        nbr=_CA_LAND[z][1:];seen=[ore[k] for k in nbr if k in vis]
        new=[int(move==0),change,dist,int(z in homes),int(z in foes),min(risk,9),min(prey,9),min(path,9),
             min(block,9),min(rich,650)/100,sum(dz[p]<=1 for p in allied),sum(dz[p]<=2 for p in allied),
             sum(dz[p]<=1 or (_CA_DIST[p][z]==2 and p//15!=z//15 and p%15!=z%15) for p in trees),
             int(z in vis),ore[z]/100 if z in vis else 0,len(seen),max(seen,default=0)/100,
             int(z in allied),min(v,650)/100*change,ore[q]/100*(move==0)]
        rows.append(prior+new)
        group=int(move!=0)*(1+int(change<=0)+int(change<0))
        key=(((min(near,6)//2)*3+int(v>=130)+int(v>=250))*2+int(prior[2]<=2))*4
        keys.append((key+group,group))
    return rows,keys


def _ca_eval(model,x,kind):
    x=np.asarray(x,dtype=np.float32)
    f,th,ch,val=model['f'+kind],model['t'+kind],model['c'+kind],model['v'+kind]
    preds=np.full((len(x),),float(model['b'+kind]),dtype=np.float64)
    # All trees are traversed together; leaves point back to themselves.
    ix=np.zeros((len(x),len(f)),dtype=np.int32)
    trees=np.arange(len(f))[None,:];rows=np.arange(len(x))[:,None]
    for _ in range(int(model['depth'])):
        choice=x[rows,f[trees,ix]]>th[trees,ix]
        ix=ch[trees,ix,choice.astype(np.int32)]
    return np.maximum(0,preds+val[trees,ix].sum(1))


def _ca_proposal(obs,base,locked,previous,model):
    ctx=_ca_board(obs);us,op,homes,foes,enemy=ctx[:5];best=None;candidates=[]
    prior_workers=previous[obs['player']][2] if previous else {}
    if not homes:return None,None
    landing={u:_CA_LAND[p][_CA_NAMES.index(base.get(u) or 'NONE')] for u,(p,_,_) in us[2].items()}
    hired={p for u,p in us[1].items() if u in base}
    for uid,(q,v,k) in us[2].items():
        if k!='COLLECTOR' or v<60 or uid in (locked or {}):continue
        a=_CA_NAMES.index(base.get(uid) or 'NONE');old=landing[uid]
        d=min(_CA_DIST[q][h] for h in homes);next_d=min(_CA_DIST[old][h] for h in homes)
        if old in homes or (next_d<d and (d<=2 or prior_workers.get(uid) and min(_CA_DIST[prior_workers[uid][0]][h] for h in homes)>d)):continue
        if any(w>v+30 and (_CA_DIST[old][p]==0 or (_CA_DIST[old][p]==1 and _CA_DIST[q][p]==2 and w>=v+80)) for p,w in enemy):continue
        avoid=hired|{pos for u,pos in landing.items() if u!=uid}
        moves=[i for i,z in enumerate(_CA_LAND[q]) if i!=a and z not in avoid and z not in foes and
               not any(_CA_DIST[z][p]<=1 and w<=v for p,w in enemy)]
        if not moves:continue
        acts=[a]+moves;rows,keys=_ca_inputs(ctx,uid,acts);start=len(candidates)
        for i, (act,row,key) in enumerate(zip(acts,rows,keys)):
            bin_,group=key;freq=model['support'][bin_];tot=model['support'][bin_-group:bin_-group+4].sum()
            candidates.append((start,uid,act,row,bool(freq>=25 and freq>=.035*tot),v))
    if not candidates:return None,None
    x=[item[3] for item in candidates]
    p10=_ca_eval(model,x,'10');p20=np.maximum(p10,_ca_eval(model,x,'20'))
    for i,(start,uid,act,_,supported,v) in enumerate(candidates):
        if i==start or not supported or not candidates[start][4]:continue
        delta=.5*(p10[i]-p10[start])+.5*(p20[i]-p20[start])
        if delta<=max(15.,.08*v) or (best and delta<=best[0]):continue
        best=(delta,uid,act,start,i,v)
    if best is None:return None,None
    val,uid,act,src,i,v=best;new=dict(base)
    if act:new[uid]=_CA_NAMES[act]
    else:new.pop(uid,None)
    return new,dict(uid=uid,load=v,old=base.get(uid,'NONE'),new=_CA_NAMES[act],delta=round(val,3),
                    estimate=[[round(p10[src],3),round(p20[src],3)],[round(p10[i],3),round(p20[i],3)]])


def build(code,model=None):
    model=Path(model or Path(__file__).with_name('carrier-action-model.npz'))
    payload=base64.b64encode(model.read_bytes()).decode()
    anchor="    _mw_pair=_ST['v5_history'].get('mw_workers')\n"
    if code.count(anchor)!=1:raise ValueError('Expected a single compatible Harbormaster root')
    method='\n\n'.join(inspect.getsource(f) for f in (_ca_board,_ca_inputs,_ca_eval,_ca_proposal))
    setup="_CA_NAMES=('NONE','UP','RIGHT','DOWN','LEFT')\n_CA_LAND=tuple(tuple(int(v) for v in line) for line in _V7_L)\n_CA_DIST=_V7_RECEIPT_DIST\n"
    setup+="with np.load(io.BytesIO(base64.b64decode("+repr(payload)+")),allow_pickle=False) as _ca_archive:\n    _CA_MODEL=dict(_ca_archive)\n"
    marker='def _act(obs, configuration=None, t0=None):'
    if code.count(marker)!=1:raise ValueError('Ambiguous strategy entry')
    insertion="""    if most>=4 and len(cands)<most:
        _ca_locked=set(fixed or ())
        _ca_locked.update(uid for t,uid,_ in _ST.get('corridor_track',()) if t==step)
        _ca_alt,_=_ca_proposal(obs,act0,_ca_locked,prev,_CA_MODEL)
        if _ca_alt is not None:
            key=tuple(sorted(_ca_alt.items()))
            if key not in keys:
                cands.append(_ca_alt);keys.add(key)
"""
    return code.replace(marker,setup+method+'\n\n'+marker).replace(anchor,insertion+anchor)


_CA_NAMES=NAMES
_CA_LAND=LAND
_CA_DIST=DIST
