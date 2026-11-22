"""A Master-family cash recovery proposal from the hidden maps already at the root."""
import argparse
import inspect
import io
import base64
from pathlib import Path
import numpy as np

MODEL=Path(__file__).with_name('master-model.npz')
ROT_ANCHOR='    sides = []\n    for side, player in enumerate((mine, rival)):'
ROOT_ANCHOR='    _shadow_update(obs,x,opponent_x,state,mine,econ,styles,style,rival)'


def _ms_base_rows(players,side,t,view,heads,mine,hidden):
    own=players[side];foe=players[1-side];units=own[2];bid=list(own[1])
    bp=np.array(list(own[1].values()),int);pos=np.array([v[0] for v in units.values()],int)
    planes=view.reshape(NPL,N);_,ba,offset=heads
    center=lambda z:np.clip(z-z.mean(1,keepdims=True),-12.,12.)
    bz=np.stack((mine[1][bp],ba[bp]+np.stack([offset[u][1][1] for u in bid]),
                 ba[bp]+np.stack([offset[u][1][2] for u in bid])))
    base_delta=np.stack([center(z) for z in bz],axis=2)
    n=len(bp);bs=_MS_LAND[bp];cash=own[0];nt=len(own[3])
    nc=sum(k=='COLLECTOR' for p,v,k in units.values());np_=len(units)-nc
    nearest=planes[39,pos]*14. if len(pos) else np.array([])
    cargo=sum(v for p,v,k in units.values())
    nearcargo=sum(v for (p,v,k),d in zip(units.values(),nearest) if d<=2)
    static=np.tile(np.array([1.,min(cash/100,3),cash<60,cash<90,min(nc,10)/8,np_/3,nt/8, t<3,t<14,t<40,
                            min(1.,t/100.),min(cargo/160,2.),min(nearcargo/90,2.),len(foe[2])/10.]),(n,1))
    local=np.stack((bp==bp.min(),planes[2,bp],planes[4,bp],planes[42,bp],planes[0,bs].mean(1),
                    planes[0,bs].max(1),planes[40,bp],planes[44,bp]),axis=1)
    vals=np.concatenate((static,local,hidden),axis=1)
    actions=np.arange(3)
    bx=np.concatenate((base_delta,vals[:,None,:]*(actions==1)[None,:,None],
                       vals[:,None,:]*(actions==2)[None,:,None]),axis=2).astype(np.float32)
    return bz,bx


def _ms_rows(players,side,t,view,previous,heads,mine,latent):
    own=players[side];foe=players[1-side];units=own[2];planes=view.reshape(NPL,N)
    uid=list(units);bid=list(own[1]);pos=np.array([units[u][0] for u in uid],int)
    bp=np.array(list(own[1].values()),int)
    allpos=np.concatenate((pos,bp))
    # Rotated hidden maps only need to be read on the five candidate cells.
    sites=_MS_LAND[allpos]
    h=sum(latent[j,cells[sites]] for j,(cells,_,_,_) in enumerate(_V5_ROTATIONS))*.25
    center=lambda z:np.clip(z-z.mean(1,keepdims=True),-12.,12.)
    wa,ba,offset=heads
    if len(pos):
        worker_z=np.stack((mine[0][pos],wa[pos]+np.stack([offset[u][1][1] for u in uid]),
                          wa[pos]+np.stack([offset[u][1][2] for u in uid])))
        role=np.array([units[u][2]=='PLANTER' for u in uid],int)
        q=sites[:len(pos)];at=lambda i:planes[i,q]
        wait=np.zeros(q.shape);wait[:,0]=1.
        loads=np.array([min(2.5,units[u][1]/90.) for u in uid])[:,None]
        toward=(at(39)[:,:1]-at(39))*14.
        danger=planes[25:30,pos].T;tie=planes[30:35,pos].T
        momentum=np.zeros_like(wait)
        if previous:
            for i,u in enumerate(uid):
                old=previous[side][2].get(u)
                if old and pos[i] in _ROLLOUT_LANDING[old[0]]:momentum[i,_ROLLOUT_LANDING[old[0]].index(pos[i])]=1.
        poor=own[0]<80;rich=loads>=.65;open_=t<40
        physical=np.stack((center(worker_z[0]),center(worker_z[1]),center(worker_z[2]),wait,momentum,toward,
             toward*loads,toward*rich*poor,wait*at(0),at(0),wait*loads,wait*poor,at(12)*loads,
             danger,tie,danger*loads,at(10),at(8),at(2)*(1-wait),at(4)*(1-wait),
             (at(40)[:,:1]-at(40))*14.,at(42),at(44),wait*open_,toward*loads*open_),axis=2)
        hu=h[:len(pos)]
        wx=np.concatenate((physical,hu-hu[:,:1],wait[:,:,None]*hu),axis=2).astype(np.float32)
    else:
        role=np.empty(0,int);worker_z=np.empty((3,0,5));wx=np.empty((0,5,217),np.float32)
    bz,bx=_ms_base_rows(players,side,t,view,heads,mine,h[len(pos):,0])
    return uid,bid,role,worker_z,bz,wx,bx


def _ms_prepare(state,obs,x,heads,mine,latent):
    side=obs['player'];t=obs['step'];pl=obs['players'];own=pl[side]
    if t==1:
        nc=sum(k=='COLLECTOR' for p,v,k in own[2].values())
        state['ms_style']=2 if nc>=2 else (1 if nc==0 else 0)
    state['ms_hiring']=None
    if t<2 or t>75 or not 30<=own[0]<145 or len(own[2])>=10:return
    sites=np.asarray(list(own[1].values()),int)
    hidden=sum(latent[j,cells[sites]] for j,(cells,_,_,_) in enumerate(_V5_ROTATIONS))*.25
    _,row=_ms_base_rows(pl,side,t,x,heads,mine,hidden)
    weights=_MS_M['w2'];specific=_MS_M['s2'][state.get('ms_style',0)]
    state['ms_hiring']=row@weights+row[:,:,:len(specific)]@specific


def _ms_recruits(obs,base,logits):
    if logits is None:return None
    own=obs['players'][obs['player']];sites=list(own[1].items())
    if len(sites)!=2:return None
    commands=[None,'RECCOLLECTOR','RECPLANTER']
    initial=[commands.index(base.get(k)) for k,p in sites]
    land={_ROLLOUT_LANDING[p][WIDX.get(base.get(u),0)] for u,(p,v,k) in own[2].items()}
    budget=min(10-len(own[2]),int(own[0]//30))
    original=sum(logits[i,a] for i,a in enumerate(initial))
    best=original+.55;choice=None
    for first in range(3):
        for second in range(3):
            pair=(first,second)
            if sum(a!=0 for a in pair)>budget:continue
            if any(a and p in land for a,(k,p) in zip(pair,sites)):continue
            score=logits[0,first]+logits[1,second]
            if score>best:best=score;choice=pair
    if choice is None:return None
    proposal=dict(base)
    for (uid,p),a in zip(sites,choice):
        if a:proposal[uid]=commands[a]
        else:proposal.pop(uid,None)
    return proposal


def build(code):
    """Offer one alternative pair of affordable depot commands, preserving every worker move."""
    anchor='def _v5_root(obs, est, x, previous):'
    search="    if most >= 3:\n        for aw, ab in _ST['v5_history']['alternatives']"
    if any(code.count(k)!=1 for k in (anchor,ROT_ANCHOR,ROOT_ANCHOR,search)) or '_MS_M=' in code:raise ValueError('Incompatible root')
    with np.load(MODEL) as trained:weights={k:trained[k] for k in ('w2','s2')}
    buf=io.BytesIO();np.savez_compressed(buf,**weights)
    blob=base64.b64encode(buf.getvalue()).decode()
    const="import base64 as _ms64, io as _msio\n_MS_M=dict(np.load(_msio.BytesIO(_ms64.b64decode("+repr(blob)+"))))\n_MS_LAND=np.asarray([[move(p,a) for a in range(5)] for p in range(225)])\n"
    methods='\n\n'.join(inspect.getsource(f) for f in (_ms_base_rows,_ms_prepare,_ms_recruits))
    code=code.replace(anchor,const+methods+'\n\n'+anchor)
    code=code.replace(ROT_ANCHOR,"    _ST['v5_history']['ms_hidden']=latent[::2]\n"+ROT_ANCHOR)
    code=code.replace(ROOT_ANCHOR,ROOT_ANCHOR+"\n    _ms_prepare(state,obs,x,own,mine,state.pop('ms_hidden'))")
    hook="""    if most>=3 and len(cands)<most:
        master=_ms_recruits(obs,act0,_ST['v5_history'].get('ms_hiring'))
        if master is not None:
            marker=tuple(sorted(master.items()))
            if marker not in keys:
                keys.add(marker);cands.append(master)
"""
    return code.replace(search,hook+search)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();src=build(a.parent.read_text());compile(src,str(a.output),'exec')
    with a.output.open('x') as f:f.write(src)
