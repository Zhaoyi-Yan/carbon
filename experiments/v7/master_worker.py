"""Role- and style-dependent Master worker proposal using rotation-preserving root activations."""
import base64
import inspect
import io
from pathlib import Path
import numpy as np


def _mw_inputs(obs,x,previous,hidden):
    own=obs['players'][obs['player']];foe=obs['players'][1-obs['player']]
    uid=list(own[2]);pos=[own[2][u][0] for u in uid]
    roles=np.asarray([own[2][u][2]=='PLANTER' for u in uid],dtype=np.int64)
    if not uid:return uid,roles,None,None,None,None
    xy=x.reshape(NPL,N);sites=_V5_REACH[pos];geometry=_V5_GEOMETRY
    rs=np.stack([hidden[j,cells[sites]] for j,(cells,_,_,_) in enumerate(_V5_ROTATIONS)],axis=1)
    enemy=[(q,c) for q,c,k in foe[2].values() if k=='COLLECTOR']
    assets=tuple(foe[1].values());homes=tuple(own[1].values())
    wood=sum(obs.get('trees',{}).get(k,(None,0.))[1] for k in own[3])
    old=previous[obs['player']][2] if previous else {}
    cash,c2=own[0],foe[0];n=len(uid)
    loads=[own[2][u][1] for u in uid]
    g=np.asarray([min(cash,400)/200.,min(c2,400)/200.,n/10.,sum(roles)/4.,len(own[3])/12.,len(foe[3])/12.,
                  min(30.,wood)/15.,len(enemy)/8.,sum(min(v,250.) for v in loads)/600.,
                  sum(min(geometry[p,q] for q in assets)<=3 for p in pos)/n,obs['step']/299.],dtype=np.float32)
    local=[];methods=[]
    for u,p,load,kind,places in zip(uid,pos,loads,roles,sites):
        d=min(geometry[p,k] for k in homes);dg=min(geometry[p,k] for k in assets)
        prior=old.get(u,(p,load,''));progress=min(geometry[prior[0],k] for k in homes)-d
        prev_move=next((i for i,c in enumerate(_V5_REACH[prior[0]]) if c==p),0)
        dangerous=[q for q,c in enemy if c<=load];victims=[q for q,c in enemy if c>=load+35.]
        dp=min((geometry[p,q] for q in dangerous),default=15);near=min((geometry[p,q] for q in victims),default=15)
        local.append([float(kind),min(load,400.)/170.,d/7.,dg/7.,min(dp,8)/5.,min(near,8)/5.,
                      float(xy[0,p]),progress,min(100.,max(-100.,load-prior[1]))/50.,*g])
        ctx=[]
        for a,q in enumerate(places):
            dz=min(geometry[q,b] for b in homes);enemy_d=min(geometry[q,b] for b in assets)
            prey=min((geometry[q,z] for z in victims),default=15)
            ctx.append([float(a==0),float(a==prev_move),float(d-dz),float(dg-enemy_d),float(near-prey),
                        float(xy[0,q]),float(xy[25+a,p]),float(xy[30+a,p]),float(q in homes),float(q in assets)])
        methods.append(ctx)
    return uid,roles,rs,np.asarray(local,dtype=np.float32),np.asarray(methods,dtype=np.float32),g


def _mw_router(g,z):
    standardized=(g-z['gate_mean'])/z['gate_scale']
    return 1./(1.+np.exp(-np.clip(standardized@z['gate_w']+z['gate_b'],-20.,20.)))


def _mw_logits(raw,local,context,role,base,style,model):
    center=raw[:,:,0]
    c=(center-model['cmean'])/model['cscale']
    diff=(raw-center[:,:,None,:])/model['dscale']
    features=(local-model['lmean'])/model['lscale']
    h=np.tanh(c@model['cw']+features[:,None,:]@model['lw']+model['hb'])
    direct=h@model['dirs']
    directional=np.stack([direct[:,j,actions] for j,(_,_,actions,_) in enumerate(_V5_ROTATIONS)],axis=1)
    d=np.tanh(diff@model['dw']+context[:,None,:,:]@model['kw']+model['db'])
    common=directional+d@model['dv']
    spec=(h@model['sty_dirs']).reshape(len(h),4,3,5)
    directional_style=np.stack([spec[:,j][:,:,actions] for j,(_,_,actions,_) in enumerate(_V5_ROTATIONS)],axis=1)
    sr=np.sum(directional_style*style[:,None,:,None],axis=2)
    correction=(common+sr).mean(1)+np.einsum('naf,nf->na',context,model['role'][role])
    return base*model['temperature'][role,None]+np.clip(correction,-4.,4.)


def _mw_propose(obs,x,previous,state,latent,base):
    uid,roles,r,loc,ctx,g=_mw_inputs(obs,x,previous,latent)
    if not uid:return None
    p=_mw_router(g,_MW_NET)
    style=np.broadcast_to([1.-p,0.,p],(len(uid),3))
    positions=[obs['players'][obs['player']][2][u][0] for u in uid]
    logits=base[0].copy()
    logits[positions]=_mw_logits(r,loc,ctx,roles,logits[positions],style,_MW_NET)
    return logits,base[1]


def _mw_joint(obs,root,logits,fixed,seed):
    # Sample coordinated workers with the normal collision checks, reserving exactly the incumbent's hires.
    workers,bases=logits;own=obs['players'][obs['player']]
    locked_bases=bases.copy()
    for uid,cell in own[1].items():
        locked_bases[cell]=-100.
        locked_bases[cell,BACT.index(root[uid]) if uid in root else 0]=0.
    return _sample(obs['players'],obs['player'],workers*2.,locked_bases,np.random.RandomState(seed^86173),obs['step'],fixed)


def build(code,model=None):
    marker='def _v5_root(obs, est, x, previous):'
    hidden='    sides = []\n    for side, player in enumerate((mine, rival)):'
    root='    _shadow_update(obs,x,opponent_x,state,mine,econ,styles,style,rival)'
    search='    if most>=3:\n        tree_action=_r6_first_tree(obs,configuration,est,act0,lw0)'
    for a in (marker,hidden,root,search):
        if code.count(a)!=1:raise ValueError('Unexpected worker-expert hook '+a)
    blob=base64.b64encode(Path(model or Path(__file__).with_name('master-worker-model.npz')).read_bytes()).decode()
    init='import base64 as _mw64,io as _mwio\n_MW_NET=dict(np.load(_mwio.BytesIO(_mw64.b64decode('+repr(blob)+')),allow_pickle=False))\n'
    f='\n\n'.join(inspect.getsource(a) for a in (_mw_inputs,_mw_router,_mw_logits,_mw_propose,_mw_joint))
    code=code.replace(marker,init+f+'\n\n'+marker)
    code=code.replace(hidden,"    _ST['v5_history']['mw_hidden']=latent[::2]\n"+hidden)
    code=code.replace(root,root+"""
    _mw_lat=state.pop('mw_hidden')
    _mw_us,_mw_foe=players[me],players[op]
    state['mw_workers']=None
    if step>=10 and (_mw_us[0]<125 or _mw_foe[0]<100 or any(u[2]=='COLLECTOR' and u[1]>=90 for u in _mw_foe[2].values())):
        _mw_extra=_mw_propose(obs,x,previous,state,_mw_lat,mine)
        state['mw_workers']=_mw_extra""")
    hook="""    _mw_pair=_ST['v5_history'].get('mw_workers')
    if _mw_pair is not None and most>=4 and len(cands)<most:
        _mw_action=_mw_joint(obs,act0,_mw_pair,fixed,seed0)
        _mw_key=tuple(sorted(_mw_action.items()))
        if _mw_key not in keys:
            keys.add(_mw_key);cands.append(_mw_action)
"""
    return code.replace(search,hook+search)
