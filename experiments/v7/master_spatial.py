"""Optional Master spatial expert. The frozen 21/22 policies keep their original code and parameters."""
import argparse
import base64
import inspect
import io
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def _mx_trace(players,side,past,cells,actions):
    # past is newest first; only positions of workers still public in both adjacent observations are used.
    out=np.zeros((30,225),np.float32)
    for team in (side,1-side):
        units=players[team][2]
        for uid,(p,_,_) in units.items():
            here=p
            for lag,old in enumerate(past[:3]):
                then=old[team][2].get(uid)
                if then is None:break
                possible=_ROLLOUT_LANDING[then[0]]
                if here in possible:
                    a=possible.index(here);out[(15*(team!=side)+5*lag+int(actions[a])),cells[p]]=1.
                here=then[0]
    return out.reshape(30,15,15)

def _mx_heads(obs,x,past,style='E',rotations=(0,1,2,3),latent=None):
    import torch
    import torch.nn.functional as F
    if style not in ('E','D','B2'):raise ValueError('Expected a Master specialist ID')
    global _MX_WEIGHTS
    if not isinstance(_MX_WEIGHTS['stem.weight'],torch.Tensor):
        _MX_WEIGHTS={k:torch.from_numpy(v.astype(np.float32)) for k,v in _MX_WEIGHTS.items()}
    data=_MX_WEIGHTS;sid={'D':0,'B2':1,'E':2}[style]
    frames=[];memory=[]
    for j in rotations:
        cells,inv,acts,channels=_V5_ROTATIONS[j]
        frames.append(x.reshape(NPL,225)[channels][:,inv].reshape(NPL,15,15))
        memory.append(_mx_trace(obs['players'],obs['player'],past,cells,acts))
    X=np.stack(frames).astype(np.float32)
    latent=(_net()(X,np.full(len(X),22,dtype=np.int64),hidden=True) if latent is None
            else latent[np.asarray(rotations)])
    h=latent.transpose(0,2,1).reshape(len(X),96,15,15)
    with torch.inference_mode():
        d=lambda name:data[name]
        conv=lambda z,n:F.conv2d(z,d(n+'.weight'),d(n+'.bias'))
        circ=lambda z,n,dr:F.conv2d(F.pad(z,(dr,dr,dr,dr),mode='circular'),d(n+'.weight'),d(n+'.bias'),dilation=dr)
        inp=torch.from_numpy(np.concatenate((h,X,np.stack(memory)),axis=1))
        z=torch.relu(conv((inp-d('mean'))*d('inv'),'stem')+d('style')[sid].reshape(1,-1,1,1))
        for j,dr in enumerate((1,2)):
            mid=torch.relu(circ(z,'spans.'+str(j),dr))
            z=torch.relu(z+conv(mid,'mix.'+str(j))+conv(z.mean((2,3),keepdim=True),'world.'+str(j)))
        q=conv(z,'heads').numpy().reshape(len(X),10,225).transpose(0,2,1)
    flat=latent.reshape(-1,96);policy=_POLICY.p
    raww=(flat@policy['hw.w'].reshape(5,96).T+policy['hw.b']).reshape(len(X),225,5)
    rawb=(flat@policy['hb.w'].reshape(3,96).T+policy['hb.b']).reshape(len(X),225,3)
    w=np.zeros((225,5),np.float32);b=np.zeros((225,3),np.float32)
    temp=data['temperature'].numpy()
    for k,j in enumerate(rotations):
        cells,inv,actions,chans=_V5_ROTATIONS[j]
        worker=raww[k]*temp[0]+q[k,:,:5]
        base=rawb[k]*temp[1]+q[k,:,5:8]
        w+=worker[cells][:,actions]/len(rotations);b+=base[cells]/len(rotations)
    return w,b

def _mx_roles(obs):
    own,enemy=(obs['players'][s] for s in (obs['player'],1-obs['player']))
    depots=tuple(own[1].values())
    gates=tuple(enemy[1].values())
    if not gates or not depots:return {}
    near_gate=[p for p,c,k in enemy[2].values()
               if k=='COLLECTOR' and c>=65 and min(dist(p,b) for b in gates)<=4]
    roles={}
    for uid,(p,c,k) in own[2].items():
        if k!='COLLECTOR':continue
        if c>=65 and min(dist(p,b) for b in depots)<=4:
            roles[uid]='loaded_home'
        elif c<=15 and min(dist(p,b) for b in gates)<=4:
            roles[uid]='light_enemy_depot'
        elif c<=15 and any(dist(p,q)<=3 for q in near_gate):
            roles[uid]='light_enemy_carrier'
    return roles

def _mx_tactical(obs,root,fixed,context,style='E',roles=None,ambiguous=False,existing=()):
    if context is None:return None
    units=obs['players'][obs['player']][2]
    selected=[u for u in (_mx_roles(obs) if roles is None else roles) if u in units and u not in fixed]
    if not selected:return None
    x,past,latent=context
    logits,_=_mx_heads(obs,x,past,style,latent=latent)
    occupied=[0]*225
    for bid,p in obs['players'][obs['player']][1].items():
        if root.get(bid) in ('RECPLANTER','RECCOLLECTOR'):occupied[p]+=1
    for uid,(p,_,_) in units.items():
        occupied[_ROLLOUT_LANDING[p][WIDX.get(root.get(uid),0)]]+=1
    def gain(uid):
        p=units[uid][0];a=WIDX.get(root.get(uid),0)
        return float(np.max(logits[p])-logits[p,a])
    out=dict(root)
    for uid in sorted(selected,key=gain,reverse=True):
        p=units[uid][0];old=WIDX.get(root.get(uid),0);land=_ROLLOUT_LANDING[p]
        occupied[land[old]]-=1
        for a in np.argsort(-logits[p],kind='stable'):
            if a==old or occupied[land[a]]==0:
                occupied[land[a]]+=1
                if a!=old:
                    if a:out[uid]=WACT[a]
                    else:out.pop(uid,None)
                break
    if out!=root:
        if ambiguous and tuple(sorted(out.items())) in existing:return None
        return out
    if not ambiguous:return None
    # When the two policies already agree, offer one calibrated close second.
    best=None
    for uid in selected:
        p=units[uid][0];old=WIDX.get(root.get(uid),0);land=_ROLLOUT_LANDING[p]
        rank=np.argsort(-logits[p],kind='stable')
        if rank[0]!=old:continue
        a=int(rank[1])
        if occupied[land[a]]:continue
        z=np.exp(logits[p]-logits[p,old]);ratio=float(z[a])
        if ratio<.45 or float(z[a]/z.sum())<.18:continue
        candidate=dict(root)
        if a:candidate[uid]=WACT[a]
        else:candidate.pop(uid,None)
        if tuple(sorted(candidate.items())) in existing:continue
        if best is None or ratio>best[0]:best=(ratio,candidate)
    if best is None:return None
    return best[1]

def build(code,weights=None,mode='library',style='E'):
    model=Path(weights or HERE/'master-spatial.npz')
    a='def _v5_root(obs, est, x, previous):'
    if code.count(a)!=1 or '_MX_WEIGHTS=' in code:raise ValueError('Expected one unmodified Harbormaster router')
    if mode not in ('library','proposal','tactical','tactical-secondary','legacy-proposal') or style not in ('E','D','B2'):raise ValueError('Unknown integration mode or style')
    with np.load(model) as check:
        if 'query.weight' in check:raise ValueError('The ten-turn objective variant is research-only')
    payload=base64.b64encode(model.read_bytes()).decode('ascii')
    init="with np.load(io.BytesIO(base64.b64decode("+repr(payload)+")),allow_pickle=False) as _mx_data:\n    _MX_WEIGHTS=dict(_mx_data)\n"
    code=code.replace(a,init+inspect.getsource(_mx_trace)+'\n'+inspect.getsource(_mx_heads)+'\n'+inspect.getsource(_mx_roles)+'\n'+inspect.getsource(_mx_tactical)+'\n'+a)
    if mode in ('proposal','tactical','tactical-secondary'):
        cache="    sides = []\n    for side, player in enumerate((mine, rival)):"
        pack="    _v5_rollout_set(me,style,rival)"
        search="    if len(cands) < 2:\n        _S_STATS['single']"
        for marker in (cache,pack,search):
            if code.count(marker)!=1:raise ValueError('Incompatible game policy integration point')
        code=code.replace(cache,"    _mx_state=_ST['v5_history']\n    _mx_state['mx_latent']=latent[::2] if _mx_state['root_ids'][0]==22 else None\n"+cache)
        code=code.replace(pack,"    _mx_past=state.get('mx_past',[]) if state.get('mx_tick')==step-1 else ([previous] if previous else [])\n    state['mx_context']=(x,_mx_past,state.pop('mx_latent',None))\n    state['mx_past']=[players]+_mx_past[:2]\n    state['mx_tick']=step\n"+pack)
        extra="    _mx_extra=None\n    if most>=3 and _time.perf_counter()-t0<.55:\n        _mx_extra=_mx_tactical(obs,act0,fixed or {},_ST['v5_history'].get('mx_context'),"+repr(style)+",ambiguous="+repr(mode=='tactical-secondary')+",existing=keys)\n        if _mx_extra is not None:\n            _mx_key=tuple(sorted(_mx_extra.items()))\n            if _mx_key not in keys:\n                keys.add(_mx_key)\n                cands.append(_mx_extra)\n"
        code=code.replace(search,extra+search)
    if mode=='legacy-proposal':
        mark="    _mw_lat=state.pop('mw_hidden')"
        use="        _mw_extra=_mw_propose(obs,x,previous,state,_mw_lat,mine)"
        if code.count(mark)!=1 or code.count(use)!=1:raise ValueError('Expected existing Master worker proposal')
        code=code.replace(mark,mark+"\n    _mx_past=state.get('mx_past',[]) if state.get('mx_tick')==step-1 else ([previous] if previous else [])\n    state['mx_past']=[players]+_mx_past[:2]\n    state['mx_tick']=step")
        code=code.replace(use,"        _mx_workers,_=_mx_heads(obs,x,_mx_past,"+repr(style)+",latent=_mw_lat if active_expert==22 else None)\n        _mw_extra=(_mx_workers,mine[1])")
    return code
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('out',type=Path);p.add_argument('--weights',type=Path);p.add_argument('--mode',choices=['library','proposal','tactical','tactical-secondary','legacy-proposal'],default='library');p.add_argument('--style',choices=['E','D','B2'],default='E')
    a=p.parse_args();src=build(a.parent.read_text(),a.weights,a.mode,a.style);compile(src,str(a.out),'exec')
    with a.out.open('x') as f:f.write(src)
