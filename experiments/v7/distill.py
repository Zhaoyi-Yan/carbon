"""Prospectively gated distillation of a search policy's observed worker commands."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).with_name("distill-model.json")


def _ds_experts(view):
    # Use the frozen specialist nets on the current rival board. No new trunk.
    out = []
    rotations = (_V5_ROTATIONS[0], _V5_ROTATIONS[2])
    for _, inverse, _, channels in rotations:
        native = view.reshape(NPL,N)[channels][:,inverse].reshape(NPL,SIZE,SIZE)
        out.extend((native,native))
    work, _, _ = _net()(np.stack(out).astype(np.float32), np.array([21,22,21,22]))
    result = np.zeros((2,N,5),np.float32)
    for j,(cells,_,actions,_) in enumerate(rotations):
        result += .5 * work[2*j:2*j+2][:,cells][:,:,actions]
    return result


def _ds_features(view,players,side,previous,prior,experts):
    units = players[side][2]
    ids = list(units)
    if not ids:
        return ids,None,None,None
    positions = np.array([units[u][0] for u in ids])
    load = np.array([min(2.5,units[u][1]/90.) for u in ids])[:,None]
    roles = np.array([int(units[u][2]=='PLANTER') for u in ids])
    planes = view.reshape(NPL,N)
    land = _DS_LAND[positions]
    cells = lambda i: planes[i,land]
    z = prior[positions].astype(np.float32)
    center = lambda q:np.clip(q-q.mean(axis=1,keepdims=True),-10.,10.)
    z0=center(z)
    a,b=(center(experts[k,positions]) for k in (0,1))
    momentum=np.zeros_like(z)
    old=previous[side][2] if previous else {}
    for i,u in enumerate(ids):
        if u in old and positions[i] in _ROLLOUT_LANDING[old[u][0]]:
            momentum[i,_ROLLOUT_LANDING[old[u][0]].index(positions[i])]=1.
    wait=np.zeros_like(z);wait[:,0]=1.
    home=(cells(39)[:,:1]-cells(39))*14.
    danger=(cells(42)>0)&(cells(43)<planes[3,positions,None])
    inside=np.broadcast_to(planes[18,positions,None],z.shape)
    stock=cells(0)
    nearest=planes[39,positions,None]*14.
    features=np.stack((z0,a,b,wait,momentum,stock,stock*wait,stock*load,
                       home,home*load,home*(nearest<4),cells(12)*load,cells(13),
                       cells(10),cells(8),cells(2)*(1-wait),cells(4)*(1-wait),
                       cells(5),cells(7),danger,danger*load,cells(44),cells(46),
                       inside*wait,home*(inside>.83)*load,(a+b)*load*.5,
                       z0*(np.abs(z0).max(1,keepdims=True)>3)),axis=2).astype(np.float32)
    return ids,roles,z,features


def _ds_logprob(rows):
    z=rows-rows.max(axis=-1,keepdims=True)
    return z-np.log(np.exp(z).sum(axis=-1,keepdims=True))


def _ds_gate(mem,step,units):
    if mem.get('step')!=step-1:
        mem.clear()
    observed=[]
    for uid,(pos,advantage) in mem.get('pending',{}).items():
        if uid in units and units[uid][0] in _ROLLOUT_LANDING[pos]:
            move=_ROLLOUT_LANDING[pos].index(units[uid][0])
            observed.append(float(np.clip(advantage[move],-1.,1.)))
    count=.93*mem.get('count',0.)
    evidence=.93*mem.get('evidence',0.)
    if observed:
        weight=min(4,len(observed))
        count+=weight
        evidence+=weight*float(np.mean(observed))
    mem.update(step=step,count=count,evidence=evidence,pending={})
    gain=evidence/max(1.,count)
    return float(np.clip((gain-.008)*4.,0.,.78)) if count>=12 else 0.


def _ds_update(state,players,side,step,view,previous):
    units=players[side][2]
    mem=state.setdefault('distilled_reply',{})
    alpha=_ds_gate(mem,step,units)
    if not units:return
    baseline,bases=state['opponent']
    ids,roles,z,feats=_ds_features(view,players,side,previous,baseline,_ds_experts(view))
    model=np.einsum('nak,nk->na',feats,_DS_COEF[roles])+_DS_SCALE[roles,None]*z
    ref=_DS_SCALE[roles,None]*z
    evidence=np.clip(_ds_logprob(model)-_ds_logprob(ref),-3.,3.)
    mem['pending']={u:(units[u][0],evidence[j]) for j,u in enumerate(ids)}
    if alpha>0:
        # _draw_rows divides these logits by the frozen simulator's sampling temperature.
        out=baseline.copy()
        pos=[units[u][0] for u in ids]
        out[pos]=(1.-alpha)*baseline[pos]+alpha*_TEMP*model
        state['opponent']=out,bases


def build(code):
    weights=json.loads(WEIGHTS.read_text())
    anchor='def _v5_root(obs, est, x, previous):'
    before='    _v5_rollout_set(me,style,rival)'
    if code.count(anchor)!=1 or code.count(before)!=1 or '_DS_COEF=' in code:
        raise ValueError('Expected compatible v6 opponent prediction at the root')
    helpers='\n\n'.join(inspect.getsource(f) for f in (_ds_experts,_ds_features,_ds_logprob,_ds_gate,_ds_update))
    const='_DS_LAND=np.array([[move(p,a) for a in range(5)] for p in range(225)])\n'
    const+='_DS_COEF=np.array('+repr(weights['coefficients'])+',np.float32)\n'
    const+='_DS_SCALE=np.array('+repr(weights['scale'])+',np.float32)\n'
    return code.replace(anchor,const+helpers+'\n\n'+anchor).replace(before,'    _ds_update(state,players,op,step,opponent_x,previous)\n'+before)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();code=build(a.parent.read_text());compile(code,str(a.output),'exec')
    with a.output.open('x') as f:f.write(code)
