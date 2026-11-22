"""Use a trained rival-class likelihood ratio before adopting distilled search replies."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
import distill

FIT=Path(__file__).with_name('distill-id-model.json')

def _ds_identify(mem,step,units):
    if mem.get('step')!=step-1:
        mem.clear()
    observed=[]
    for u,(pos,ratios) in mem.get('pending',{}).items():
        if u in units and units[u][0] in _ROLLOUT_LANDING[pos]:
            a=_ROLLOUT_LANDING[pos].index(units[u][0])
            observed.append(float(np.clip(ratios[a],-1.4,1.4)))
    count=.965*mem.get('count',0.);total=.965*mem.get('evidence',0.)
    if observed:
        n=min(5,len(observed));count+=n;total+=n*float(np.mean(observed))
    total=float(np.clip(total,-6.,6.))
    mem.update(step=step,count=count,evidence=total,pending={})
    if count<14 or total<.55:return 0.
    return float(.76*np.tanh((total-.55)/2.))


def _ds_class_reply(state,players,side,step,view,previous):
    units=players[side][2];memory=state.setdefault('ds_class',{})
    alpha=_ds_identify(memory,step,units);memory['weight']=alpha
    if not units:return
    base,bases=state['opponent']
    ids,roles,z,x=_ds_features(view,players,side,previous,base,_ds_experts(view))
    pos_z=_DS_SCALE[roles,None]*z+np.einsum('nak,nk->na',x,_DS_COEF[roles])
    neg_z=_DS_NEG_SCALE[roles,None]*z+np.einsum('nak,nk->na',x,_DS_NEG_COEF[roles])
    odds=_ds_logprob(pos_z)-_ds_logprob(neg_z)
    memory['pending']={u:(units[u][0],odds[i]) for i,u in enumerate(ids)}
    if alpha>0:
        # Halfway (geometric scale) between a calibrated search imitator and the
        # sharper one-reply sampler; the latter carries the same noise as the incumbent.
        calibration=np.sqrt(_TEMP/_DS_SCALE[roles,None])
        out=base.copy();cells=[units[u][0] for u in ids]
        out[cells]=(1.-alpha)*z+alpha*calibration*pos_z
        state['opponent']=out,bases


def build(code):
    base=distill.build(code)
    anchor='def _v5_root(obs, est, x, previous):'
    old='    _ds_update(state,players,op,step,opponent_x,previous)'
    if base.count(old)!=1:raise ValueError('Expected original positive model hook')
    m=json.loads(FIT.read_text())
    const='_DS_NEG_SCALE=np.asarray('+repr(m['scale'])+',np.float32)\n'
    const+='_DS_NEG_COEF=np.asarray('+repr(m['coefficients'])+',np.float32)\n'
    helpers='\n\n'.join(inspect.getsource(f) for f in (_ds_identify,_ds_class_reply))
    return base.replace(anchor,const+helpers+'\n\n'+anchor).replace(old,'    _ds_class_reply(state,players,op,step,opponent_x,previous)')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();code=build(a.parent.read_text());compile(code,str(a.output),'exec')
    with a.output.open('x') as f:f.write(code)
