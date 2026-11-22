"""A prospective, order-preserving root sampler estimate for opponents using the published MT19937 policy."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np

MODEL = Path(__file__).with_name('quantile-model.json')

def _qt_soft(z):
    e=np.exp(z-z.max(axis=1,keepdims=True))
    return e/e.sum(axis=1,keepdims=True)

def _qt_logratio(sample, draws, h, mixture):
    # Reflect a uniform jitter at 0 and 1. For uniform draws the expected conditional
    # probabilities exactly equal the input row, including tiny edge classes.
    boundaries=np.cumsum(sample,axis=1)
    lo,hi=draws[:,None]-h,draws[:,None]+h
    clip=lambda x: np.minimum(np.maximum(x,0.),hi-lo)
    # Direct [0,b] is clip(b-lo)-clip(-lo); negative [-b,0] is clip(-lo)-clip(-b-lo).
    cdf=(clip(boundaries-lo)-clip(-boundaries-lo)+clip(hi-2.+boundaries))/(2.*h)
    conditional=np.diff(cdf,axis=1,prepend=0.).clip(0.,1.)
    return np.log((1.-mixture)+mixture*conditional/np.maximum(sample,1.e-20))

def _qt_gate(memory,step,units):
    # The current command is never available to recognition. Deaths are ambiguous.
    if memory.get('last')!=step-1:
        memory['evidence']=0.;memory['count']=0.;memory['pending']={}
    seen=[]
    for uid,(pos,logratio) in memory.get('pending',{}).items():
        w=units.get(uid)
        if w is not None and w[0] in _ROLLOUT_LANDING[pos]:
            action=_ROLLOUT_LANDING[pos].index(w[0])
            seen.append(float(np.clip(logratio[action],-1.4,1.4)))
    n=min(4,len(seen));count=.965*memory.get('count',0.)+n
    odds=.965*memory.get('evidence',0.) + (n*np.mean(seen) if seen else 0.)
    memory.update(last=step,count=count,evidence=float(np.clip(odds,-6.,6.)),pending={})
    return float(.85*np.tanh(max(0.,odds-.35)/1.7)) if count>=10 else 0.

def _qt_open(memory,players,side,step):
    own=players[side]
    if step==1:
        other=list(players[1-side][2].values())
        sites=list(own[1].values())
        memory['delayed']=len(sites)==2 and not own[2]
        expected=None
        if len(sites)==2 and len(other)==2:
            if all(w[2]=='COLLECTOR' for w in other):expected=max(sites)
            elif side==0:
                p=next((w[0] for w in other if w[2]=='PLANTER'),None)
                if p is not None:expected=sites[int(sites[1]//15==p//15)]
        memory['expected']=expected
    elif step==2 and memory.get('delayed'):
        roles={w[2]:w[0] for w in own[2].values()}
        memory['matched']=(len(own[2])==2 and set(roles)=={'PLANTER','COLLECTOR'} and
                           (memory['expected'] is None or memory['expected']==roles['PLANTER']))

def _qt_weight(opening,count,behavior,seed_evidence):
    if not opening or count<14:return 0.
    # Positive likelihood from the existing classifier must precede the current turn.
    family=.78*np.tanh(max(0.,behavior-.55)/2.)
    # A consistent historical draw supports attribution; repeated contradictions cancel it.
    confidence=float(np.clip(.65+.25*seed_evidence,0.,1.))
    return float(family*confidence)

def _qt_update(state,players,side,step,sample_logits):
    units=players[side][2];mem=state.setdefault('quantile_reply',{})
    if mem.get('tick')==step:return
    if step==0:mem.clear();mem.update(rng=np.random.RandomState(37),tick=-1)
    if mem.get('tick')!=step-1 or 'rng' not in mem:
        mem['invalid']=True
    mem['tick']=step
    if mem.get('invalid'):return
    nbase=len(players[side][1]);draws=mem['rng'].random_sample(nbase+len(units))[nbase:]
    _qt_open(mem,players,side,step)
    if not mem.get('matched',False):return
    _qt_gate(mem,step,units)
    classifier=state.get('ds_class',{})
    alpha=_qt_weight(mem.get('matched',False),classifier.get('count',0.),classifier.get('evidence',0.),mem.get('evidence',0.))
    if not units:return
    pos=[w[0] for w in units.values()]
    base,bases=state['opponent']
    sample=_qt_soft(sample_logits[pos]/_QT_TEMP)
    ratio=_qt_logratio(sample,draws,_QT_WIDTH,_QT_MIX)
    actual=_qt_soft(base[pos]/_TEMP)
    tilted=_qt_soft(np.log(np.maximum(actual,1.e-25))+ratio)
    loggain=np.log(np.maximum(tilted,1.e-25))-np.log(np.maximum(actual,1.e-25))
    mem['pending']={uid:(pos[j],loggain[j]) for j,uid in enumerate(units)}
    if alpha:
        out=base.copy();out[pos]+=(_TEMP*alpha)*loggain
        state['opponent']=out,bases

def build(code):
    fit=json.loads(MODEL.read_text())
    if fit['sample']!='bare':raise ValueError('This lightweight adapter expects pre-distillation opponent logits.')
    hook="    _v5_rollout_set(me,style,rival)"
    gate="    _ds_class_reply(state,players,op,step,opponent_x,previous)"
    if code.count(hook)!=1 or '_QT_WIDTH=' in code:raise ValueError('Expected one frozen v5 root.')
    anchor='def _v5_root(obs, est, x, previous):'
    if code.count(anchor)!=1:raise ValueError(anchor)
    if code.count(gate)!=1:raise ValueError('Expected existing fast opponent-class likelihood filter')
    capture=gate
    code=code.replace(capture,"    quantile_source=state['opponent'][0]\n"+capture)
    params=f"_QT_TEMP={fit['temperature']!r}\n_QT_WIDTH={fit['width']!r}\n_QT_MIX={fit['mixture']!r}\n"
    functions='\n\n'.join(inspect.getsource(f) for f in (_qt_soft,_qt_logratio,_qt_gate,_qt_open,_qt_weight,_qt_update))
    return code.replace(anchor,params+functions+'\n\n'+anchor).replace(hook,'    _qt_update(state,players,op,step,quantile_source)\n'+hook)

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('parent',type=Path);cli.add_argument('output',type=Path)
    a=cli.parse_args();code=build(a.parent.read_text());compile(code,str(a.output),'exec')
    with a.output.open('x') as f:f.write(code)
