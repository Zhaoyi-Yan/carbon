"""Propose a coordinated intercept before a loaded rival crosses its own depot tile."""
import argparse
import inspect
from pathlib import Path


def _r6_intercept(obs, configuration, carbon, base, locked, enemy_logits):
    me=obs['player'];pl=obs['players'];own,rival=pl[me],pl[1-me]
    left=LAST_TURN+1-obs['step']
    hunters=[(key,p,v) for key,(p,v,k) in own[2].items() if k=='COLLECTOR' and v<14. and key not in locked]
    loads=[(p,v) for p,v,k in rival[2].values() if k=='COLLECTOR' and v>=55.]
    if left<2 or len(own[2])<4 or not hunters or not loads:return None
    step=min(5,left)
    foe_home=tuple(rival[1].values());own_home=tuple(own[1].values())
    dist=np.asarray(_v5_home(foe_home));returns=np.asarray(_v5_home(own_home))
    land=_V5_REACH
    cheap=[(p,v) for p,v,k in rival[2].values() if k=='COLLECTOR']
    occupancy={key:land[p,WIDX.get(base.get(key),0)] for key,(p,_,_) in own[2].items()}
    hires={p for key,p in own[1].items() if key in base}
    plans=[]
    for start,value in sorted(loads,key=lambda unit:-unit[1])[:4]:
        if dist[start]<2 or dist[start]>left+1:continue
        probs=np.zeros(N,dtype='f')
        probs[land[start]]=_probs(enemy_logits[start],.8)
        bias=min(2.3,.70+value/180.+(1. if left<26 else 0.))
        toward=np.exp(bias*(dist[:,None]-dist[land]));toward[:,0]*=.80
        toward/=toward.sum(axis=1,keepdims=True)
        field=np.zeros(N,dtype='f')
        for t in range(1,step+1):
            probs[list(foe_home)]=0.
            delay=.83**(t-1)
            # Hunter may take t-1 further ordinary actions after the proposed move.
            reachable=probs
            for _ in range(t-1):reachable=reachable[land].max(axis=1)
            field=np.maximum(field,delay*reachable)
            probs=np.bincount(land.ravel(),weights=(probs[:,None]*toward).ravel(),minlength=N)
        field*=value*(.76+.30/(1.+returns/6.))
        for key,pos,load in hunters:
            if _V5_DIST[pos][start]>2*step:continue
            blocked={q for other,q in occupancy.items() if other!=key}|hires|set(foe_home)
            scores=field[land[pos]].copy()
            for a,dest in enumerate(land[pos]):
                if dest in blocked or any(c<=load and _V5_DIST[p][dest]<=1 for p,c in cheap):
                    scores[a]=-1000.
            old=WIDX.get(base.get(key),0);new=int(scores.argmax())
            if new==old or scores[new]<13.:continue
            before=max(0.,scores[old]);cost=6.+(.21*max(0.,carbon[pos]) if old==0 else 0.)
            gain=scores[new]-before-cost
            if gain>4.:plans.append((gain,key,new,start,land[pos,new]))
    if not plans:return None
    revised=base.copy();assigned=set();targets=set();landings=set()
    for _,key,a,target,q in sorted(plans,reverse=True):
        if key in assigned or target in targets or q in landings:continue
        if a:revised[key]=WACT[a]
        else:revised.pop(key,None)
        assigned.add(key);targets.add(target);landings.add(q)
        if len(assigned)==2:break
    return revised


def build(code):
    start='def _search(obs,';spot='    for T, cap in _S_GEN:'
    if code.count(start)!=1 or code.count(spot)!=1:raise ValueError('search signature changed')
    code=code.replace(start,inspect.getsource(_r6_intercept)+'\n\n'+start)
    extra="""    if most>=4 and len(cands)<most:
        hunt=_r6_intercept(obs,configuration,est,act0,fixed or {},_ST['v5_history']['opponent'][0])
        if hunt is not None:
            key=tuple(sorted(hunt.items()))
            if key not in keys:
                cands.append(hunt);keys.add(key)
"""
    code=code.replace(spot,extra+spot)
    compile(code,'<intercept>','exec');return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(build(a.source.read_text()))
