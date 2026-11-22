"""Compare reachable deposits at the scoring deadline under the observed enemy move distribution."""
import argparse
import inspect
from pathlib import Path


def _r6_return_routes(obs, enemy_logits):
    remaining=LAST_TURN+1-obs['step']
    if remaining>23 or remaining<1:return {}
    me=obs['player'];own=obs['players'][me];rival=obs['players'][1-me]
    bases=list(own[1].values());home=np.asarray(_v5_home(tuple(bases)))
    carriers=[(key,pos,cargo) for key,(pos,cargo,role) in own[2].items()
              if role=='COLLECTOR' and cargo>=45 and 0<home[pos]<=remaining
              and remaining-home[pos]<=min(10.,3.+cargo/42.)]
    if not carriers:return {}
    land=np.asarray(_ROLLOUT_LANDING)
    enemies=[(p,c) for p,c,k in rival[2].values() if k=='COLLECTOR']
    danger=[]
    for pos,cargo in enemies:
        terms=np.zeros((remaining,225),dtype=np.float32)
        terms[0,land[pos]]=_probs(enemy_logits[pos],.72)
        for t in range(1,remaining):
            before=terms[t-1]
            # Predict only short-range interception. Farther out the current board
            # supplies progressively less information about an enemy patrol.
            terms[t]=.36*before+.16*before[land[:,1:]].sum(1)
        danger.append((cargo,terms))
    advice={}
    for key,pos,cargo in carriers:
        risk=sum((field for load,field in danger if load<=cargo),np.zeros((remaining,225),dtype=np.float32))
        survival=np.clip(1.-.9*risk,.09,1.)
        survival[:,bases]=1.
        # The value of each cell is its best reachable probability of banking.
        value=np.zeros(225,dtype=np.float32);value[bases]=1.
        for t in reversed(range(remaining)):
            next_value=survival[t,land]*value[land]*.98
            next_value[:,0]*=.985
            if t==0:
                advice[key]=next_value[pos]
                break
            value=next_value.max(1);value[bases]=1.
    return advice


def build(code):
    def change(old,new):
        nonlocal code
        if code.count(old)!=1:raise ValueError(old)
        code=code.replace(old,new)
    change('def _act(obs, configuration=None, t0=None):',inspect.getsource(_r6_return_routes)+'\n\ndef _act(obs, configuration=None, t0=None):')
    anchor="    for wid, w in mine:"
    change(anchor,"""    enemy = st.get('v5_history', {}).get('opponent')
    settlement = _r6_return_routes(obs,enemy[0]) if enemy is not None else {}
"""+anchor)
    anchor='        if k_ is not None:\n            cand.append'
    change(anchor,"""        if wid in settlement:
            route=settlement[wid]
            target=int(np.argmax(route+.008*p))
            previous=int(k_) if k_ is not None else int(a)
            if route[target]>.15 and route[target]-route[previous]>.09:
                k_=target
"""+anchor)
    anchor='        c=sum(p[2][uid][1]*min(1.,max(-.45,row@_R6_CARGO)) for uid,row in car.items())'
    change(anchor,"""        home=_v5_home(tuple(p[1].values()))
        remaining=LAST_TURN+1-t
        c=sum(p[2][uid][1]*min(1.,max(-.45,row@_R6_CARGO)) for uid,row in car.items()
              if home[p[2][uid][0]]<=remaining)""")
    anchor='                    value += 0.26 * (_r6_asset_score(future, me) - value)'
    change(anchor,"""                    liquidation=_r6_asset_score(future, me)
                    near_close=max(0.,1.-horizon/22.)
                    value+=(.26+.48*near_close)*(liquidation-value)""")
    compile(code,'<settlement>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(build(a.source.read_text()))
