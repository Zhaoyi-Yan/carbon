"""Build a simultaneous one-ply best response for loaded workers and contested trees."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np

from reply_features import _reply_features
from reply_weights import REPLY_COEFFICIENTS

ROOT=Path(__file__).resolve().parents[2]


def _reply_assets(board,side,config):
    remaining=max(0,LAST_TURN+1-board['step'])
    values=[]
    for pl in board['players']:
        cash,base,work,trees=pl
        amount=cash
        home=_v5_home(tuple(base.values()))
        for p,v,k in work.values():
            # Approximate the recoverable cargo and the cost of replacing an active worker.
            amount+=v*(home[p]<=remaining)*(.84/(1.+.035*home[p]))
            amount+=min(1.,remaining/18.)*(43. if k=='COLLECTOR' else 39.)
        for uid,(p,age) in trees.items():
            can_pay=min(remaining,max(0,config.get('treeLifespan',50)-age),12)
            info=board['_st'].trees_meta
            amount+=can_pay*min(13.,info.get(uid,(None,0.))[1])
        if not work and cash<min(config.get('recCollectorCost',30),config.get('recPlanterCost',30)) and remaining>1:
            amount-=1200.
        values.append(amount)
    return values[side]-values[1-side]


def _reply_choice(obs,config,est,act,logits,locked,t0):
    if _time.perf_counter()-t0>.84:
        return act
    me=obs['player']; op=1-me; players=obs['players']; step=obs['step']
    if not players[me][2] or not players[op][2]:
        return act
    attack={}
    vulnerable=[]
    for uid,(pos,load,kind) in players[me][2].items():
        if uid in locked:
            continue
        prey=kind=='PLANTER'
        dest=_ROLLOUT_LANDING[pos][WIDX.get(act.get(uid),0)]
        danger=0.
        for rival,(q,v,role) in players[op][2].items():
            if FEAT_DIST[pos,q]>2:continue
            can_lose=role=='COLLECTOR' and (prey or v<=load)
            can_win=kind=='COLLECTOR' and (role=='PLANTER' or load<v)
            if not (can_lose or can_win):continue
            # Include intercepts as well as evasions. Actual collisions are settled by the engine.
            stake=(42.+min(load,250.)*.8) if can_lose else (28.+min(v,220.)*.45)
            value=stake*(1. if FEAT_DIST[q,dest]<=1 else .43)
            danger=max(danger,value)
            attack[rival]=max(attack.get(rival,0.),value)
        if danger:
            vulnerable.append((danger,uid,pos))
    if not vulnerable:return act
    opponent,base=_ST['v5_history']['opponent']
    corrected=opponent.copy()
    for uid,(pos,_,kind) in players[op][2].items():
        features=_reply_features(players,op,uid,_ST.get('prev'),FEAT_DIST,_ROLLOUT_LANDING)
        if features is None:continue
        weights=_REPLY_COEF[int(kind=='PLANTER')]
        corrected[pos]=weights[0]*opponent[pos]+_TEMP*(features@weights[1:])
    # Enumerate the focal enemy's likely intentions, sampling its teammates conditionally
    # from the same prior and random stream. This also covers an unlikely but affordable capture.
    pivot=max(attack,key=attack.get)
    pivot_pos=players[op][2][pivot][0]
    p=_probs(corrected[pivot_pos])
    order=np.argsort(-p)
    modes=[i for i in order[:4] if p[i]>=.04] or [order[0]]
    root={'players':players,'step':step,'player':me,'trees':obs.get('trees',{}),'carbon':est}
    seed=1973*step+997*me+89
    replies=[]
    for mode in modes:
        rng=np.random.RandomState(seed)
        joint=_sample(players,op,corrected,base,rng,step,{pivot:WACT[mode] if mode else None})
        replies.append((p[mode],joint))
    mass=sum(w for w,_ in replies)
    replies=[(.78*w/mass,a) for w,a in replies]
    for k in (1,2):
        joint=_sample(players,op,corrected,base,np.random.RandomState(seed+119*k),step)
        replies.append((.11,joint))
    merged={}
    for w,a in replies:
        key=tuple(sorted(a.items()))
        merged[key]=merged.get(key,0.)+float(w)
    weights=np.asarray(list(merged.values()))
    actions=[dict(k) for k in merged]
    # Candidate moves are single-worker deviations, with friendly landings reserved.
    destinations={uid:_ROLLOUT_LANDING[p][WIDX.get(act.get(uid),0)]
                  for uid,(p,_,_) in players[me][2].items()}
    recruits={p for uid,p in players[me][1].items() if act.get(uid)}
    candidates=[(act,0.)]
    for _,uid,pos in sorted(vulnerable,reverse=True)[:2]:
        probabilities=_probs(logits[pos])
        original=WIDX.get(act.get(uid),0)
        reserved={q for worker,q in destinations.items() if worker!=uid}|recruits
        for mode,dest in enumerate(_ROLLOUT_LANDING[pos]):
            if mode==original or dest in reserved:continue
            proposal=dict(act)
            if mode:proposal[uid]=WACT[mode]
            else:proposal.pop(uid,None)
            prior_loss=max(0.,np.log(max(1e-9,probabilities[original])/max(1e-9,probabilities[mode])))
            candidates.append((proposal,7.+2.5*min(5.,prior_loss)))
    best,best_score=act,-float('inf')
    for proposal,penalty in candidates:
        if _time.perf_counter()-t0>.875:break
        vals=[]
        for enemy in actions:
            pair=[proposal,enemy] if me==0 else [enemy,proposal]
            vals.append(_reply_assets(_turn(root,config,pair),me,config))
        order=np.argsort(vals)
        lower=0.;left=.25
        for i in order:
            part=min(left,weights[i]);lower+=part*vals[i];left-=part
            if left<=1e-9:break
        score=.70*float(weights@vals)+.30*lower/.25-penalty
        if score>best_score:
            best,best_score=proposal,score
    return best


def build(source,coefficients):
    if len(coefficients)!=2 or any(len(w)!=27 for w in coefficients):
        raise ValueError('Expected coefficients for both kinds and every contact feature')
    text=source.decode()
    anchor='def _act(obs, configuration=None, t0=None):'
    call='        act = _v6_last_worker(obs, configuration, act, lw)'
    if text.count(anchor)!=1 or text.count(call)!=1:
        raise ValueError('Expected compatible yan-v6 search + last-worker guard')
    additions='\n\n'.join([f'_REPLY_COEF = np.asarray({coefficients!r},dtype=np.float32)',
                            inspect.getsource(_reply_features),inspect.getsource(_reply_assets),inspect.getsource(_reply_choice)])
    text=text.replace(anchor,additions+'\n\n'+anchor)
    text=text.replace(call,"""        if _SEARCH_OK and _RX and t0 is not None:
            act = _reply_choice(obs,configuration,est,act,lw,{c[1] for c in cand if c[0]==2.0},t0)
"""+call)
    compile(text,'<reply-search>','exec')
    return text


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path,nargs='?',help='Optional refit from reply_train.py');p.add_argument('output',type=Path)
    p.add_argument('--parent',type=Path,default=ROOT/'yan-v6.py')
    a=p.parse_args()
    coeff=[r['weights'] for r in json.loads(a.model.read_text())['roles']] if a.model else REPLY_COEFFICIENTS
    source=a.parent.read_bytes()
    text=build(source,coeff)
    with a.output.open('x') as f:f.write(text)
    print(hashlib.sha256(source).hexdigest(),hashlib.sha256(text.encode()).hexdigest())
