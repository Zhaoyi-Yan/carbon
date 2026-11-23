"""Learn a bounded v7 finalist correction from paired closed-loop cash outcomes."""
import argparse
import itertools
import json
from pathlib import Path
import inspect
import hashlib
from instrument import V7_SHA256, build as trace_build

SCORE_HOOK = '    best = max(alive, key=lambda c: mean[c])'
RANGE = (180,285)


def _v8_rank_features(obs, me, action):
    own, enemy = obs['players'][me], obs['players'][1-me]
    bases = tuple(own[1].values())
    home = _fc_home(tuple(sorted(bases)))
    foe = [(p,c) for p,c,k in enemy[2].values() if k=='COLLECTOR']
    trees = {p for p,age in enemy[3].values() if age<50}
    deposit = progress = vulnerable = harvest = seizure = 0.
    for uid,(p,cargo,kind) in own[2].items():
        a = WIDX.get(action.get(uid),0)
        q = move(p,a)
        if kind=='PLANTER':
            seizure += not a and p in trees
            continue
        if q in bases:
            deposit += cargo
        else:
            progress += cargo*(home[p]-home[q])
            vulnerable += cargo*any(v<=cargo and _V5_DIST[q][z]<=1 for z,v in foe)
            if not a and p not in trees:
                harvest += min(30., .3*max(0.,obs['carbon'][p]))
    return (deposit/200., progress/250., vulnerable/200., harvest/30., seizure)


def _v8_rank_correction(obs, side, cands, alive, scores):
    if not 180<=obs['step']<285 or len(alive)<2:
        return
    features = {i:_v8_rank_features(obs,side,cands[i]) for i in alive}
    # Center on the highest v7 score, so changing a candidate can only move its
    # relative score by the cap. A wide search advantage stays intact.
    ref = features[max(alive,key=scores.get)]
    for i in alive:
        change=sum(w*(x-y) for w,x,y in zip(_V8_RANK_WEIGHTS,features[i],ref))
        scores[i]+=max(-.055,min(.055,change))


def inject(source, text):
    if source.count(SCORE_HOOK)!=1:
        raise ValueError('Missing v7 finalist selection')
    return source.replace(SCORE_HOOK,text+SCORE_HOOK)


def trace_source(source):
    source=trace_build(source)
    hook='def _act(obs, configuration=None, t0=None):'
    source=source.replace(hook,inspect.getsource(_v8_rank_features)+'\n\n'+hook)
    return inject(source,"    if _AUDIT is not None:\n        _AUDIT['rank_features']={i:_v8_rank_features(obs,me,cands[i]) for i in alive}\n")


def build(source, weights):
    if hashlib.sha256(source.encode()).hexdigest()!=V7_SHA256 or len(weights)!=5:
        raise ValueError('Expected frozen v7 and five fitted weights')
    hook='def _act(obs, configuration=None, t0=None):'
    functions='\n\n'.join(inspect.getsource(f) for f in (_v8_rank_features,_v8_rank_correction))
    source=source.replace(hook,f'_V8_RANK_WEIGHTS={tuple(weights)!r}\n'+functions+'\n\n'+hook)
    return inject(source,'    _v8_rank_correction(obs,me,cands,alive,mean)\n')


def pairs(game):
    for root in game['roots']:
        if not RANGE[0]<=root['turn']<RANGE[1]:
            continue
        trace=root['trace']
        scores={int(i):v for i,v in trace.get('final_scores',{}).items()}
        features={int(i):v for i,v in trace.get('rank_features',{}).items()}
        indexed={tuple(sorted(a.items())):i for i,a in enumerate(trace['admitted'])}
        outcomes={}
        for result in root['results']:
            i=indexed.get(tuple(sorted(result['action'].items())))
            if i in scores and i in features:
                outcomes[i]=result['margin']
        for i,j in itertools.combinations(sorted(outcomes),2):
            yield (scores[i]-scores[j],
                   [a-b for a,b in zip(features[i],features[j])],
                   outcomes[i]-outcomes[j])


def fit(games):
    import numpy as np
    # A root can contribute multiple comparisons. Each source game has total weight one.
    rows=[list(pairs(game)) for game in games]
    usable=[row for row in rows if row]
    if len(usable)<8:
        raise ValueError('At least eight labeled maps are required')
    x = np.array([x for rows in usable for _,x,_ in rows])
    y = np.array([np.clip(gap/3000.,-.33,.33)-score for rows in usable for score,_,gap in rows])
    w = np.array([1/len(rows) for rows in usable for _ in rows])
    ridge = np.eye(x.shape[1])*7.
    coef = np.linalg.solve(x.T@(w[:,None]*x)+ridge,x.T@(w*y))
    return dict(weights=coef.round(8).tolist(),maps=len(usable),comparisons=len(y))


def evaluate(games, weights):
    rows=[]
    for game in games:
        examples=list(pairs(game))
        result=dict(seed=game['seed'],opponent=game['opponent'],comparisons=len(examples),
                    baseline_correct=0,candidate_correct=0,flips=0,delta=0.)
        for score,x,gap in examples:
            correction=max(-.055,min(.055,sum(a*b for a,b in zip(x,weights))))
            better=score+correction
            result['baseline_correct']+=int(score*gap>0)
            result['candidate_correct']+=int(better*gap>0)
            if (score>0)!=(better>0):
                result['flips']+=1
                result['delta']+=float(gap if better>0 else -gap)
        rows.append(result)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    with args.out.open('x') as f:
        f.write(build((root/'yan-v7.py').read_text(),json.loads(args.weights.read_text())['weights']))
