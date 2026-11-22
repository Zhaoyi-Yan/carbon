"""Reconcile the loss on both seats of the closest Soundings layout."""
import json
from collections import Counter
from pathlib import Path
from branch_replay import OUT,step
from benchmark import limit_math_threads,configure_engine,new_game
from close_replay import referee_events


def distance(a,b):
    y,x=divmod(a,15);j,i=divmod(b,15)
    return min(abs(y-j),15-abs(y-j))+min(abs(x-i),15-abs(x-i))


def main():
    limit_math_threads();configure_engine()
    f=OUT/'05-qualifying.jsonl';items=[json.loads(l) for l in f.open()]
    groups={}
    for g in items:
        s=next(i for i,p in enumerate(g['players']) if p['name']=='soundings')
        groups.setdefault(g['seed'],[]).append(g['scores'][s]-g['scores'][1-s])
    losses=[k for k,v in groups.items() if len(v)==2 and max(v)<0]
    results=[]
    for g in items:
        if g['seed'] not in losses:continue
        side=next(i for i,p in enumerate(g['players']) if p['name']=='soundings')
        st,env=new_game(g['seed'],300);ledger=[Counter(),Counter()];frames={};drops=[]
        for t,(action,expected) in enumerate(g['actions_and_post_cash']):
            old=st[0].observation.players;events=[]
            with referee_events(events):step(st,env,action)
            p=st[0].observation.players
            assert expected==[q[0] for q in p]
            for e in events:
                if e[0] in ('bank','base_collision','tree_yield','plant','seize','hire_collector','hire_planter'):
                    ledger[e[1]][e[0]]+=e[2]
            for s in (0,1):
                for u,(z,v,k) in old[s][2].items():
                    if u not in p[s][2] and v>=30 and t>=240:drops.append(dict(turn=t+1,team=s,cargo=v,pos=z,command=action[s].get(u)))
            if t+1 in (190,225,250,265,275,280,285,290,293,295,297,298,299):
                frames[t+1]=[dict(cash=a[0],trees=len(a[3]),staff=len(a[2]),load=round(sum(v for q,v,k in a[2].values()),2),
                                  lots=sorted([(round(v,2),min(distance(q,b) for b in a[1].values())) for q,v,k in a[2].values() if v>=10],reverse=True),
                                  ledger={k:round(v,2) for k,v in ledger[s].items()}) for s,a in enumerate(p)]
        results.append(dict(seed=g['seed'],seat=side,frames=frames,drops=drops))
        print(side,[(t,round(v[side]['cash']-v[1-side]['cash'],2),v[side]['load'],v[1-side]['load']) for t,v in frames.items()])
    (OUT/'soundings-close-losses.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
