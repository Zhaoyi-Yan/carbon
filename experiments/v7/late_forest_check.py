"""Compare complete final 12-turn accounts with the late-tree filter and recorded replies."""
import argparse
import json
from collections import Counter
from pathlib import Path
from branch_replay import ROOT,OUT,HERE,step
from benchmark import configure_engine,limit_math_threads,load_strategy,new_game,observe
from close_replay import restore,archive_state
from late_forest import _v7_late_forest


def game(g,m):
    st,env=new_game(g['seed'],300);h=g['actions_and_post_cash'];start=287
    if len(h)!=299:return []
    tr=[m.BeliefTracker(m._Bel,j) for j in (0,1)];frames={};reference=None
    for t,(a,expected) in enumerate(h):
        if t==start:
            import copy
            shot=archive_state(st);reference=copy.deepcopy(tr)
        obs=[observe(st,j,t) for j in (0,1)]
        for j in (0,1):
            tr[j].step(obs[j]['players'],t,obs[j]['carbon']);tr[j].set_actions(obs[j]['players'],a[j])
        if t>=start:frames[t]=obs[0]['players']
        step(st,env,a);assert expected==[p[0] for p in st[0].observation.players]
    frames[299]=st[0].observation.players
    result=[]
    for side in (0,1):
        st,e=restore(shot,dict(env.configuration));tracker=reference[side];changes=[]
        mapping=[{u:u for u in z[2]} for z in frames[start]]
        for t in range(start,299):
            o=observe(st,side,t);old=o['players'];stock,_=tracker.step(old,t,o['carbon'])
            pairs=[]
            for s in (0,1):
                record=h[t][0][s];act={u:v for u,v in record.items() if u in old[s][1]};act.update({w:record[u] for u,w in mapping[s].items() if u in record and w in old[s][2]});pairs.append(act)
            new_action=_v7_late_forest(o,e.configuration,stock,pairs[side])
            if new_action!=pairs[side]:changes.append(t)
            pairs[side]=new_action;tracker.set_actions(old,new_action);step(st,e,pairs);new=st[0].observation.players
            for s in (0,1):
                real=frames[t+1][s][2];mapping[s]={u:w for u,w in mapping[s].items() if u in real and w in new[s][2]}
                births={(p,k):w for w,(p,_,k) in new[s][2].items() if w not in old[s][2]}
                for u,(p,_,k) in real.items():
                    if u not in frames[t][s][2] and (p,k) in births:mapping[s][u]=births[p,k]
            if not changes:assert h[t][1]==[p[0] for p in new]
        c=st[0].observation.players;margin=c[side][0]-c[1-side][0];baseline=g['cash'][side]-g['cash'][1-side]
        result.append(dict(seed=g['seed'],seat=side,name=g['players'][side]['name'],changes=changes,original=round(baseline,2),alternative=round(margin,2)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('files',nargs='+',type=Path);a=p.parse_args()
    limit_math_threads();configure_engine();m=load_strategy('crop_close',HERE/'checkpoints/soundings.py')
    _v7_late_forest.__globals__.update({k:getattr(m,k) for k in ('LAST_TURN','_V7_D','_V7_L','NB','WIDX','WACT')})
    rows=[];sums={}
    for path in a.files:
        part=[r for g in map(json.loads,path.open()) for r in game(g,m)];rows.extend(dict(batch=path.name,**r) for r in part)
        a1=[round(r['alternative']-r['original'],2) for r in part if r['changes']]
        sums[path.name]=dict(activated=len(a1),improved=sum(d>0 for d in a1),worse=sum(d<0 for d in a1),mean=round(sum(a1)/max(1,len(a1)),2))
        print(path.name,sums[path.name],flush=True)
    (OUT/'late-forest.json').write_text(json.dumps(dict(batches=sums,games=rows),indent=2)+'\n')
if __name__=='__main__':main()
