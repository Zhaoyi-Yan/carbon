"""Remove the final solo collector hire in each seat, with exact replay agreement before the intervention."""
import argparse
import json
from pathlib import Path
from statistics import mean
import numpy as np
from branch_replay import ROOT,OUT,HERE,step
from benchmark import configure_engine,limit_math_threads,load_strategy,new_game,observe
from close_replay import restore,archive_state
from late_hire import _v7_hire_screen,_v7_hire_features


def run(g,m,initial=None):
    h=g['actions_and_post_cash'];st,env=new_game(g['seed'],300)
    if initial is not None:
        from zerosum_env.envs.carbon import idgen
        st,env=restore(dict(seats=initial['seats'],id_counters=dict(idgen._id_dict)),initial['config'])
    ids=[set(q[1]) for q in st[0].observation.players]
    last=[next((i for i in range(len(h)-1,max(-1,len(h)-19),-1) if any(k in ids[s] for k in h[i][0][s])),None) for s in (0,1)]
    picks=[]
    for s,t in enumerate(last):
        if t is None:continue
        bids=[k for k in h[t][0][s] if k in ids[s]]
        if len(bids)==1 and h[t][0][s][bids[0]]=='RECCOLLECTOR':picks.append((s,t,bids[0]))
    if not picks:return []
    track=[m.BeliefTracker(m._Bel,s) for s in (0,1)];saved={};rows=[]
    for t,(moves,expected) in enumerate(h):
        for s in (0,1):
            o=observe(st,s,t);stock,_=track[s].step(o['players'],t,o['carbon'])
            for side,start,bid in picks:
                if s==side and t==start:
                    saved[s]=(archive_state(st),_v7_hire_features(o,env.configuration,stock,s,bid),
                              bid not in _v7_hire_screen(o,env.configuration,stock,moves[s]))
            track[s].set_actions(o['players'],moves[s])
        step(st,env,moves);assert expected==[p[0] for p in st[0].observation.players]
    for side,t0,bid in picks:
        checkpoint,features,skip=saved[side];state,e=restore(checkpoint,dict(env.configuration))
        for t in range(t0,len(h)):
            p=state[0].observation.players;a=[{uid:v for uid,v in move.items() if uid in p[s][1] or uid in p[s][2]} for s,move in enumerate(h[t][0])]
            if t==t0:a[side].pop(bid)
            step(state,e,a)
        cash=[q[0] for q in state[0].observation.players];alt=cash[side]-cash[1-side]
        actual=g['cash'][side]-g['cash'][1-side]
        rows.append(dict(seed=g['seed'],name=g['players'][side]['name'],seat=side,turn=t0,features=features,skip=skip,
                         original=round(actual,2),without=round(alt,2),net_of_removing=round(alt-actual,2)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('files',nargs='+',type=Path);a=p.parse_args()
    limit_math_threads();configure_engine();m=load_strategy('terminal_hires',HERE/'checkpoints/soundings.py')
    _v7_hire_features.__globals__.update({k:getattr(m,k) for k in ('LAST_TURN','_V7_D','np')})
    rows=[]
    for path in a.files:rows+=sum((run(g,m) for g in map(json.loads,path.open())),[])
    def summary(r):return dict(n=len(r),mean=round(mean(v['net_of_removing'] for v in r),2) if r else None,positive=sum(v['net_of_removing']>0 for v in r),negative=sum(v['net_of_removing']<0 for v in r),flipped=[v['seed'] for v in r if (v['original']>0)!=(v['without']>0)])
    out={name:summary([r for r in rows if (r['skip'] if name=='screened' else True)]) for name in ('all','screened')}
    (OUT/'last-hire-counterfactual.json').write_text(json.dumps(dict(summary=out,rows=rows),indent=2)+'\n');print(out)
if __name__=='__main__':main()
