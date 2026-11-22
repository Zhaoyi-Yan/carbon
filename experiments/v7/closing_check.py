"""Replay unseen final moves after choosing a new decision against modeled replies."""
from pathlib import Path
import argparse, json, sys
from branch_replay import ROOT,step
from benchmark import configure_engine,limit_math_threads,load_strategy,new_game,observe


def run(candidate, games):
    limit_math_threads();configure_engine();m=load_strategy('closing_check',candidate)
    rows=[]
    for match in games:
        if match['turns']!=299 or match['status']!='completed':continue
        r,env=new_game(match['seed'],300);cfg=dict(env.configuration)
        seat=next(i for i,p in enumerate(match['players']) if p['name']!='yan-v6')
        tracker=m.BeliefTracker(m._Bel,seat);previous=None;h=match['actions_and_post_cash']
        for t,(moves,cashes) in enumerate(h):
            obs=observe(r,seat,t);pl=obs['players'];est,known=tracker.step(pl,t,obs['carbon'])
            if t==298:break
            tracker.set_actions(pl,moves[seat]);previous=pl;step(r,env,moves)
        enemy_x=m._quant(m.encode(pl,t,1-seat,est,[1.]*225,previous))
        # Keep the audit independent of the recorded rival's chosen last move.
        w,b=m._POLICY.forward(enemy_x,22)
        proposal=m._v7_settle(obs,cfg,est,moves[seat],(w,b))
        actions=list(moves);actions[seat]=proposal
        step(r,env,actions);cash=[p[0] for p in r[0].observation.players]
        old=cashes[seat]-cashes[1-seat];new=cash[seat]-cash[1-seat]
        rows.append(dict(seed=match['seed'],seat=seat,before=round(old,2),after=round(new,2),gain=round(new-old,2),
                         changed=proposal!=moves[seat]))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('games',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();rows=run(a.candidate,list(map(json.loads,a.games.open())))
    a.out.write_text(json.dumps(rows,indent=2)+'\n')
    print('gains',[r['gain'] for r in rows])
