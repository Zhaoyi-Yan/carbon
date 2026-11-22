"""Reconcile cash and count severe liquidity droughts on both seats of a frozen tournament."""
import argparse
import json
from pathlib import Path
from branch_replay import step
from benchmark import configure_engine, limit_math_threads, new_game


def count(rows,sha):
    results=[]
    for r in rows:
        if r['status']!='completed':continue
        st,env=new_game(r['seed'],r['episode_steps'])
        seat=next(i for i,p in enumerate(r['players']) if p['sha256']==sha)
        data=[dict(severe_turns=0,longest_severe=0,without_income_source=0,longest_without_income=0) for _ in range(2)]
        streaks=[[0,0],[0,0]]
        for turn,(moves,balances) in enumerate(r['actions_and_post_cash'],1):
            step(st,env,moves);pl=st[0].observation.players
            assert [p[0] for p in pl]==balances
            for s,p in enumerate(pl):
                cargo=sum(c for _,c,k in p[2].values() if k=='COLLECTOR')
                is_severe=30<=turn<=250 and p[0]<30 and cargo>=90 and pl[1-s][0]-p[0]>=60
                empty=not any(k=='COLLECTOR' for _,_,k in p[2].values()) and not p[3] and p[0]<20
                d=data[s];a=streaks[s];a[0]=(a[0]+1) if is_severe else 0;a[1]=(a[1]+1) if empty else 0
                d['severe_turns']+=int(is_severe);d['without_income_source']+=int(empty)
                d['longest_severe']=max(d['longest_severe'],a[0]);d['longest_without_income']=max(d['longest_without_income'],a[1])
        results.append(dict(seed=r['seed'],seat=seat,candidate=data[seat],baseline=data[1-seat]))
    def total(name):
        v=[r[name] for r in results]
        return dict(games=len(v),sustained_low_cash=sum(d['longest_severe']>=10 for d in v),peak_low_cash_drought=max(d['longest_severe'] for d in v),
                    lost_all_income=sum(d['longest_without_income']>=10 for d in v),peak_without_income=max(d['longest_without_income'] for d in v))
    return dict(candidate_sha256=sha,candidate=total('candidate'),frozen_v6=total('baseline'),games=results,
                severe='Turns 30..250, cash < 30, cargo >= 90, cash deficit >= 60, at least ten consecutive turns.',
                lost_all_income='No collector, no tree and cash < 20. External enemy deposits on an own base can still restore liquidity.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('games',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();limit_math_threads();configure_engine()
    result=count(map(json.loads,a.games.open()),json.loads(a.manifest.read_text())['sha256'])
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(result['candidate'],result['frozen_v6'])
