"""Regress the two actions that led to 282 turns without a source of working capital."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from branch_replay import step
from benchmark import configure_engine,limit_math_threads,load_strategy,new_game,observe


def run(p,archive):
    limit_math_threads();configure_engine()
    m=load_strategy('test_collector',p)
    rec=next(r for r in map(json.loads,archive.open()) if r['seed']==499044317 and r['players'][0]['name']=='quay')
    state,env=new_game(rec['seed'],300)
    rows=[];last=None
    for t,(moves,cash) in enumerate(rec['actions_and_post_cash']):
        obs=observe(state,0,t);own,foe=obs['players']
        if t in (14,16):
            logits=np.zeros((225,5));alternative=m._v7_solvency(obs,env.configuration,moves[0],logits)
            # Equal logits are enough to prove the finance gate and avoid every unsafe next square.
            changes={k:[moves[0].get(k),alternative.get(k)] for k in (moves[0].keys()|alternative.keys()) if moves[0].get(k)!=alternative.get(k)}
            if t==14:
                assert any(a=='RECPLANTER' and b is None for a,b in changes.values())
            else:
                uid,(pos,cargo,_)=next((u,w) for u,w in own[2].items() if w[2]=='COLLECTOR')
                dst=m._V7_L[pos,m.WIDX.get(alternative.get(uid),0)]
                assert changes and all(m._V7_D[dst,p]>1 for p,c,k in foe[2].values() if k=='COLLECTOR' and c<=cargo)
            rows.append(dict(step=t,bank=own[0],changes=changes))
        step(state,env,moves)
        assert [p[0] for p in state[0].observation.players]==cash
    return dict(candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seed=rec['seed'],seat=0,observed=rec['scores'],decisions=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('games',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();result=run(a.candidate,a.games);a.out.write_text(json.dumps(result,indent=2)+'\n');print(result['decisions'])
