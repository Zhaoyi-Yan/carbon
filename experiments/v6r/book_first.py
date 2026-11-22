"""Check the Artemis response to a revealed two-collector opener and its first tree."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from book import ROOT
from book_train import load_policy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();m=load_policy()
    rows=[json.loads(s) for s in (ROOT/'analysis/v6r-artemis-games.jsonl').read_text().splitlines()]
    cases=[]
    for r in rows:
        if r['name']!='Artemis':continue
        path=a.data/'replays'/f"{r['match']}-{r['game']}.json.gz"
        blob=path.read_bytes();assert hashlib.sha256(blob).hexdigest()==r['file_sha256']
        fr=json.loads(gzip.decompress(blob))['steps'];side=r['seat'];op=1-side
        p1=fr[1][0]['observation']['players'];actors=p1[op][2]
        if sorted(v[2] for v in actors.values())!=['COLLECTOR','COLLECTOR'] or p1[side][2]:continue
        # The choice is made after the enemy workforce appears, at state 1.
        tr=m.BeliefTracker(m._Bel,side);m._ST.clear();prev=None
        for t in (0,1):
            st=fr[t];shared=st[0]['observation'];pl=shared['players'];carbon=st[side]['observation']['carbon']
            est,known=tr.step(pl,t,carbon);x=m._quant(m.encode(pl,t,side,est,known,prev))
            obs=dict(step=t,player=side,players=pl,carbon=carbon,trees=shared['trees'])
            _,base=m._v5_root(obs,est,x,prev)
            tr.set_actions(pl,fr[t+1][side]['action'] or {});prev=pl
        lower,upper=sorted(p1[side][1].values())
        raw=[int(base[q].argmax()) for q in (lower,upper)]
        ids=fr[2][0]['observation']['players'][side][2]
        uid,(spawn,_,_)=next((k,v) for k,v in ids.items() if v[2]=='PLANTER')
        first_tree=r['first_tree']
        tid,(cell,_)=next(iter(fr[first_tree][0]['observation']['players'][side][3].items()))
        planter_id=fr[first_tree][0]['observation']['trees'][tid][0]
        move=m.WIDX[fr[3][side]['action'][uid]]
        ix=set(m._ROLLOUT_LANDING[spawn])
        history=[fr[t][side]['action'].get(uid,'NONE') for t in range(3,first_tree+1)]
        cases.append(dict(match=r['match'],layout=r['layout'],seat=side,lower=lower,upper=upper,spawn=spawn,
                          planter_spawn_high=spawn==upper,collector_spawn_low=any(k=='COLLECTOR' and q==lower for q,l,k in ids.values()),
                          frozen_argmax_low_high=raw,frozen_matched=raw==[1,2],move=move,tree_step=first_tree,cell=cell,
                          first_planter_made_tree=uid==planter_id,tree_distance=int(m.FEAT_DIST[spawn,cell]),route=history))
    with a.output.open('x') as f:json.dump(cases,f,indent=2)


if __name__=='__main__':main()
