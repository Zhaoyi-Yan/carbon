"""Replay the recorded two-seat games and attribute deposits to the actual worker in the referee."""
import argparse,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import selfcredit as sc
from localhead_data import load
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine,new_game
from trace_economy import rule_lines,CASH_RULES


def collect(m,g,side0,rows):
    from zerosum_env.envs.carbon import carbon,helpers
    state,env=new_game(g['seed'],g['episode_steps'])
    log=g['actions_and_post_cash'];history=[];banks=[];samples=[]
    before=None;land=np.array(m._ROLLOUT_LANDING)
    cash_lines=rule_lines(helpers.Board.next,CASH_RULES);code=helpers.Board.next.__code__
    def on_cash(p,k,new):
        f=sys._getframe(1)
        if k=='_cash' and f.f_code is code and cash_lines.get(f.f_lineno)=='bank':
            w=f.f_locals['worker']
            if w.player_id==p.id and new>p.cash:
                moved[(p.id,w.id)]=float(new-p.cash)
        return object.__setattr__(p,k,new)
    with patch.object(helpers.Player,'__setattr__',on_cash):
        for t,(pair,cash) in enumerate(log):
            pl=state[0].observation.players;history.append(pl)
            if t%4==g['seed']%4 and t+16<=len(log):
                # The return model uses public worker, home, threat and cash planes only.
                x=m._v5_pair(pl,t,np.zeros(225),before).reshape(2,54,225)
                for seat in (0,1):
                    charged=[(uid,p,l) for uid,(p,l,k) in pl[seat][2].items() if k=='COLLECTOR' and l>=35.]
                    if not charged:continue
                    pos=[p for uid,p,l in charged]
                    v=x[seat].T[land[pos]][:,:,sc.PLANES]
                    features,estimated=sc._selfcredit_features(v)
                    for i,(uid,p,l) in enumerate(charged):
                        action=m.WIDX[pair[seat].get(uid) or 'NONE']
                        samples.append((t,seat,uid,l,action,features[i]))
            before=pl
            moved={}
            for s,a in zip(state,pair):s.action=a
            carbon.compute_next_board(state,env)
            got=[p[0] for p in state[0].observation.players]
            if not np.allclose(cash,got,rtol=0,atol=.005):raise ValueError((g['seed'],t,cash,got))
            banks.append(moved)
    history.append(state[0].observation.players)
    if [p[0] for p in history[-1]]!=g['scores']:raise ValueError('terminal mismatch')
    for t,seat,uid,cargo,action,feats in samples:
        paid=0.;dies=False
        for u in range(t,t+16):
            paid+=banks[u].get((seat,uid),0.)
            if paid>=.5*cargo:break
            if uid not in history[u+1][seat][2]:
                dies=True;break
        rows.append((g['seed'],side0,seat,t,cargo,feats,action,int(paid>=.5*cargo),int(dies)))


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('parent',type=Path);a.add_argument('out',type=Path)
    p=a.parse_args();sc._SC_PLANES=sc.PLANES;configure_engine();m=load(p.parent)
    rows=[];meta={}
    for fn in ('v6r-qual-02-canopy.jsonl','v6r-qual-03-zenith.jsonl'):
        path=ROOT/'analysis'/fn;games=[json.loads(l) for l in path.read_text().splitlines()]
        meta[fn]=dict(sha=hashlib.sha256(path.read_bytes()).hexdigest(),games=[])
        for g in games:
            cand=next(i for i,p in enumerate(g['players']) if p['name']!='yan-v5')
            collect(m,g,cand,rows)
            meta[fn]['games'].append(dict(seed=g['seed'],candidate_seat=cand,layout=g['layout_sha256'],steps=g['turns']))
        print(fn,len(rows),flush=True)
    d={k:np.array([r[i] for r in rows]) for i,k in enumerate(('seed','candidate','seat','step','cargo','features','action','paid','dies'))}
    with p.out.open('xb') as f:np.savez_compressed(f,**d,meta=json.dumps(meta))


if __name__=='__main__':main()
