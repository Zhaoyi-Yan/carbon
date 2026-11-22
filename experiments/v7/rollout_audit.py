"""Compare two simulated continuations with paired real 5/10/15-turn cash differences."""
import argparse
import json
from pathlib import Path

import numpy as np
from branch_replay import ROOT, OUT, configure_engine
from benchmark import limit_math_threads,load_strategy
from starvation_replay import read_replay

TURNS=(40,70,90,130,164,190,220,255)
HORIZONS=(5,10,15)


def starts(module,game,side):
    tracker=module.BeliefTracker(module._Bel,side)
    frames=game['steps'];cfg={k:v for k,v in game['configuration'].items() if k!='randomSeed'}
    prev=None;boards=[]
    for t in range(min(TURNS[-1]+1,len(frames)-1)):
        f=frames[t]
        public=f[0]['observation'];players=public['players'];carbon=f[side]['observation']['carbon']
        est,_=tracker.step(players,t,carbon)
        if t in TURNS and t+max(HORIZONS)<len(frames):
            gold={}
            for n in HORIZONS:
                z=frames[t+n][0]['observation']['players'];gold[n]=z[side][0]-z[1-side][0]
            start={'step':t,'player':side,'players':players,'trees':public.get('trees',{}),'carbon':list(est)}
            boards.append(dict(t=t,side=side,board=start,prev=prev,gold=gold,config=cfg,forecast={}))
        tracker.set_actions(players,frames[t+1][side]['action'] or {})
        prev=players
    return boards


def rollout(model,roots,refit,prior=None,split=0):
    jobs=[dict(r,forecast={}) for r in roots]
    # Freeze the own rollout style to Artemis and the rival to the generic embedding
    # for this audit. The shipped entry selects both styles from observed history.
    for n in range(1,max(HORIZONS)+1):
        x=[];ids=[]
        for job in jobs:
            board=job['board'];x.extend(model._v5_pair(board['players'],board['step'],board['carbon'],job['prev']))
            styles=[20,20];styles[job['side']]=22;ids.extend(styles)
        use_old=prior is not None and n<=split
        engine=prior if use_old else model
        w,b,_=engine._V6R_TINY(np.array(x,dtype=np.float32),np.array(ids) if refit and not use_old else 0)
        for i,j in enumerate(jobs):
            o=j['board'];pl=o['players'];turn=o['step']
            actions=[model._sample(pl,s,w[2*i+s],b[2*i+s],np.random.RandomState(917+373*turn+17*s),turn) for s in (0,1)]
            j['prev']=pl;j['board']=model._turn(o,j['config'],actions)
            if n in HORIZONS:
                q=j['board']['players'];side=j['side']
                j['forecast'][n]=q[side][0]-q[1-side][0]
    return jobs


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('candidate',type=Path);ap.add_argument('public',type=Path)
    ap.add_argument('--switch',action='store_true');a=ap.parse_args();limit_math_threads();configure_engine()
    old=load_strategy('older_small',ROOT/'yan-v6.py');new=load_strategy('refit_small',a.candidate)
    old._torch.set_num_threads(1)
    index=json.loads((a.public/'index.json').read_text()); rows=[]
    args=[(a.public/r['file'],r['seat']) for r in index[:6]]
    args.append((a.public.parent/'starvation-4063.html.gz',1))
    for path,side in args:
        g=read_replay(path);r=starts(old,g,side);before=rollout(old,r,False);after=rollout(new,r,True)
        mixed=rollout(new,r,True,old,5) if a.switch else None
        for i,(b,improved) in enumerate(zip(before,after)):
            rows.append(dict(file=path.name,turn=b['t'],side=side,actual=b['gold'],old=b['forecast'],new=improved['forecast'],**({'switch5':mixed[i]['forecast']} if mixed else {})))
    mae={k:{n:round(np.mean([abs(r[k][n]-r['actual'][n]) for r in rows]),2) for n in HORIZONS} for k in (('old','new','switch5') if a.switch else ('old','new'))}
    (OUT/('rollout-switch.json' if a.switch else 'rollout-cash.json')).write_text(json.dumps(dict(mae=mae,rows=rows),indent=2)+'\n');print(len(rows),mae)


if __name__=='__main__':main()
