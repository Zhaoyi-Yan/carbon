"""Supervise each tree and carrier on the future receipts of that same asset."""
import gzip
import json
import hashlib
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import assets
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/v5'))
from features import _V5_DIST,_v5_home
assets._V5_DIST=_V5_DIST;assets._v5_home=_v5_home
MOVES={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
def dest(p,a):
    r,c=MOVES.get(a,(0,0));return (p//15+r)%15*15+(p%15+c)%15

def read(path):
    frames=json.loads(gzip.decompress(path.read_bytes()))['steps'];end=len(frames)-1
    totals=[([],[],[]),([],[],[])]
    tree_returns={};worker_returns={}
    for t in range(end-1,-1,-1):
        before=frames[t][0]['observation'];pl=before['players'];after=frames[t+1][0]['observation']
        nxt=after['players'];newtree={};newworker={}
        for side in (0,1):
            p=pl[side];me=nxt[side];rival=nxt[1-side]
            ac=frames[t+1][side]['action'] or {}
            owned=set(me[1].values());enemy=set(rival[1].values())
            for uid,(pos,v,k) in p[2].items():
                if k!='COLLECTOR':continue
                q=dest(pos,ac.get(uid))
                if q in owned:ret=1.
                elif q in enemy:ret=-1.
                elif uid in me[2]:ret=.97*worker_returns.get(uid,0.)
                else:ret=-.20 if any(z==q and role=='COLLECTOR' for z,_,role in rival[2].values()) else 0.
                newworker[uid]=ret
            for uid,(pos,age) in p[3].items():
                owner=side if uid in me[3] else (1-side if uid in rival[3] else None)
                if owner is None:ret=0.
                else:
                    sign=1. if owner==side else -1.
                    ret=sign*(after['trees'][uid][1]+.968*tree_returns.get(uid,0.))
                newtree[uid]=ret
            if t%7==int(path.name[:5],16)%7:
                cf,tf=assets._r6_assets(pl,before['trees'],side,t,298)
                for j,(features,future) in enumerate(((cf,newworker),(tf,newtree))):
                    x,y,w=totals[j]
                    for uid,row in features.items():
                        x.append(row);y.append(future[uid]);w.append(min(3.,.25+p[2][uid][1]/90.) if j==0 else 1.)
        tree_returns,worker_returns=newtree,newworker
    return [(np.array(x),np.array(y),np.array(w)) for x,y,w in totals]

def main():
    raw,output=map(Path,sys.argv[1:]);files=sorted(raw.glob('*.json.gz'),key=lambda p:hashlib.sha256(('asset-v1'+p.stem).encode()).digest())
    with ProcessPoolExecutor(3) as pool:rows=list(pool.map(read,files,chunksize=3))
    cut=len(rows)*4//5;report={'training':[p.name for p in files[:cut]],'validation':[p.name for p in files[cut:]]}
    for idx,name in enumerate(('cargo','tree')):
        def matrix(group):
            return [np.concatenate([v[idx][i] for v in group if len(v[idx][i])]) for i in range(3)]
        x,y,w=matrix(rows[:cut]);u,v,z=matrix(rows[cut:]);scale=np.maximum(.15,x.std(0));xx=x/scale
        co=np.linalg.solve(xx.T@(xx*w[:,None])+np.eye(x.shape[1])*100.,xx.T@(w*y))/scale
        avg=np.average(y,weights=w);report[name]=dict(coef=co.tolist(),train=len(y),validation=len(v),constant_mae=float(np.average(np.abs(v-avg),weights=z)),mae=float(np.average(np.abs(u@co-v),weights=z)))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print([(k,report[k]['mae']) for k in ('cargo','tree')])
if __name__=='__main__':main()
