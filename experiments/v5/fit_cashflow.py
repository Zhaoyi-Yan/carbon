"""Estimate the incremental value of funding future workers from complete training games."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import numpy as np
from features import _V5_DIST,_v5_home,_v5_observe,_v5_context
from fit import ridge
import cashflow
cashflow.np=np;cashflow._V5_DIST=_V5_DIST;cashflow._v5_home=_v5_home


def samples(args):
    n,path=args
    frames=json.loads(gzip.decompress(path.read_bytes()))['steps']
    F=[];E=[];Y=[];locked=[]
    for t in range(int(n[:4],16)%4,299,4):
        o=frames[t][0]['observation'];pl=o['players'];trees=o['trees']
        ec,style=_v5_observe(pl,trees,t,298)
        E.append(_v5_context(ec,style,pl,t,298,0)[0])
        F.append(cashflow._v5_funding_features(pl,trees,t,298,0))
        f=frames[min(299,t+48)][0]['observation']['players']
        Y.append(((f[0][0]-f[1][0])-(pl[0][0]-pl[1][0]))/1000.)
        for side in (0,1):
            if not pl[side][3] and all(k!='COLLECTOR' for _,_,k in pl[side][2].values()) and pl[side][0]<20:
                locked.append(t)
    return n,np.stack(F),np.stack(E),np.array(Y),len(locked)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('split','prior','output','report'):p.add_argument(k,type=Path)
    p.add_argument('snapshots',type=Path,nargs='+')
    a=p.parse_args();split=json.loads(a.split.read_text())
    tr=set(split['training_matches']);va=set(split['validation_matches']);games={}
    for root in a.snapshots:
        for g in json.loads((root/'index.json').read_text()):
            path=root/g['file'];n=path.name.split('.')[0]
            if n in tr|va and path.exists():games[n]=path
    with ProcessPoolExecutor(2) as pool:rows=list(pool.map(samples,games.items(),chunksize=2))
    def matrix(names):
        sel=[v for v in rows if v[0] in names]
        return [np.concatenate([q[i] for q in sel]) for i in (1,2,3)]
    X,E,Y=matrix(tr);V,D,Z=matrix(va);data=dict(np.load(a.prior))
    ew=data['economic'];scale=np.maximum(X.std(0),.12)
    coef=ridge(X/scale,Y-E@ew,180.)/scale
    data['cashflow']=coef
    stats=dict(train_states=len(X),validation_states=len(V),locked_train_seat_frames=sum(q[4] for q in rows if q[0] in tr),
               old_mae=float(abs(Z-D@ew).mean()*1000.),new_mae=float(abs(Z-D@ew-V@coef).mean()*1000.))
    np.savez_compressed(a.output,**data);a.report.write_text(json.dumps(stats,indent=2)+'\n');print(stats,flush=True)


if __name__=='__main__':
    main()
