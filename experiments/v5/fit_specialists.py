"""Fit contextual per-worker returns from real deposits, tree creation and cargo transfers."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import numpy as np
from features import _V5_DIST,_v5_home
from backbone import load
from extract import TEACHERS
from fit import ridge
import specialists
specialists.np=np;specialists._v5_home=_v5_home;specialists._V5_DIST=_V5_DIST


def init():
    global MODEL
    MODEL=load()


def match(task):
    name,path,game=task
    f=json.loads(gzip.decompress(path.read_bytes()));frames=f['steps'];obs=[a[0]['observation'] for a in frames]
    out=[]
    for side,slot in enumerate(game['order']):
        teacher=TEACHERS.get(game['strategies'][slot]['id'])
        if teacher is None:continue
        flows=defaultdict(lambda:np.zeros(300));base=set(obs[0]['players'][side][1].values())
        for t in range(1,300):
            old=obs[t-1]['players'][side];pl=obs[t]['players'][side]
            for uid,(pos,load,kind) in old[2].items():
                ledger=flows[uid]
                if uid not in pl[2]:
                    ledger[t]-=32.*min(1.,(299-t)/30.)
                    continue
                new=pl[2][uid]
                if kind=='COLLECTOR' and new[0] in base and pos not in base:
                    flows[uid][t]+=load
            for tid,(pos,age) in pl[3].items():
                if tid in old[3]:continue
                uid=obs[t]['trees'][tid][0]
                if uid is None:continue
                income=0.
                for k in range(t+1,min(300,t+38)):
                    if tid not in obs[k]['players'][side][3]:break
                    income+=obs[k]['trees'][tid][1]*.98**(k-t)
                enemy_tree=any(p==pos for p,age in obs[t-1]['players'][1-side][3].values())
                field=sum(len(p[3]) for p in obs[t-1]['players'])
                cost=20. if enemy_tree else (20.+5.*1.235**min(30,field) if old[3] else 20.)
                flows[uid][t]+=income*(1.55 if enemy_tree else 1.)-cost
        prefix={uid:np.cumsum(v) for uid,v in flows.items()};tr=MODEL.BeliefTracker(MODEL._Bel,side)
        phase=int(name[:5],16)%5
        for t in range(299):
            pl=obs[t]['players']
            est,_=tr.step(pl,t,frames[t][side]['observation']['carbon'])
            if t%5==phase:
                ahead=min(299,t+30)
                for uid,(pos,cargo,kind) in pl[side][2].items():
                    if uid not in prefix:continue
                    end=obs[ahead]['players'][side][2].get(uid)
                    def bank(p,v):return v*(1.-.045*_v5_home(tuple(base))[p])
                    delta=(bank(end[0],end[1]) if end else 0.)-bank(pos,cargo)
                    reward=(prefix[uid][ahead]-prefix[uid][t]+delta)/100.
                    ctx=specialists._v5_assign_features(pl,est,t,298,side,uid)
                    out.append((teacher,int(kind=='PLANTER'),ctx,np.clip(reward,-7.,7.)))
            tr.set_actions(pl,frames[t+1][side]['action'])
    return name,out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('split','prior','output','report'):p.add_argument(arg,type=Path)
    p.add_argument('snapshots',nargs='+',type=Path)
    a=p.parse_args();split=json.loads(a.split.read_text());train=set(split['training_matches']);valid=set(split['validation_matches'])
    tasks={}
    for root in a.snapshots:
        for g in json.loads((root/'index.json').read_text()):
            path=root/g['file'];n=path.name.split('.')[0]
            if n in train|valid and path.exists():tasks.setdefault(n,(n,path,g))
    with ProcessPoolExecutor(2,initializer=init) as pool:rows=list(pool.map(match,tasks.values(),chunksize=2))
    params=dict(np.load(a.prior));coeff=[];centers=[];stats=[]
    for role in (0,1):
        def table(names):
            flat=[(j,x,y) for n,items in rows if n in names for j,k,x,y in items if k==role]
            j,x,y=zip(*flat);return np.array(j),np.stack(x),np.array(y)
        T,X,Y=table(train);tt,xx,yy=table(valid)
        scale=np.maximum(X.std(0),.15);base=ridge(X/scale,Y,200.)/scale
        err=Y-X@base
        models=np.stack([ridge(X[T==j]/scale,err[T==j],400.)/scale for j in range(20)])
        coeff.append(models);centers.append(models@X.mean(0))
        stats.append(dict(train=len(T),validation=len(tt),pooled_mae=float(abs(yy-xx@base).mean()),
                          specialist_mae=float(abs(yy-xx@base-(xx*models[tt]).sum(1)).mean())))
    params['unit_route']=np.stack(coeff);params['unit_center']=np.stack(centers)
    np.savez_compressed(a.output,**params);a.report.write_text(json.dumps(stats,indent=2)+'\n')
    print(stats,flush=True)


if __name__=='__main__':
    main()
