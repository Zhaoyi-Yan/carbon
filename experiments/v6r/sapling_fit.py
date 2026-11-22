"""Fit and test signed, discounted future income of each young tree using legal-view features."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import gzip
import hashlib
import json
import importlib.util
from pathlib import Path
import numpy as np
import assets
import sapling

ROOT=Path(__file__).resolve().parents[2]


def returns(frames):
    future={}; labels=[None]*len(frames)
    for t in range(len(frames)-2,-1,-1):
        p=frames[t][0]['observation']['players']
        after=frames[t+1][0]['observation']; then=after['players']
        nxt={}
        for side,pl in enumerate(p):
            for uid in pl[3]:
                if uid in then[side][3]:sgn=1
                elif uid in then[1-side][3]:sgn=-1
                else:sgn=0
                nxt[uid]=sgn*(after['trees'][uid][1]+.968*future.get(uid,0)) if sgn else 0.
        labels[t]=nxt;future=nxt
    return labels


def data_for(row,data,m,coef):
    g=json.load(gzip.open(data/row['file']));frames=g['steps']; h=hashlib.sha256(row['id'].encode()).digest()
    seat=h[1]%2;tracker=m.BeliefTracker(m._Bel,seat)
    traces=returns(frames); rows=[];old=None
    for t,fr in enumerate(frames[:-1]):
        o=fr[0]['observation'];pl=o['players']
        estimate,_=tracker.step(pl,t,fr[seat]['observation']['carbon'])
        tracker.set_actions(pl,frames[t+1][seat]['action'] or {})
        young=any(age<=6 for p in pl for _,age in p[3].values())
        audit=(t+h[2])%30==0
        if young or audit:
            board=dict(players=pl,carbon=estimate,step=t)
            context=sapling._sapling_context(pl,old)
            for side,p in enumerate(pl):
                if not (audit or any(age<=6 for _,age in p[3].values())):continue
                _,woods=assets._r6_assets(pl,o['trees'],side,t,298)
                for uid,(_,age) in p[3].items():
                    if age>6 and not audit:continue
                    base=float(woods[uid]@coef)
                    x=sapling._sapling_features(board,o['trees'],side,uid,base,context,298) if age<=6 else [0.]*len(sapling.FEATURES)
                    rows.append([t,age,base,traces[t][uid],*x])
        old=pl
    return np.asarray(rows,dtype=float),seat


def fit(x,y):
    # Huber weights downweight rare very large cash taxes and ownership reversals.
    sc=np.maximum(.12,np.sqrt(np.mean(x*x,axis=0))); z=x/sc
    w=np.zeros(x.shape[1])
    for _ in range(8):
        importance=np.minimum(1.,22./np.maximum(np.abs(y-z@w),1e-6))
        w=np.linalg.solve(z.T@(z*importance[:,None])+np.eye(len(w))*40,z.T@(importance*y))
    return w/sc


def bounds(age):
    return (age==1,(age>=2)&(age<=3),(age>=4)&(age<=6),age>6)


def metrics(y,baseline,proposed):
    return dict(n=int(len(y)),mean_actual=round(float(y.mean()),3),
                base_mae=round(float(np.abs(y-baseline).mean()),4),sapling_mae=round(float(np.abs(y-proposed).mean()),4),
                base_rmse=round(float(np.mean((y-baseline)**2)**.5),4),
                sapling_rmse=round(float(np.mean((y-proposed)**2)**.5),4),
                base_mean=round(float(baseline.mean()),3),sapling_mean=round(float(proposed.mean()),3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--train-limit',type=int,default=10000)
    p.add_argument('--holdout-limit',type=int,default=10000)
    a=p.parse_args();a.cache.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('ref',ROOT/'yan-v5.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for key in ('NB','NB8','DIST'):setattr(sapling,key,getattr(m,key))
    assets._V5_DIST=m._V5_DIST;assets._v5_home=m._v5_home
    base=json.loads((ROOT/'experiments/v6r/assets-model.json').read_text());coef=np.array(base['tree']['coef'])
    original_training=set(base['training'])
    source=json.loads((a.data/'index.json').read_text())
    split={};maps={}
    for name,is_train,cap in [('training',True,a.train_limit),('validation',False,a.holdout_limit)]:
        selected=[r for r in source if (Path(r['file']).name in original_training)==is_train]
        selected.sort(key=lambda r:hashlib.sha256(('sapling-v1:'+r['id']).encode()).digest())
        allrows=[];ids=[]
        for r in selected[:cap]:
            path=a.cache/(r['id']+'.npz')
            if path.is_file():
                with np.load(path) as cached:batch=cached['rows'];seat=int(cached['seat'])
            else:
                batch,seat=data_for(r,a.data,m,coef)
                np.savez_compressed(path,rows=batch,seat=seat)
            allrows.append(batch);ids.append(dict(match=r['id'],observer=seat,rows=len(batch)))
        split[name]=allrows;maps[name]=ids
    training=np.concatenate(split['training']);validation=np.concatenate(split['validation'])
    masks=bounds(training[:,1]);weights=[]
    for keep in masks[:3]:
        rows=training[keep]
        weights.append(fit(rows[:,4:],rows[:,3]-rows[:,2]))
    age=validation[:,1];y=validation[:,3];old=np.clip(validation[:,2],-90,300);v=validation[:,2].copy()
    for select,w in zip(bounds(age),weights): v[select]+=validation[select,4:]@w
    pred=np.clip(v,-90,300)
    report={k:metrics(y[z],old[z],pred[z]) for k,z in zip(('age1','age2_3','age4_6','age7_plus'),bounds(age))}
    birth=(age==1)
    train_birth=masks[0]
    cols=[0,1,2,3,4,5,6,7]
    simple=fit(training[train_birth,4:][:,cols],training[train_birth,3]-training[train_birth,2])
    soil_only=np.clip(validation[birth,2]+validation[birth,4:][:,cols]@simple,-90,300)
    birth_comparison=dict(soil_and_baseline_mae=round(float(np.mean(np.abs(y[birth]-soil_only))),4),
                          correlation={k:round(float(np.corrcoef(v,y[birth])[0,1]),4)
                                       for k,v in [('original',old[birth]),('soil_and_baseline',soil_only),('full',pred[birth])]})
    report['age1_3']=metrics(y[age<=3],old[age<=3],pred[age<=3])
    by_map=[];offset=0
    for r,arr in zip(maps['validation'],split['validation']):
        indexes=np.flatnonzero(arr[:,1]<=3)+offset;offset+=len(arr)
        by_map.append(float(np.mean(np.abs(y[indexes]-old[indexes])-np.abs(y[indexes]-pred[indexes]))))
    held_back = {}
    if len(split['validation'])>12:
        first_unseen=sum(len(batch) for batch in split['validation'][:12])
        unseen=(np.arange(len(y))>=first_unseen)
        held_back={k:metrics(y[z&unseen],old[z&unseen],pred[z&unseen]) for k,z in zip(('age1','age2_3','age4_6','age7_plus'),bounds(age))}
    ground=training[:,4+3:4+6].sum(axis=1); q=np.quantile(ground[training[:,1]==1],[.25,.75]); vg=validation[:,4+3:4+6].sum(axis=1)
    bins={k:metrics(y[z],old[z],pred[z]) for k,z in [('low',(age==1)&(vg<=q[0])),('high',(age==1)&(vg>=q[1]))]}
    result=dict(method='Signed actual future receipts of the same tree, 0.968 discount per step; player perspective flips at seizure. Same labels and clipping as assets-model. Feature input uses a hash-selected legal observer for both players. All validation matches excluded from both regressions.',
                age1_ablation=birth_comparison,features=sapling.FEATURES,coefficients=[w.tolist() for w in weights],maps=maps,groups=report,
                untouched_after_12_map_pilot=held_back,young_1_3_improved_maps=sum(i>0 for i in by_map),validation_maps=len(by_map),
                age1_ground_quartile_cutoffs=q.tolist(),age1_ground_quartiles=bins)
    a.output.write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps(dict(groups=report,improved_maps=sum(i>0 for i in by_map),maps=len(by_map),bins=bins)),flush=True)


if __name__=='__main__':
    main()
