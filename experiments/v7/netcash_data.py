"""Collect exact incremental cash targets; reuse frozen receipt inputs and cached v6 features."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import numpy as np
import netcash, receipts
from receipt_data import layout, TARGET_MATCH, TARGET_LAYOUT
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'experiments/v6r'
sys.path.insert(0,str(OLD))
import forecast, assets

DATA=ROOT.parent/'v7-data';CACHE=DATA/'receipts-cache';OUT=DATA/'netcash-rows.npz'
MODELPATH=HERE/'receipt-model.npz'
H=(20,35)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def initialize():
    receipts._V7_RECEIPT_DIST=forecast._FC_DIST
    receipts._V7_RECEIPT_STEPS=np.arange(1,11)
    with np.load(MODELPATH) as z:receipts._V7_RECEIPT_MODEL=dict(z)
    with np.load(OLD/'forecast-model.npz') as z:
        forecast._FC_WEIGHTS=tuple(z[k] for k in ('mean','inv','ridge','w1','b1','w2','b2','w3','b3'))
    assets.np=np;assets._V5_DIST=forecast._FC_DIST;assets._v5_home=forecast._fc_home;assets.LAST_TURN=298
    a=json.loads((OLD/'assets-model.json').read_text())
    assets._R6_CARGO=np.array(a['cargo']['coef']);assets._R6_TREE=np.array(a['tree']['coef'])


def select():
    allrows=json.loads((CACHE/'index.json').read_text())['games']
    prior={g['layout'] for g in json.loads((OLD/'forecast-validation.json').read_text())['games']}
    blacklist={TARGET_LAYOUT};files={TARGET_MATCH};seeds=set()
    shots=json.loads(gzip.decompress((ROOT/'analysis/v7/branch-breakout-replays.json.gz').read_bytes()))
    for game in shots.values():
        o=game['initial']['seats'][0]['observation']
        blacklist.add(digest(json.dumps([o['full_carbon'],[sorted(p[1].values()) for p in o['players']]],separators=(',',':'))))
        files.add(game['match_id']);seeds.add(game['configuration']['randomSeed'])
    seeds.add(205864169)
    def collect(v):
        if isinstance(v,int):seeds.add(v)
        elif isinstance(v,list):
            for q in v:collect(q)
        elif isinstance(v,dict):
            for q in v.values():collect(q)
    collect(json.loads((HERE/'panels.json').read_text()))
    rows=[r for r in allrows if r['layout'] not in blacklist and r['seed'] not in seeds and r['file'].split('-')[0] not in files and r['end']>=90]
    rows.sort(key=lambda r:digest('netcash-map-v1/'+r['layout']))
    # The deployed frozen receipt estimator used the other 4/5 in training. Old public
    # forecast/asset development layouts are also excluded from all final test scores.
    hold=[r for r in rows if int(r['layout'][:8],16)%5==0 and r['layout'] not in prior]
    pool=[r for r in rows if int(r['layout'][:8],16)%5!=0]
    split=[('fit',r) for r in pool[:290]]+[('tune',r) for r in pool[290:346]]+[('test',r) for r in hold]
    return split,dict(blacklisted_layouts=sorted(blacklist),blacklisted_games=sorted(files),excluded_seed_count=len(seeds),
                      previous_forecast_dev_maps=len(prior),total_receipt_maps=len(allrows),receipt_holdout_only=True)


def cached_forecasts():
    cache={}
    for f in ('forecast-rows.npz','forecast-external.npz'):
        p=ROOT.parent/'v6r-data'/f
        if not p.exists():continue
        with np.load(p) as z:
            games=json.loads(str(z['games']));t=z['T'];group=z['G'];x=z['X']
            for idx,g in enumerate(games):
                mask=group==idx;cache[g['layout']]={int(k):v for k,v in zip(t[mask],x[mask])}
    return cache


def receipts_batch(d):
    model=receipts._V7_RECEIPT_MODEL
    ts=d['t'][::2];ix=np.searchsorted(ts,d['wt'])*2+d['ws'].astype(np.intp)
    x=d['x'].reshape(-1,24);dist=d['d'];loads=d['load']
    h=np.tanh(x@model['w0']+model['b0']);o=h@model['w1']+model['b1']
    reach=np.arange(1,11)[None,:]>=np.maximum(1,dist[:,None])
    log=o[:,:11].copy();log[:,:10]=np.where(reach,log[:,:10],-100)
    prob=np.exp(log-log.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    inc=np.cumsum(loads[:,None]*prob[:,:10]+8*np.logaddexp(0.,o[:,11:])*reach,axis=1)
    by=np.stack([np.bincount(ix,weights=inc[:,j],minlength=len(ts)*2) for j in (4,9)],axis=1)
    n=np.bincount(ix,minlength=len(ts)*2);bands=np.select([n==0,n<=2,n<=5],[0,1,2],3)
    phase=np.repeat(np.where(ts<20,0,np.where(ts<100,1,2)),2)
    by=by+model['background'][bands,phase][:,[4,9]]
    return ts,by.reshape(-1,2,2)


def forecast_batch(raw):
    x=np.asarray(raw);mu,inv,w,w1,b1,w2,b2,w3,b3=forecast._FC_WEIGHTS
    phase=x[:,0,-4: -3];late=1.-x[:,0,-2:-1]
    z=np.clip((x-mu)*inv,-6.,6.);d=z[:,0]-z[:,1]
    lin=np.concatenate((d,d*phase,d*late),axis=1)@w
    h=np.maximum(0.,z@w1+b1);out=np.maximum(0.,h@w2+b2)@w3+b3
    return (lin+out[:,0]-out[:,1])*250.


def main():
    initialize();maps,exclusions=select();cache=cached_forecasts()
    xs=[];ys=[];bs=[];groups=[];times=[];cash=[];games=[];reused=0
    for i,(role,r) in enumerate(maps):
        with gzip.open(ROOT/r['source']/'replays'/r['file']) as f:g=json.load(f)
        frames=g['steps'];assert layout(g)==r['layout']
        gp=[frame[0]['observation'] for frame in frames]
        margin=np.asarray([o['players'][0][0]-o['players'][1][0] for o in gp])
        with np.load(CACHE/(r['layout']+'.npz')) as z:ts,flow=receipts_batch(z)
        size=len(frames)-1
        idx=np.flatnonzero(ts<=size-max(H));tlist=ts[idx]
        history={t:forecast._fc_summary(o['players'],o['trees']) for t,o in enumerate(gp)}
        raw=[];cc=cache.get(r['layout'],{})
        for t in tlist:
            o=gp[t]
            if t in cc:raw.append(cc[t]);reused+=1
            else:raw.append(forecast._fc_inputs(o['players'],o['trees'],t,298,history))
        public=forecast_batch(raw)
        for j,(t,ii) in enumerate(zip(tlist,idx)):
            pl=gp[t]['players'];meta=gp[t]['trees'];fl=flow[ii];old=public[j];board=dict(step=t,players=pl,trees=meta)
            asset=assets._r6_asset_score(board,0)*3000-margin[t]
            distance=sum((1-2*s)*v*max(0.,1-forecast._fc_home(tuple(p[1].values()))[q]/11.) for s,p in enumerate(pl) for q,v,k in p[2].values() if k=='COLLECTOR')
            if j==min(10,len(tlist)-1):
                assert np.allclose(fl,receipts._v7_receipts(pl,t,298),atol=2e-3)
                assert np.allclose(old,forecast._fc_predict(board,298,history),atol=3e-3)
            xs.append(netcash._v7_nc_inputs(pl,meta,int(t),298,fl,old))
            ys.append(margin[t+np.asarray(H)]-margin[t]);bs.append((fl[0,1]-fl[1,1],distance,asset,*old))
            cash.append(margin[t]);groups.append(i);times.append(t)
        games.append(dict(layout=r['layout'],file=r['file'],seed=r['seed'],fold=role,source=r['source']))
        if (i+1)%50==0:print('maps',i+1,len(maps),flush=True)
    np.savez_compressed(OUT,X=np.asarray(xs,'f'),Y=np.asarray(ys,'f'),B=np.asarray(bs,'f'),C=np.asarray(cash,'f'),T=np.asarray(times,'i2'),G=np.asarray(groups,'i2'),games=json.dumps(games),excluded=json.dumps(exclusions),reused_forecast_rows=reused)
    print('done',len(maps),len(xs),reused,flush=True)


if __name__=='__main__':main()
