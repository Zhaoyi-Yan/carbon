"""Fit and evaluate the motion residual on maps disjoint from both the target and original validation."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import receipt_motion as motion
from receipt_check import predict_env,real_target
from receipt_data import targets,ROOT,TARGET_MATCH,TARGET_LAYOUT,layout,impl
HERE=Path(__file__).resolve().parent
m=motion
m._V7_RECEIPT_DIST=impl._V7_RECEIPT_DIST


def extract(g,fn,every,include=None):
    f=g['steps'];last=g['configuration']['episodeSteps']-2;end=len(f)-1
    increments,_=targets(g);cash=np.vstack((np.zeros(2),increments.cumsum(0)))
    raw=[];features=[];truth=[];times=[];latencies=[];steps=[]
    for t in range(0,end+1):
        if t%every and (include is None or t not in include):continue
        pl=f[t][0]['observation']['players']
        prev=f[t-1][0]['observation']['players'] if t else None
        older=f[t-2][0]['observation']['players'] if t>1 else None
        p=fn(pl,t,last)
        y=np.stack([cash[min(end,t+h)]-cash[t] for h in (5,10)],axis=-1)
        start=time.perf_counter_ns()
        x=m._v7_motion_features(pl,prev,older,p,t,last)
        latencies.append((time.perf_counter_ns()-start)/1000)
        raw.append(p);features.append(x);truth.append(y);times.append([max(0,last+1-t)]*2);steps.append(t)
    return np.concatenate(raw),np.concatenate(features),np.concatenate(truth),np.asarray(times).ravel(),latencies,steps


def fit(raw,x,y,left,dim):
    coefficients=np.zeros((2,x.shape[-1]))
    for i in range(2):
        ok=(left>=10)
        X=x[ok,i,:dim]; target=y[ok,i]-raw[ok,i]
        scale=np.sqrt((X*X).mean(0)).clip(.02,None)
        X=X/scale
        penalty=np.eye(dim)*len(X)*.012;penalty[0,0]*=.02
        coef=np.zeros(dim)
        for _ in range(8):
            error=target-X@coef
            # Huber cash loss limits leverage from a single late 1,000+ collision on a map.
            w=np.minimum(1.,(95.+50.*i)/np.maximum(1.,np.abs(error)))
            coef=np.linalg.solve(X.T@(w[:,None]*X)+penalty,X.T@(w*target))
        coefficients[i,:dim]=coef/scale
    return coefficients


def prediction(raw,x,left,w):
    out=np.maximum(0.,raw+np.einsum('nhf,hf->nh',x,w))
    out[:,1]=np.where(left<=5,out[:,0],np.maximum(out[:,0],out[:,1]))
    out[left<=0]=0
    return out


def logistic(pred,y):
    a=np.column_stack([np.ones(len(pred)),np.log1p(pred)])
    beta=np.zeros(2)
    for _ in range(10):
        p=expit(a@beta);v=(p*(1-p)).clip(.001,None)
        beta+=np.linalg.solve(a.T@(v[:,None]*a)+np.eye(2)*.01,a.T@(y-p))
    return beta


def metrics(y,guess,log):
    err=guess-y
    freq=y>0
    prob=np.stack([expit(log[i][0]+log[i][1]*np.log1p(guess[:,i])) for i in range(2)],axis=1)
    return dict(mae=np.abs(err).mean(0).round(4).tolist(),rmse=np.sqrt((err**2).mean(0)).round(4).tolist(),
                mean_predicted_cash=guess.mean(0).round(3).tolist(),mean_realized_cash=y.mean(0).round(3).tolist(),
                observed_receipt_frequency=freq.mean(0).round(4).tolist(),forecast_receipt_frequency=prob.mean(0).round(4).tolist(),
                brier=((prob-freq)**2).mean(0).round(5).tolist(),auc=[round(roc_auc_score(freq[:,i],prob[:,i]),5) for i in range(2)])


def credited_after_190():
    from collections import Counter
    from branch_replay import SNAPSHOT,INPUT,configure_engine,step
    from close_replay import restore,referee_events
    configure_engine()
    snap=json.loads(SNAPSHOT.read_text());record=json.loads(INPUT.read_text())
    assert record['match_id']==TARGET_MATCH
    state,env=restore(snap['snapshot'],snap['configuration'])
    pl=state[0].observation.players[1]
    alive={u for u in pl[2]};totals={};paid=set()
    for t in range(190,200):
        events=[]
        with referee_events(events):step(state,env,record['actions'][t])
        for k,seat,value,*info in events:
            if seat!=1 or k not in ('bank','harvest','loot','enemy_tree') or info[0] not in alive or info[0] in paid:continue
            uid=info[0]
            c=totals.setdefault(uid,Counter(initial=pl[2][uid][1]))
            c[k]+=value
            if k=='bank':
                paid.add(uid);c['receipt_step']=t+1
    return {u:{k:round(v,3) for k,v in totals[u].items()} for u in sorted(paid)}


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--cache',type=Path,required=True);a.add_argument('--target',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--model',type=Path,required=True)
    args=a.parse_args();cache=args.cache
    idx=json.loads((cache/'index.json').read_text());old=json.loads((ROOT/'analysis/v7/receipt-validation.json').read_text())
    assert hashlib.sha256((HERE/'receipt-model.npz').read_bytes()).hexdigest()==old['model_sha256']
    games=sorted(idx['games'],key=lambda r:r['layout'])
    assert len(set(r['layout'] for r in games))==len(games)
    assert not any(TARGET_MATCH in r['file'] or r['layout']==TARGET_LAYOUT for r in games)
    valid_keys=set(old['validation_layouts'])
    # Systematic by hash before any measurements. Use the entire original held-out split.
    training=[r for r in games if r['layout'] not in valid_keys][::4]
    valid=[r for r in games if r['layout'] in valid_keys]
    fn=predict_env()
    data=[];timings=[];groups=[];all_ids=[];map_stats=[]
    for group,rows in enumerate([training,valid]):
        items=[]
        for n,row in enumerate(rows):
            p=Path(row['source'])/'replays'/row['file']
            game=json.load(gzip.open(p,'rt'))
            assert layout(game)==row['layout']
            b,x,y,left,ms,_=extract(game,fn,6)
            items.append((b,x,y,left));timings.extend(ms);groups.extend([n]*len(b) if group else [])
            if n%40==0:print('extracted',group,n,len(rows),flush=True)
        data.append(tuple(np.concatenate(v) for v in zip(*items)))
        all_ids.append([(r['file'],r['layout']) for r in rows])
    (bt,xt,yt,lt),(bv,xv,yv,lv)=data;groups=np.asarray(groups)
    models={'same_features_calibrated':fit(bt,xt,yt,lt,4),'motion':fit(bt,xt,yt,lt,xt.shape[-1])}
    weights=models['motion']
    args.model.write_text(json.dumps(dict(weights=weights.tolist(),feature_dim=xt.shape[-1],forecast_order=[5,10]),separators=(',',':'))+'\n')
    pred={'frozen':bv};logs={};training_pred={'frozen':bt}
    for name,w in models.items():
        pred[name]=prediction(bv,xv,lv,w);training_pred[name]=prediction(bt,xt,lt,w)
    for name,p in training_pred.items():logs[name]=[logistic(p[:,i],(yt[:,i]>0).astype(float)) for i in range(2)]
    validation={name:metrics(yv,p,logs[name]) for name,p in pred.items()}
    for i,row in enumerate(valid):
        mask=groups==i
        map_stats.append(dict(layout=row['layout'],file=row['file'],mae={k:np.abs(p[mask]-yv[mask]).mean(0).round(3).tolist() for k,p in pred.items()}))
    # The target is deliberately opened only after both regressions and frequency calibrators are frozen.
    target=real_target(args.target);assert layout(target)==TARGET_LAYOUT
    b,x,y,left,lat,steps=extract(target,fn,1);where=steps.index(190)*2;slice_=slice(where,where+2)
    tp={'frozen':b,**{k:prediction(b,x,left,w) for k,w in models.items()}}
    target_metrics={k:metrics(y,p,logs[k]) for k,p in tp.items()}
    # Root and hypothetical-board frozen integration must agree with the offline features at this branch.
    t=190;frames=target['steps'];pl=lambda k:frames[k][0]['observation']['players']
    live=m._v7_motion_apply(b[slice_],x[slice_],weights,299-t)
    assert np.allclose(live,tp['motion'][slice_])
    import challenger
    source=challenger.build((ROOT/'yan-v6.py').read_bytes())
    code=m.build(source,args.model);marker=code.index('_V7_MOTION_COEF=');end=code.index('def _v7_modes',marker)
    stub=dict(np=np,_V7_RECEIPT_DIST=impl._V7_RECEIPT_DIST,_v7_receipts=fn,LAST_TURN=298,_ST={'prev':pl(189),'receipt_previous2':pl(188)})
    exec(compile(code[marker:end],'<motion-flow-compat>','exec'),stub)
    for board in ({'players':pl(190),'step':190},{'players':pl(190),'step':190,'_sap_previous':pl(189),'_v7_motion_older':pl(188)}):
        assert np.allclose(stub['_v7_flows'](board),live)
        assert stub['_v7_flows'](board) is board['_v7_flow']
    improvement={}
    for reference in ('frozen','same_features_calibrated'):
        paired=np.array([np.array(r['mae']['motion'])-r['mae'][reference] for r in map_stats])
        rng=np.random.default_rng(19207);boot=np.stack([paired[rng.integers(len(paired),size=len(paired))].mean(0) for _ in range(1500)])
        improvement[reference]=dict(map_mae_wins=(paired<0).sum(0).tolist(),ties=(paired==0).sum(0).tolist(),map_mean_mae_change=paired.mean(0).round(4).tolist(),
                                    map_bootstrap_95pct=np.percentile(boot,[2.5,97.5],0).round(4).tolist())
    report=dict(frozen_model_sha256=old['model_sha256'],target_match=TARGET_MATCH,target_layout=TARGET_LAYOUT,target_steps=steps,
                train_maps=len(training),validation_maps=len(valid),train_team_snapshots=len(yt),validation_team_snapshots=len(yv),
                maps={'training':all_ids[0],'validation':all_ids[1]},sample='Each sixth state, both seats; target all 300 states. Whole-layout split matches original validation; residual uses every fourth sorted original training layout.',
                validation=validation,paired_map_summary=improvement,per_map_mae=map_stats,
                target_190=dict(side_order=['QQ-v6','Artemis'],horizons=[5,10],actual=y[slice_].round(2).tolist(),
                                predicted={k:p[slice_].round(2).tolist() for k,p in tp.items()}),
                target_all=target_metrics,target_190_official_carrier_events=credited_after_190(),feature_latency_microseconds_median_p95=np.percentile(timings,[50,95]).round(2).tolist(),
                binary_frequency='A positive receipt in the horizon. Each model gets its own two-parameter cash-to-frequency calibration learned only on training maps; AUC is threshold free.',
                tests='The generated v7 flow on real and hypothetical turn 190 boards equals standalone output. Per-board cache returns the same ndarray on repeat.')
    args.output.write_text(json.dumps(report,indent=1)+'\n')
    print(validation,improvement,report['target_190'],flush=True)

if __name__=='__main__':main()
