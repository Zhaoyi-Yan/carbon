"""Run one-CPU whole-map verification of 500/700+ incoming receipts, using viewer fog throughout."""
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
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
import receipt_tail as tail
from receipt_check import predict_env,real_target
from receipt_data import ROOT,TARGET_LAYOUT,TARGET_MATCH,layout,targets,impl,LAND
from surrogate import _engine
HERE=Path(__file__).resolve().parent
tail._V7_RECEIPT_DIST=impl._V7_RECEIPT_DIST
tail._V7_TAIL_L=LAND


def collect(g,viewer,fn,every):
    engine=_engine();tr=engine.BeliefTracker(engine._Bel,viewer)
    frames=g['steps'];inc,_=targets(g);cs=np.vstack((np.zeros(2),inc.cumsum(0)))
    features=[];base=[];ys=[];time_ms=[];ts=[]
    for t,frame in enumerate(frames[:-10]):
        obs=frame[0]['observation'];pl=obs['players'];stock,_=tr.step(pl,t,frame[viewer]['observation']['carbon'])
        if t%every==0:
            prev=frames[t-1][0]['observation']['players'] if t else None
            b=fn(pl,t,g['configuration']['episodeSteps']-2)
            at=time.perf_counter_ns()
            x=tail._v7_tail_features(pl,stock,prev,b,t,g['configuration']['episodeSteps']-2)
            time_ms.append((time.perf_counter_ns()-at)*1.e-6)
            features.append(x);base.append(b);ys.append(cs[t+10]-cs[t]);ts.append(t)
        tr.set_actions(pl,frames[t+1][viewer]['action'] or {})
    return np.concatenate(features),np.concatenate(base),np.concatenate(ys),ts,time_ms


def fit(x,base,y):
    mean=x.mean(0);scale=x.std(0).clip(.04,None);z=(x-mean)/scale
    logits=[];bias=[]
    for limit in (500,700):
        clf=LogisticRegression(C=.15,max_iter=80,tol=1.e-6).fit(z,y>=limit)
        logits.append(clf.coef_[0]);bias.append(clf.intercept_[0])
    ridge=Ridge(alpha=len(x)*.025).fit(z,y-base[:,1])
    excess=Ridge(alpha=len(x)*.025).fit(z,np.maximum(0.,y-500.))
    return dict(mean=mean,scale=scale,logits=np.array(logits),bias=np.array(bias),
                cash=ridge.coef_,cash_b=ridge.intercept_,excess=excess.coef_,excess_b=excess.intercept_)


def scores(y,p,thresholds):
    result=dict(cash_mae=round(float(np.mean(abs(p[:,2]-y))),3),cash_rmse=round(float(np.mean((p[:,2]-y)**2)**.5),3),
                true_excess_mean=round(float(np.maximum(0.,y-500).mean()),3),excess_mae=round(float(abs(p[:,3]-np.maximum(0.,y-500)).mean()),3))
    for i,threshold in enumerate((500,700)):
        gold=y>=threshold;flag=p[:,i]>=thresholds[i]
        div=lambda a,b:round(float(a/max(1,b)),5)
        result[str(threshold)]=dict(events=int(gold.sum()),base_rate=div(gold.sum(),len(y)),mean_probability=round(float(p[:,i].mean()),5),
          auc=round(float(roc_auc_score(gold,p[:,i])),5),average_precision=round(float(average_precision_score(gold,p[:,i])),5),
          brier=round(float(np.mean((p[:,i]-gold)**2)),5),fpr=div((flag&~gold).sum(),(~gold).sum()),recall=div((flag&gold).sum(),gold.sum()),
          precision=div((flag&gold).sum(),flag.sum()),false_alarms=int((flag&~gold).sum()),threshold_locked_on_train=round(float(thresholds[i]),6))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    a=parser.parse_args();start=time.monotonic();engine=_engine();engine._torch.set_num_threads(1)
    idx=json.loads((a.cache/'index.json').read_text())['games'];validation=set(json.loads((ROOT/'analysis/v7/receipt-validation.json').read_text())['validation_layouts'])
    # Independent from the original receipt training. The fixed model is the paired baseline.
    order=lambda r:hashlib.sha256(('tail-v1:'+r['layout']).encode()).digest()
    train=sorted((r for r in idx if r['layout'] not in validation),key=order)[:171]
    valid=sorted((r for r in idx if r['layout'] in validation),key=order)[:60]
    source={}
    with (ROOT/'analysis/v7/receipt-games.jsonl').open() as f:
        for line in f:
            item=json.loads(line);source[item['layout']]=item['canonical']
    assert set(source[r['layout']] for r in train).isdisjoint(source[r['layout']] for r in valid)
    assert not any(r['layout']==TARGET_LAYOUT or TARGET_MATCH in r['file'] for r in train+valid)
    fn=predict_env();batches=[];groups=[];times=[];map_ids=[]
    for group,rows in enumerate((train,valid)):
        items=[]
        for j,row in enumerate(rows):
            g=json.load(gzip.open(Path(row['source'])/'replays'/row['file'],'rt'))
            assert layout(g)==row['layout']
            viewer=int(row['layout'][-2:],16)%2
            x,b,y,_,ms=collect(g,viewer,fn,9);items.append((x,b,y));times.extend(ms)
            if group:groups.extend([j]*len(y))
            map_ids.append(dict(id=row['file'],layout=row['layout'],viewer=viewer,split=['training','validation'][group]))
            if j%40==0:print(group,j,time.monotonic()-start,flush=True)
        batches.append(tuple(np.concatenate(vals) for vals in zip(*items)))
    (xt,bt,yt),(xv,bv,yv)=batches;groups=np.asarray(groups)
    models={};dev_cut={};result={};val_pred={}
    for name,d in [('frozen_calibration',5),('cargo_and_progress',14),('mine_and_return',20)]:
        z=fit(xt[:,:d],bt,yt);models[name]=z
        tr=tail._v7_tail_forward(xt[:,:d],bt,z)
        cuts=[float(np.quantile(tr[yt<lim,i],.95)) for i,lim in enumerate((500,700))]
        out=tail._v7_tail_forward(xv[:,:d],bv,z);val_pred[name]=out
        result[name]=scores(yv,out,cuts);dev_cut[name]=cuts
    result['frozen_unadjusted_cash']={'cash_mae':round(float(abs(bv[:,1]-yv).mean()),3),'cash_rmse':round(float(((bv[:,1]-yv)**2).mean()**.5),3)}
    for name,predicted in val_pred.items():
        for column,limit in enumerate((500,700)):
            alarms=predicted[:,column]>=dev_cut[name][column]
            normal=alerts=checks=false=0
            for j in range(len(valid)):
                for side in (0,1):
                    mask=(groups==j)&(np.arange(len(groups))%2==side)
                    if not mask.any() or np.any(yv[mask]>=limit):continue
                    normal+=1;alerts+=int(alarms[mask].any())
                    checks+=int(mask.sum());false+=int(alarms[mask].sum())
            result[name][str(limit)].update(normal_sampled_game_sides=normal,normal_game_sides_with_alert=alerts,
                                            checks_in_normal_game_sides=checks,false_alerts_on_normal_game_sides=false)
    full=models['mine_and_return'];a.model.write_text(json.dumps({k:np.asarray(v).tolist() for k,v in full.items()},separators=(',',':'))+'\n')
    finished_fit=time.monotonic()-start
    # Only after writing the frozen model, forecast the target from QQ's view.
    g=real_target(a.target);assert layout(g)==TARGET_LAYOUT
    x,b,y,ts,_=collect(g,0,fn,1);q=2*ts.index(190)+1
    report_target={'actual':float(y[q]),'frozen_amount':round(float(b[q,1]),3)}
    for name,d in [('frozen_calibration',5),('cargo_and_progress',14),('mine_and_return',20)]:
        p=tail._v7_tail_forward(x[:,:d],b,models[name])
        report_target[name]=dict(p500=round(float(p[q,0]),4),p700=round(float(p[q,1]),4),cash=round(float(p[q,2]),2),expected_excess=round(float(p[q,3]),2))
    import challenger
    original=challenger.build((ROOT/'yan-v6.py').read_bytes())
    generated=tail.build(original,a.model)
    at=generated.index('_V7_TAIL_MODEL=')
    end_code=generated.index('def _v7_flows(board):',at)
    # Feed identical hashed board inputs. Generated head and independent fitting/inference must agree.
    stub=dict(np=np,_V7_RECEIPT_DIST=impl._V7_RECEIPT_DIST,_V5_REACH=LAND,LAST_TURN=298,
              _ST={},_v7_flows=lambda _: b[q-1:q+1])
    exec(compile(generated[at:end_code],'<v7-tail-export>','exec'),stub)
    reference=tail._v7_tail_forward(x[q-1:q+1],b[q-1:q+1],full)
    assert np.allclose(stub['_v7_tail_forward'](x[q-1:q+1],b[q-1:q+1],stub['_V7_TAIL_MODEL']),reference)
    assert stub['_v7_receipt_tail']({'step':295}) is None
    from receipt_manifest import semantic_layout
    assert semantic_layout(g) not in {source[r['layout']] for r in train+valid}
    # Map-level bootstrap compares pairs; every resample retains each team's complete set of observations.
    boots={};rng=np.random.default_rng(71131)
    for ref in ['frozen_calibration','cargo_and_progress']:
        paired=[]
        for j in range(len(valid)):
            keep=groups==j
            pair=[]
            for i,lim in enumerate((500,700)):
                gold=yv[keep]>=lim
                pair.append(np.mean((val_pred['mine_and_return'][keep,i]-gold)**2-(val_pred[ref][keep,i]-gold)**2))
            paired.append(pair)
        boot=np.asarray(paired)[rng.integers(len(paired),size=(1500,len(paired)))].mean(1)
        boots[ref]=dict(map_brier_diff_95=np.quantile(boot,[.025,.975],axis=0).round(6).tolist(),map_wins=(np.asarray(paired)<0).sum(0).tolist())
    report=dict(training=171,validation=60,training_team_rows=len(yt),validation_team_rows=len(yv),maps=map_ids,
                raw_labels='Actual deposits (including fatal depot collisions), regardless of which team brought the cargo. Two hard cutoffs >=500 or >=700 per side over the next 10 turns. Exclude incomplete horizons.',
                features='Single reproducibly chosen viewer per map, predicts both seats. Chronologically updates own v6r BeliefTracker from public positions, own fog carbon and past own actions only. No full_carbon values in features. Each side can stay to mine or move one cell to mine, then return within ten turns; legal tree cover and enemy intercepts discount the mining estimate.',
                validation_metrics=result,map_bootstrap=boots,training_fpr_target=.05,target_viewer='QQ-v6 seat 0',target_step_190_artemis=report_target,
                extraction_and_fit_seconds=round(finished_fit,1),extra_feature_ms_p50_p95=np.percentile(times,[50,95]).round(4).tolist())
    a.output.write_text(json.dumps(report,indent=1)+'\n')
    print(result,report_target,boots,report['extraction_and_fit_seconds'],flush=True)

if __name__=='__main__':main()
