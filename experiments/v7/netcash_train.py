"""Fit a map-balanced short-horizon cash-flow model and evaluate temporal ordering on untouched maps."""
import gzip
import json
import os
import time
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np
from netcash_data import DATA,HERE,ROOT,OUT,OLD,digest,MODELPATH,H
from netcash import MODEL

REPORT=ROOT/'analysis/v7/netcash-validation.json'


def mapweights(groups):
    _,inv,n=np.unique(groups,return_inverse=True,return_counts=True)
    w=1./n[inv];return w/w.sum()


def fit(x,y,g,t,rows,reg,local):
    a=x[rows];b=y[rows];groups=g[rows]
    w=mapweights(groups)
    scale=np.sqrt((w[:,None]*a**2).sum(0)).clip(.05,None)
    a=a/scale
    loc=(np.diff(groups)==0)&(np.diff(t[rows])==4)
    v=np.diff(a,axis=0)[loc];z=np.diff(b,axis=0)[loc]
    l=mapweights(groups[1:][loc])
    gram=(a*w[:,None]).T@a+local*(v*l[:,None]).T@v
    cross=(a*w[:,None]).T@b+local*(v*l[:,None]).T@z
    return np.linalg.solve(gram+reg*np.eye(a.shape[1]),cross)/scale[:,None]


def rank(a,b,minimum=0):
    keep=abs(a)>minimum
    count=keep.sum()
    if not count:return None
    score=np.where(abs(b[keep])<1e-7,.5,(a[keep]*b[keep]>0).astype(float))
    return round(float(score.mean()),5)


def score(y,p,g,t,c,rows):
    a=y[rows];b=p[rows];groups=g[rows];money=c[rows]
    d=(np.diff(groups)==0)&(np.diff(t[rows])==4)
    delta=np.diff(a,axis=0)[d];pred=np.diff(b,axis=0)[d]
    future=delta+np.diff(money)[d,None];estimated=pred+np.diff(money)[d,None]
    w=mapweights(groups)
    return dict(boards=len(a),maps=len(np.unique(groups)),mae=np.mean(abs(a-b),axis=0).round(3).tolist(),
                map_balanced_mae=np.sum(abs(a-b)*w[:,None],axis=0).round(3).tolist(),
                adjacent_pairs=int(sum(d)),rank_increment=[rank(delta[:,i],pred[:,i]) for i in (0,1)],
                rank_increment_over20=[rank(delta[:,i],pred[:,i],20.) for i in (0,1)],
                rank_future_cash=[rank(future[:,i],estimated[:,i]) for i in (0,1)])


def pick(x,y,g,t,c,fitmask,tune):
    options=[]
    for reg in (.0003,.0015,.008,.035):
        for local in (0.,.6,2.):
            w=fit(x,y,g,t,fitmask,reg,local)
            s=score(y,x@w,g,t,c,tune)
            # Tune on disjoint layouts: MAE first, penalizing incorrect ordering by
            # up to 20 cash units to avoid selecting a flat, weakly ordering model.
            merit=np.mean(s['map_balanced_mae'])+20*(1.-np.mean(s['rank_increment_over20']))
            options.append((merit,reg,local,w,s))
    return min(options,key=lambda v:v[0])


def main():
    with np.load(OUT) as z:
        x=z['X'].astype(float);y=z['Y'].astype(float);g=z['G'];t=z['T'];c=z['C'];b=z['B']
        games=json.loads(str(z['games']));ex=json.loads(str(z['excluded']));n_reuse=int(z['reused_forecast_rows'])
    folds=np.array([v['fold'] for v in games])[g];tr=folds=='fit';tu=folds=='tune';test=folds=='test'
    names=['incoming10','distance','asset','forecast20','forecast50']
    pred={name:np.repeat(b[:,i,None],2,axis=1) for i,name in enumerate(names[:3])}
    pred['cash_only']=np.zeros_like(y)
    pred['frozen_forecast']=np.stack((b[:,3],.5*b[:,3]+.5*b[:,4]),axis=1)
    # Keep the tuning map block disjoint from model fitting even after selecting the hyperparameters.
    stride=x.shape[1]//4
    _,_,_,noasset,_=pick(x,y,g,t,c,tr,tu)
    pred['without_explicit_asset']=x@noasset
    power=np.stack((np.ones(len(t)),np.maximum(0.,1.-t/90.),t/298.,np.maximum(0.,1.-(299-t)/100.)),axis=1)
    x=np.concatenate((x,b[:,2,None]*power),axis=1)
    merit,reg,local,w,choice=pick(x,y,g,t,c,tr,tu)
    pred['netcash']=x@w
    mask=np.ones(x.shape[1],bool)
    for j in range(4):
        mask[j*stride+np.array([0,1,2,3,17,18,19,20,21])]=False
    _,r2,l2,wo,_=pick(x[:,mask],y,g,t,c,tr,tu)
    pred['same_model_without_receipts']=x[:,mask]@wo
    # Fit an honest scalar/phase calibration of the original simplified asset price.
    phase=t/298.;asst=b[:,2];ax=np.stack([asst,asst*phase,asst*np.maximum(0,1.-t/90.)],axis=1)
    pred['calibrated_asset']=ax@fit(ax,y,g,t,tr,.001,0)
    blended=pred['netcash']@np.asarray([.55,.45])
    native=b[:,2]
    pred['bounded_asset_blend']=np.repeat((native+.45*np.clip(blended-native,-300.,300.))[:,None],2,axis=1)
    def evaluate(rows):
        return {k:score(y,v,g,t,c,rows) for k,v in pred.items()}
    scores=evaluate(test)
    phases={f'{lo}-{hi}':evaluate(test&(t>=lo)&(t<=hi)) for lo,hi in ((0,59),(60,149),(150,229),(230,264))}
    active=np.array([np.ptp(c[g==i])>0 for i in range(len(games))])[g]
    phase_cash=[]
    for feature in ('frozen_forecast','incoming10','same_model_without_receipts','distance','calibrated_asset'):
        comparison=[]
        for j in np.unique(g[test]):
            idx=g==j
            comparison.append((abs(pred['netcash'][idx]-y[idx])-abs(pred[feature][idx]-y[idx])).mean(axis=0))
        m=np.asarray(comparison)
        rng=np.random.default_rng(81871)
        boot=np.asarray([m[rng.integers(0,len(m),len(m))].mean(0) for _ in range(1500)])
        phase_cash.append(dict(comparison=feature,mae_difference=m.mean(0).round(3).tolist(),map_wins=(m<0).sum(0).tolist(),
                               interval95=np.quantile(boot,[.025,.975],axis=0).round(3).tolist()))
    # Describe early cash reversals on held layouts after the fit. Ten-step actuals
    # are diagnostic labels only; they never enter selection, feature building or search.
    cash10=np.zeros(len(x))
    for j in np.unique(g[test]):
        record=games[j];idx=g==j
        with gzip.open(ROOT/record['source']/'replays'/record['file']) as f:frames=json.load(f)['steps']
        money=np.array([o[0]['observation']['players'][0][0]-o[0]['observation']['players'][1][0] for o in frames])
        cash10[idx]=money[t[idx]+10]-money[t[idx]]
    sign=np.sign(b[:,0])
    cohorts={}
    strong=test&(abs(b[:,0])>=100.)
    for label,mask in [('predicted_10step_receipt_advantage_at_least_100',strong),
                       ('also_observed_10step_net_cash_increase',strong&(sign*cash10>0)),
                       ('observed_10step_increase_reversed_by_35',strong&(sign*cash10>0)&(sign*y[:,1]<=0))]:
        u=sign[mask,None];truth=y[mask]*u
        cohorts[label]=dict(boards=int(sum(mask)),maps=int(len(set(g[mask]))),positive_10step_cash=float((cash10[mask]*sign[mask]>0).mean()),
            nonpositive_net_at_horizon=(truth<=0).mean(0).round(4).tolist(),observed_net_mean=truth.mean(0).round(2).tolist(),
            average_10step_actual_net=round(float((sign[mask]*cash10[mask]).mean()),2),
            models={k:dict(predicted_net_mean=(p[mask]*u).mean(0).round(2).tolist(),mae=np.mean(abs(p[mask]-y[mask]),axis=0).round(2).tolist()) for k,p in pred.items()})
    np.savez_compressed(MODEL,coef=w)
    report=dict(target='(cash0[t+h]-cash1[t+h]) minus current cash gap. Exact h=20,35 including all ordinary, tree, hire, seize and collision flows. Excludes truncated horizons.',
                frozen_model_files={str(p.relative_to(ROOT)):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in (MODELPATH,OLD/'forecast-model.npz',OLD/'assets-model.json',MODEL)},
                selected=dict(reg=reg,derivative_loss=local,chosen_on='tune layouts only',selection_score=round(float(merit),4),tune=choice),
                folds={s:int(len({r['layout'] for r in games if r['fold']==s})) for s in ('fit','tune','test')},
                test=scores,early_receipt_cohorts=cohorts,active_maps=evaluate(test&active),phases=phases,paired_map_mae=phase_cash,exclusions=ex,
                reused_legacy_feature_rows=n_reuse,receipt_rows_from_cache=len(x),maps=games,
                limitations='Adjacent comparisons use observations four turns apart within one trajectory, not alternative moves from an identical state. rank_increment excludes ties; predicted ties get half credit. Simplified asset is the published v6r cargo/tree/workforce subscore before hidden-stock sapling valuation. Search correction is capped at 135 cash units before the 0.26 asset mixture; raw forecast scores do not measure wins.')
    REPORT.write_text(json.dumps(report,separators=(',',':'))+'\n')
    print(report['folds'])
    for k,s in scores.items():print(k,s['mae'],s['rank_increment'],s['rank_future_cash'])


if __name__=='__main__':main()
