"""Cross-fit the flow residual on six development maps and evaluate a single marginal-slot exchange on twelve others."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from antisym_report import ROOT,bootstrap,gathered,ordered

HORIZON=np.array([.35,.65])
GRID=[(th,beta) for th in (15.,30.,50.) for beta in (.25,.5,1.)]
FALLBACK=(30.,.5)
def unpack(raw):
    roots=[]
    for g in raw:
        s=g['side']
        for z in g['rows']:
            p=np.array(z['p']);n=len(p)
            if n<3:continue
            f=np.array([x[:5] for x in z['features']]);actual=np.asarray(z['observed'])
            flow=f[:,:2];true=actual[:,:,s]-actual[:,:,1-s]
            score=(p[:,6]-p[:,10])*3000.
            # Retain the named policy's action and all interior slots.
            cov=np.stack((score,f[:,3],f[:,2]),1);cov-=cov.mean(0)
            future=np.transpose(np.array(z['y'])[:,:,1,:],(1,0,2))
            target=np.column_stack((flow@HORIZON,true[:,0]@HORIZON,true[:,1]@HORIZON))
            roots.append(dict(map=g['layout'],source=g['name'],fold=g['split'],side=s,t=z['t'],score=score,cash=f[:,2],
                             cov=cov,target=target,flow=flow,actual=true,rewards=future,old=ordered(score,n)))
    return roots

def fit(roots):
    sizes=Counter(r['map'] for r in roots);xs=[];ys=[]
    for r in roots:
        n=len(r['score']);w=np.sqrt(1./(sizes[r['map']]*n))
        xs.append(r['cov']*w);ys.append((r['target']-r['target'].mean(0))*w)
    x=np.concatenate(xs);y=np.concatenate(ys)
    b,_,rank,singular=np.linalg.lstsq(x,y,rcond=1e-6)
    return b,dict(rank=int(rank),singular=singular.round(3).tolist(),dot=np.max(np.abs(x.T@(y-x@b))).item())

def choose(r,coef,params,method):
    threshold,beta=params;score=r['score'];old=r['old'];last=old[-1]
    index={'forecast':0,'agree_raw':0,'agree_cash':0,'raw':0,'oracle_literal':1,'oracle_waypoint':2}[method]
    q=r['target'][:,index]
    if method!='raw':q=q-r['cov']@coef[:,index]
    excluded=[i for i in range(len(score)) if i not in old]
    allowed=[i for i in excluded if q[i]-q[last]>=threshold]
    if method in ('agree_raw','agree_cash'):
        absolute=r['flow'] if method=='agree_raw' else r['flow']+r['cash'][:,None]
        allowed=[i for i in allowed if np.all(absolute[i]-absolute[last]>=threshold)]
    if not allowed:return old,last,last
    i=max(allowed,key=lambda c:score[c]+beta*q[c])
    if score[i]-score[last]+beta*(q[i]-q[last])<=1e-10:return old,last,last
    return old[:-1]+[i],last,i

def eval_rows(roots,model,params,method):
    rows=[];changed=[]
    for r in roots:
        coefficient=model[r['map']] if isinstance(model,dict) else model
        new,old,i=choose(r,coefficient,params,method);y=r['rewards'];kept=r['old'];flag=(i!=old)
        best=y[new].max(0);prior=y[kept].max(0);oracle=y.max(0)
        forecast=(r['flow'][i]-r['flow'][old])@HORIZON if flag else 0.
        truth=(r['actual'][i]-r['actual'][old])@HORIZON if flag else np.zeros(2)
        out=np.r_[flag,(best-prior).ravel(),(y[i]-y[old]).ravel(),(oracle-best).ravel(),(oracle-prior).ravel(),
                  (best[:,0]==oracle[:,0]).astype(float), (prior[:,0]==oracle[:,0]).astype(float)]
        rows.append((r['map'],out))
        if flag:changed.append(dict(source=r['source'],map=r['map'],side=r['side'],t=r['t'],out=old,added=i,
                             forecast_delta=float(forecast),actual=truth.round(3).tolist(),
                             pair=(y[i]-y[old]).round(3).tolist(),best=(best-prior).round(3).tolist()))
    return rows,changed

def summary(v,events):
    a,ci=bootstrap(v);n=len(v)
    return dict(roots=n,maps=len({g for g,x in v}),changed=len(events),changed_maps=len({r['map'] for r in events}),
        change_share_map=round(float(a[0]),4),delta_best=a[1:5].reshape(2,2).round(3).tolist(),
        delta_best95=ci[:,1:5].reshape(2,2,2).round(3).tolist(),
        delta_exchange=a[5:9].reshape(2,2).round(3).tolist(),delta_exchange95=ci[:,5:9].reshape(2,2,2).round(3).tolist(),
        available_regret_new=a[9:13].reshape(2,2).round(3).tolist(),available_regret_base=a[13:17].reshape(2,2).round(3).tolist(),
        retains_best_cash_new=a[17:19].round(4).tolist(),retains_best_cash_old=a[19:21].round(4).tolist(),
        predicted_flow=float(np.mean([r['forecast_delta'] for r in events])) if events else None,
        realized_flow=np.mean([r['actual'] for r in events],0).round(3).tolist() if events else None)
def objective(vals,changed):
    groups=defaultdict(list)
    for g,v in vals:groups[g].append(v[[1,3]])
    a=np.stack([np.mean(x,0) for x in groups.values()])
    # Require a minimum amount of independent coverage and improvement under both continuations.
    if len(changed)<6 or len({r['map'] for r in changed})<3:return None
    consensus=np.minimum(a.mean(0),.5*(a.mean(0)+np.median(a,axis=0)))
    return float(consensus.min())
def fit_stats(roots,b):
    r=defaultdict(list)
    for g in roots:
        tc=g['target'][:,0];pred=tc-g['cov']@b[:,0]
        for j,name in enumerate(('literal','waypoint')):
            observed=g['target'][:,j+1];adj=observed-g['cov']@b[:,j+1]
            for raw,z,y in [('raw',tc,observed),('residual',pred,adj)]:
                ix=np.triu_indices(len(z),1);diff=z[ix[0]]-z[ix[1]];truth=y[ix[0]]-y[ix[1]]
                for label,live in [('any',(np.abs(diff)>1.e-5)&(np.abs(truth)>1.e-5)),('strong', (np.abs(diff)>=30)&(np.abs(truth)>1.e-5))]:
                    if live.any():r[name,raw,label].append((g['map'],np.mean((diff[live]*truth[live])>0)))
    return {'/'.join(k):dict(conditional_root_sign=round(float(bootstrap(v)[0]),4),roots=len(v)) for k,v in r.items()}

def main(data):
    src=gathered(data/'cutflow-roots');assert len(src)==36
    roots=unpack(src);train=[r for r in roots if r['fold']=='train'];test=[r for r in roots if r['fold']=='test']
    tr={r['map'] for r in train};te={r['map'] for r in test};assert len(tr)==6 and len(te)==12 and tr.isdisjoint(te)
    cross={g:fit([r for r in train if r['map']!=g])[0] for g in tr}
    grid=[];best=None
    for params in GRID:
        v,e=eval_rows(train,cross,params,'forecast');obj=objective(v,e)
        grid.append(dict(threshold=params[0],beta=params[1],objective=obj,summary=summary(v,e)))
        if obj is not None and obj>0 and (best is None or obj>best[0]):best=(obj,params)
    selected=best[1] if best else FALLBACK;b,diag=fit(train)
    metrics={};changes={}
    for name,pool,fitmodel in [('crossfit_training',train,cross),('test',test,b)]:
        metrics[name]={}
        for method in ('forecast','raw','agree_raw','agree_cash','oracle_literal','oracle_waypoint'):
            v,e=eval_rows(pool,fitmodel,selected,method);metrics[name][method]=summary(v,e);changes[name+'-'+method]=e
    rep=dict(development=sorted(tr),holdout=sorted(te),games=[{k:g[k] for k in ('name','panel','seed','side','version','layout','split')} for g in src],
        horizon_weights=HORIZON.tolist(),candidate_filter='The current policy proposal and every non-marginal survivor are fixed. Among eliminated candidates, consider only a positive residual gain above the cash threshold, then rank by original net leaf score plus beta times the residual. Replace one when revised score is strictly larger. Score and assets are denominated in cash.',
        selection='Grid and fallback fixed before reading the paired receipts. Fit residual projection within each root on the old net leaf score, existing receipt-aware asset score, and immediate cash, weighted equally by layout, root, then candidate. Leave-one-development-layout-out tuning requires >=6 swaps over >=3 maps and positive conservative mean under each continuation. Twelve holdout maps remain untouched by fit/selection; their previous unrelated critic results were reviewed.',
        target_axis='Each 2x2 uses [literal, waypoint] x [cash difference at 35, economic asset measure at 35]. Confidence arrays are [lower,upper] then same axes. Oracle uses the actual subsequent five/ten-step gross deposits only as a non-deployable diagnostic.',
        sensitivity='After inspecting the primary null, agree_raw and agree_cash require both 5 and 10 horizons to exceed the same threshold, optionally including already banked cash. Neither changes grid selection.',selected=list(selected),passed_development=bool(best),eligible_roots=dict(train=len(train),test=len(test)),
        coefficients=b.round(8).tolist(),orthogonality=diag,validation=metrics,grid=grid,pair_direction=fit_stats(test,b),changes=changes)
    (ROOT/'analysis/v7/cutflow-validation.json').write_text(json.dumps(rep,indent=2)+'\n')
    print('accepted',bool(best),'selected',selected,'n',len(train),len(test));print('coef',b[:,0],'direction',rep['pair_direction'])
    for fold,methods in metrics.items():
        for method,m in methods.items():print(fold,method,m['changed'],m['delta_best'],m['delta_best95'],m['realized_flow'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
