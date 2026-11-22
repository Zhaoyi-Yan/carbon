"""Calibrate uncertainty about the final outcome as a function of the turns still playable."""
import gzip,json,os
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from receipt_data import ROOT,impl
HERE=Path(__file__).resolve().parent
TURNS=np.array([26,82,142,194,235,265,283,294,298])


def score(o,t):
    remaining=299-t;values=[]
    for cash,bases,units,trees in o['players']:
        val=cash;homes=list(bases.values())
        for p,v,k in units.values():
            d=min((impl._V7_RECEIPT_DIST[p][b] for b in homes),default=16)
            val+=v*(d<=remaining)*.84/(1+.035*d)+min(1.,remaining/18.)*(43 if k=='COLLECTOR' else 39)
        for u,(q,age) in trees.items():
            rate=o['trees'].get(u,(None,0.))[1];val+=min(remaining,max(0,50-age),12)*min(13.,rate)
        if not units and not trees and cash<30 and remaining>1:val-=1200.
        values.append(val)
    return values[0]-values[1]


def metric(scales,rows):
    t=rows[:,0];p=1./(1.+np.exp(-np.clip(rows[:,1]/np.interp(t,TURNS,scales),-15,15)));y=rows[:,2]
    return dict(logloss=float(np.mean(-np.log(np.where(y,p,1.-p)))),brier=float(np.mean((y-p)**2)))


def main():
    arcs=json.loads(str(np.load(ROOT.parent/'v7-data/netcash-rows.npz')['games']));rows=[];maps=[]
    for g in arcs:
        with gzip.open(ROOT/g['source']/'replays'/g['file']) as f:d=json.load(f)
        f=d['steps'];reward=[s['reward'] for s in f[-1]]
        if any(v is None for v in reward) or reward[0]==reward[1]:continue
        maps.append(dict(layout=g['layout'],fold=g['fold']))
        for t in range(2,len(f)-1,4):
            rows.append((t,score(f[t][0]['observation'],t),int(reward[0]>reward[1]),len(maps)-1))
    data=np.array(rows);folds=np.array([r['fold'] for r in maps]);train=folds[data[:,3].astype(int)]=='fit';held=folds[data[:,3].astype(int)]=='test'
    grid=np.geomspace(8.,4800.,100)
    def fit(a):
        x=a[:,1,None]/grid;err=np.logaddexp(0.,x)-(a[:,2,None]*x)
        return float(grid[err.mean(0).argmin()])
    constant=fit(data[train]);bucket=np.abs(data[:,0,None]-TURNS).argmin(1)
    scales=np.minimum.accumulate([fit(data[train&(bucket==i)]) for i in range(len(TURNS))]).round(3)
    metrics={scope:{'phase':metric(scales,data[held&mask]),'constant':metric([constant]*len(TURNS),data[held&mask])} for scope,mask in [('all',data[:,0]>=0),('last60',data[:,0]>=239),('close',np.abs(data[:,1])<300)]}
    report=dict(turns=TURNS.tolist(),scales=scales.tolist(),constant=constant,metrics=metrics,train_maps=int((folds=='fit').sum()),test_maps=int((folds=='test').sum()),target_excluded=True)
    (HERE/'win-risk-model.json').write_text(json.dumps({k:report[k] for k in ('turns','scales')},indent=2)+'\n')
    (ROOT/'analysis/v7/win-risk-calibration.json').write_text(json.dumps(dict(report,layouts=maps),indent=2)+'\n');print(report)
if __name__=='__main__':main()
