"""Learn five-step tree capture risk with a held-out layout split; outcome labels never enter policy."""
import gzip,hashlib,json,os
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.optimize import minimize
from preempt_tree import HERE,MODEL,_v7_pt_profile
ROOT=HERE.parents[1];OUT=ROOT/'analysis/v7'
digest=lambda s:hashlib.sha256(s.encode()).hexdigest()


def scan(frames,idx):
    out=[]
    obs=[f[0]['observation'] for f in frames]
    for t in range(25,len(obs)-20,4):
        for s in (0,1):
            pl=obs[t]['players'];old=obs[t-1]['players'][1-s][2]
            for tid,q,age,rate,thief,where,d,closing,x in _v7_pt_profile(dict(obs[t],step=t),old,s):
                first=next((n for n in range(1,6) if tid in obs[t+n]['players'][1-s][3]),0)
                out.append((x,int(bool(first)),int(first>=3),t,idx))
    return out


def score(y,p):
    p=p.clip(.00001,.99999);return dict(observations=len(y),positives=int(y.sum()),logloss=round(float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),5),
        brier=round(float(np.mean((p-y)**2)),5),top_quartile_capture_rate=round(float(np.mean(y[p>=np.percentile(p,75)])),4))


def main():
    rows=json.load(open(ROOT.parent/'v7-data/receipts-cache/index.json'))['games']
    rows=sorted((r for r in rows if r['end']>=240 and r['file'].split('-')[0] not in ('03100a95e8647c51a4d642fe08d856dd','4063a34afedb6d4767a9118e1dee9e5d','cf276b32e4fd38f43497e60f271a3854')),key=lambda r:digest('preempt-tree-map-1/'+r['layout']))
    train=rows[:120];hold=rows[120:162];assert not {r['layout'] for r in train}&{r['layout'] for r in hold}
    allr=[]
    for i,r in enumerate(train+hold):
        with gzip.open(ROOT/r['source']/'replays'/r['file']) as f:g=json.load(f)
        allr+=scan(g['steps'],i)
    X=np.array([r[0] for r in allr]);y=np.array([r[1] for r in allr]);late=np.array([r[2] for r in allr]);groups=np.array([r[4] for r in allr]);split=groups<len(train)
    def fit(A,Y):
        def loss(w):
            z=A@w;pr=1/(1+np.exp(-np.clip(z,-35,35)))
            return np.logaddexp(0,z).sum()-Y@z+1.5*w[1:]@w[1:], A.T@(pr-Y)+np.r_[0,3*w[1:]]
        return minimize(loss,np.zeros(A.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':160}).x
    w=fit(X[split],y[split]);dist=fit(X[split,:4],y[split])
    pred=1/(1+np.exp(-X@w));d=1/(1+np.exp(-X[:,:4]@dist));test=~split
    approach=test&(X[:,4]>0)&(X[:,1]==0)
    artifact=dict(coef=w.tolist(),training_maps=[r['layout'] for r in train],test_maps=[r['layout'] for r in hold],
       data_origin='public receipts manifest; deterministic hash split; excludes screenshot games',test=score(y[test],pred[test]),distance_only=score(y[test],d[test]),
       approaching_from_2_to_4=dict(model=score(y[approach],pred[approach]),distance=score(y[approach],d[approach]),actual_transfers_at_3_5=int(late[approach].sum())))
    MODEL.write_text(json.dumps(artifact,separators=(',',':'))+'\n')
    print('fit',len(train),len(hold),artifact['test'],artifact['approaching_from_2_to_4'],flush=True)


if __name__=='__main__':main()
