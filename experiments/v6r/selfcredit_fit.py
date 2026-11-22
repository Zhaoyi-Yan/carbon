"""Grouped held-out return-prediction and old elite policy checks for the self-play credit branch."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from scipy.stats import rankdata
import selfcredit as sc
from localhead import PLANES as HIDDEN_PLANES

def fit(x,y):
    mu=x.mean(0);s=np.maximum(.07,x.std(0));x=(x-mu)/s
    def obj(w):
        l=x@w[:-1]+w[-1];q=expit(l)
        return np.mean(np.logaddexp(0.,l)-y*l)+.006*(w[:-1]@w[:-1]),np.r_[x.T@(q-y)/len(y)+.012*w[:-1],(q-y).mean()]
    b=minimize(obj,np.zeros(x.shape[-1]+1),jac=True,method='L-BFGS-B').x
    weights=b[:-1]/s
    return weights,b[-1]-mu@weights

def model_metrics(q,y):
    rank=rankdata(q);n=len(y);n1=int(y.sum());n0=n-n1
    auc=(rank[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else None
    return dict(n=n,paid=n1,auc=auc,nll=float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))),brier=float(np.mean((q-y)**2)))

def action_metrics(q,y):
    z=q-logsumexp(q,axis=1,keepdims=True)
    return dict(n=len(y),nll=float(-np.mean(z[np.arange(len(y)),y])),accuracy=float(np.mean(z.argmax(1)==y)))

def teacher(cache,w,b):
    cols=[HIDDEN_PLANES.index(p) for p in sc.PLANES];mean=[]
    for rot in range(4):
        with np.load(cache/f'test{rot}.npz') as z:
            keep=z['role']==0
            mean.append(z['z'][keep].astype(float)[:,z['acts']])
            if rot==0:
                v=z['c'][keep][:,:,cols].astype(np.float32);y=z['y'][keep];gid=z['game'][keep]
    x,cargo=sc._selfcredit_features(v);act=np.mean(mean,axis=0)
    q=expit(x@w+b);diff=.42*(q-q.mean(axis=1,keepdims=True));active=cargo>=35
    new=act.copy();new[active]+=diff[active]
    return {k:{'base':action_metrics(act[m],y[m]),'residual':action_metrics(new[m],y[m]),'changed':int(np.sum(act[m].argmax(1)!=new[m].argmax(1)))}
            for k,m in [('all_collectors',np.ones(len(cargo),bool)),('charged',active)]}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('teacher_cache',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    sc._SC_PLANES=sc.PLANES
    with np.load(a.data) as z:d=dict(z)
    meta=json.loads(str(d['meta']))
    groups=[];hold=[]
    for name,v in meta.items():
        pairs={}
        for r in v['games']:pairs.setdefault(r['layout'],[]).append(r)
        assert len(pairs)==8 and all(len(a)==2 and {g['candidate_seat'] for g in a}=={0,1} for a in pairs.values())
        held=sorted(pairs)[::4]
        for h,entries in pairs.items():
            assert len({g['seed'] for g in entries})==1
            seed=entries[0]['seed'];groups.append(dict(panel=name,layout=h,seed=seed,split='test' if h in held else 'train'))
            if h in held:hold.append(seed)
    assert len({g['seed'] for g in groups})==16
    valid=np.isin(d['seed'],hold);candidate=d['candidate']==d['seat']
    acted=d['features'][np.arange(len(valid)),d['action']].astype(float);y=d['paid']
    tr=~valid & candidate
    # State-only comparison includes identical operating phase, current cargo/home distance/current danger.
    w0,b0=fit(acted[tr,:12],y[tr]);w,b=fit(acted[tr],y[tr])
    q0=expit(acted[:,:12]@w0+b0);q=expit(acted@w+b)
    parts={}
    for name,mask in [('held_candidate',valid&candidate),('held_v5',valid&~candidate),('held_both',valid)]+[(str(s),valid&(d['seed']==s)) for s in hold]:
        parts[name]={'state_only':model_metrics(q0[mask],y[mask]),'state_and_played_action':model_metrics(q[mask],y[mask]),'carrier_deaths_before_return':int(d['dies'][mask].sum())}
    report=dict(maps=groups,source=meta,training_rows=int(tr.sum()),training_paid=int(y[tr].sum()),horizon=16,minimum_cargo=35.,
                label='The same collector actually banked at least half its starting cargo in the following 16 turns before death. Referee bank transfers were attributed by worker ID; all recorded balances verified.',
                split='Two of eight layout hashes from each panel, sorted then every fourth; both starting seats kept together. Only the new candidate side is fitted.',
                validation=parts,teacher=teacher(a.teacher_cache,w,b),model={'weights':w.tolist(),'intercept':float(b)},
                interpretation='Predictive held-out credit at recorded actions; alternative actions have no observed future and any policy benefit requires paired games.')
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'training_rows':report['training_rows'],'held_v5':parts['held_v5'],'held_candidate':parts['held_candidate'],'teacher':report['teacher']}))

if __name__=='__main__':main()
