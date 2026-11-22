"""Fit 20 lightweight action heads, a situation-dependent selector, and a 48-step cash forecast."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from backbone import load


def ridge(x, y, penalty):
    return np.linalg.solve(x.T @ x + penalty*np.eye(x.shape[1]), x.T @ y)


def fit(cache, weights, report, credit=None):
    torch.set_num_threads(2)
    torch.manual_seed(5701)
    groups=defaultdict(list)
    for path in sorted(cache.glob('*.npz')):
        if '.partial.' in path.name:
            continue
        with np.load(path) as z:
            groups[str(z['layout'])].append((path,int(z['seed']),np.unique(z['T'])))
    train, valid = defaultdict(list),defaultdict(list)
    train_files, valid_files=[],[]
    reserved=set(json.loads((Path(__file__).parent/'design.json').read_text())['qualification'])
    selected={}
    used_seeds=set()
    for layout, rows in sorted(groups.items()):
        path,seed,teachers=rows[0]
        if seed not in reserved and seed not in used_seeds:
            selected[layout]=(path,teachers)
            used_seeds.add(seed)
    split_key=lambda layout: hashlib.sha256(('top20-v5-'+layout).encode()).hexdigest()
    validation={key for key in selected if int(split_key(key)[:8],16)%5==0}
    for teacher in range(20):
        if not any(teacher in selected[k][1] for k in validation):
            options=[k for k in selected if teacher in selected[k][1]]
            validation.add(min(options,key=split_key))
    for layout, (path, teachers) in selected.items():
        hold=layout in validation
        target=valid if hold else train
        (valid_files if hold else train_files).append(path.stem)
        with np.load(path) as z:
            for k in 'HAKTEC YJBP'.replace(' ',''):
                target[k].append(z[k])
        if credit and not hold:
            target['advantage'].append(np.load(credit/(path.stem+'.npy')))
    for part in (train,valid):
        for k in part:
            part[k]=np.concatenate(part[k])
    model=load()
    base_w=np.stack([model._POLICY.p['hw.w'].reshape(5,-1)]*2+
                    [np.pad(model._POLICY.p['hb.w'].reshape(3,-1),((0,2),(0,0)))])
    base_b=np.stack([model._POLICY.p['hw.b']]*2+[np.r_[model._POLICY.p['hb.b'],[-200.,-200.]]]).astype('float32')
    hmean=train['H'].mean(0,dtype=np.float64).astype('float32')
    hstd=np.maximum(train['H'].std(0,dtype=np.float64),.12).astype('float32')
    w=torch.nn.Parameter(torch.zeros(20,3,5,len(hmean)))
    b=torch.nn.Parameter(torch.zeros(20,3,5))
    shared=torch.nn.Parameter(torch.zeros(3,5,len(hmean)))
    intercept=torch.nn.Parameter(torch.zeros(3,5))
    opt=torch.optim.Adam([w,b,shared,intercept],lr=.0025)
    def tensors(part):
        H=part['H'].astype('float32')
        t=torch.from_numpy(part['T'].astype('int64')); k=torch.from_numpy(part['K'].astype('int64'))
        logits=np.empty((len(H),5),np.float32)
        for role in range(3):
            m=part['K']==role; logits[m]=H[m]@base_w[role].T+base_b[role]
        return torch.from_numpy((H-hmean)/hstd),torch.from_numpy(logits),t,k,torch.from_numpy(part['A'].astype('int64'))
    tr=tensors(train); va=tensors(valid)
    advantage_weights=None
    if credit:
        # Center by teacher to avoid reinforcing farmed wins against weak opponents.
        signal=train['advantage'].copy()
        for teacher in range(20):
            mask=train['T']==teacher
            signal[mask]-=np.median(signal[mask])
        rank=np.exp(-train['T'].astype(float)/10.)
        advantage_weights=torch.from_numpy(((.68+.65*rank)*np.exp(np.clip(signal/.43,-1.1,1.1))).astype('float32'))
    def scores(data,ids):
        h,prior,t,k,y=data
        ti,ki=t[ids],k[ids]
        offset=((w[ti,ki]+shared[ki])*h[ids,None,:]).sum(-1)+b[ti,ki]+intercept[ki]
        return prior[ids]+offset,y[ids]
    def metrics(data, baseline=False):
        loss=correct=0.; count=0
        with torch.no_grad():
            for ids in torch.arange(len(data[0])).split(2048):
                zz,yy=scores(data,ids)
                if baseline: zz=data[1][ids]
                loss+=float(F.cross_entropy(zz,yy,reduction='sum'))
                correct+=int((zz.argmax(1)==yy).sum());count+=len(yy)
        return dict(nll=loss/count,accuracy=correct/count)
    best=None; history=[metrics(va,True)]
    for epoch in range(14):
        for ids in torch.randperm(len(tr[0])).split(2048):
            pred,y=scores(tr,ids)
            imitation=F.cross_entropy(pred,y,reduction='none')
            if advantage_weights is not None:
                weights_batch=advantage_weights[ids]
                imitation=(weights_batch*imitation).sum()/weights_batch.sum()
            loss=imitation.mean()+.012*w.square().mean()+.015*shared.square().mean()
            opt.zero_grad();loss.backward();opt.step()
        metric=metrics(va);history.append(metric)
        if best is None or metric['nll'] < best[0]:
            with torch.no_grad():
                coef=(w+shared).detach().numpy().copy()/hstd
                bias=(b+intercept).detach().numpy().copy()-coef@hmean
                best=(metric['nll'],coef,bias,epoch+1)
        print(epoch+1,metric,flush=True)
        if epoch+1-best[3]>=3:break
    E=train['E'].astype(float);C=train['C'].astype(float);Y=train['Y'].astype(float)
    es=np.maximum(np.sqrt((E*E).mean(0)),.08)
    ew=ridge(E/es, Y, 180.)/es
    residual=Y-E@ew
    cs=np.maximum(np.sqrt((C*C).mean(0)),.15)
    cw=np.array([ridge(C[train['J']==j]/cs, residual[train['J']==j], 420.)/cs for j in range(20)])
    proto=np.empty((4,20,8))
    for phase in range(4):
        mask=train['P']==phase
        pop=train['B'][mask].mean(0)
        for j in range(20):
            X=train['B'][mask & (train['J']==j)]
            proto[phase,j]=(X.sum(0)+35*pop)/(len(X)+35)
    scale=np.maximum(train['B'].std(0),.08)
    metrics_cash={'cash_only_mae':float(np.mean(np.abs(valid['Y']))*1000),
                  'forecast_mae':float(np.mean(np.abs(valid['Y']-valid['E']@ew))*1000)}
    counts=lambda part:np.bincount(part['T'],minlength=20).tolist()
    stats=dict(training_matches=train_files,validation_matches=valid_files,train_units=counts(train),
               validation_units=counts(valid),validation_history=history,best_epoch=best[3],
               cash_validation=metrics_cash,train_states=len(train['Y']),validation_states=len(valid['Y']))
    for part,files in ((train,train_files),(valid,valid_files)):
        assert files and min(counts(part))>0, 'missing teacher in a split'
    np.savez_compressed(weights,coef=best[1].astype('float32'),bias=best[2].astype('float32'),
                        economic=ew,route=cw,prototypes=proto,scale=scale)
    report.write_text(json.dumps(stats,indent=2)+'\n')
    print('matches',len(train_files),len(valid_files),'cash',metrics_cash,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('cache','weights','report'):p.add_argument(arg,type=Path)
    p.add_argument('--credit',type=Path)
    a=p.parse_args();fit(a.cache,a.weights,a.report,a.credit)
