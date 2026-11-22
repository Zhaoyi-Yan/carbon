"""Fit worker arrival-time distributions and evaluate absolute team receipts on disjoint opening layouts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]


def prepare(rows,cache):
    allv={k:[] for k in ('x','load','d','ey','board','y','left','phase','map')}
    offset=0
    for idx,r in enumerate(rows):
        with np.load(cache/(r['layout']+'.npz'),allow_pickle=False) as d:
            n=len(d['t']);x=d['x']
            if not len(x):continue
            time_to_board={int(t):i for i,t in enumerate(d['t'][::2])}
            board=np.asarray([offset+2*time_to_board[int(t)]+int(s) for t,s in zip(d['wt'],d['ws'])])
            for key in ('x','load','d','ey','y','left'):allv[key].append(d[key])
            allv['board'].append(board);allv['phase'].append(np.where(d['t']<20,0,np.where(d['t']<100,1,2)))
            allv['map'].append(np.full(n,idx))
            offset+=n
    d={k:np.concatenate(v) for k,v in allv.items()}
    d['n']=np.bincount(d['board'],minlength=len(d['y']))
    d['band']=np.select((d['n']==0,d['n']<=2,d['n']<=5),(0,1,2),3)
    return d


def sum_workers(values,d):
    return np.stack([np.bincount(d['board'],weights=values[:,k],minlength=len(d['y'])) for k in range(values.shape[1])],axis=1)


class Arrivals(nn.Module):
    def __init__(self):
        super().__init__()
        self.w0=nn.Linear(24,32);self.w1=nn.Linear(32,21)
        nn.init.uniform_(self.w1.weight,-.06,.06)
        with torch.no_grad():
            self.w1.bias.zero_();self.w1.bias[10]=1.7;self.w1.bias[11:]=-2.8

    def forward(self,x,d):
        o=self.w1(self.w0(x).tanh());reachable=torch.arange(1,11)[None,:]>=d[:,None].clamp_min(1)
        logits=torch.cat((o[:,:10].masked_fill(~reachable,-100.),o[:,10:11]),1)
        return F.log_softmax(logits,1),8.*F.softplus(o[:,11:])*reachable


def packed(d,mu,std):
    return [torch.from_numpy(((d['x']-mu)/std).astype('f')),torch.from_numpy(d['load'].astype('f')),
            torch.from_numpy(d['d'].astype('i')),torch.from_numpy(d['ey'].astype('f'))]


def individual(model,data):
    x,load,dist,ey=data
    results=[]
    with torch.inference_mode():
        for j in range(0,len(x),4096):
            part=slice(j,j+4096)
            logp,additional=model(x[part],dist[part])
            results.append((load[part,None]*logp[:,:10].exp()+additional).cumsum(1).numpy())
    return np.concatenate(results)


def background(d):
    extra=d['y']-sum_workers(d['ey'].cumsum(1),d)
    if extra.min()<-.02:raise ValueError(('Unexpected negative background',extra.min()))
    tail=np.zeros((4,3,10),dtype=np.float32)
    for band in range(4):
        for phase in range(3):
            at=(d['band']==band)&(d['phase']==phase)
            parent=(d['phase']==phase)
            tail[band,phase]=(extra[at].sum(0)+40*extra[parent].mean(0))/(at.sum()+40)
    return np.maximum.accumulate(np.maximum(0.,tail),axis=2)


def fit(train):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(308223)
    rng=np.random.default_rng(308223);mu=train['x'].mean(0,dtype=np.float64).astype('f')
    sd=np.maximum(.15,train['x'].std(0,dtype=np.float64)).astype('f');data=packed(train,mu,sd)
    x,load,dist,ey=data;model=Arrivals();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0015)
    # Allocate an incumbent worker's current cargo across future deposits in chronological order.
    paid=ey.cumsum(1)
    allocated=torch.minimum(paid,load[:,None]);original=torch.diff(allocated,dim=1,prepend=torch.zeros_like(allocated[:,:1]))
    fractions=original/load[:,None].clamp_min(1e-6)
    label=torch.cat((fractions,(1.-fractions.sum(1,keepdim=True)).clamp_min(0.)),1)
    extra=(ey-original).clamp_min(0.)
    weights=(load/65.).sqrt().clamp(max=3.5)
    tail=background(train)
    for ep in range(13):
        if ep==8:
            for p in opt.param_groups:p['lr']=.001
        for step in np.array_split(rng.permutation(len(x)),max(1,len(x)//1536)):
            lp,add=model(x[step],dist[step])
            w=weights[step];ce=-(lp*label[step]).sum(1)
            extra_nll=(add-extra[step]*(add.clamp_min(1e-6).log())).mean()
            loss=(w*ce).sum()/w.sum().clamp_min(1.)+.32*extra_nll
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if ep in (0,5,12):print('epoch',ep+1,flush=True)
    w0=model.w0.weight.detach().numpy().T/sd[:,None]
    out=dict(w0=w0,b0=model.w0.bias.detach().numpy()-mu@w0,w1=model.w1.weight.detach().numpy().T,
             b1=model.w1.bias.detach().numpy(),background=tail)
    return model,mu,sd,out


def buckets(d):
    return np.minimum(10,d['d']),np.digitize(d['load'],[2.,20.,60.,120.,220.])


def calibrated_distance(train):
    dist,load=buckets(train)
    out=np.zeros((11,6,10));ys=train['ey'].cumsum(1)
    for i in range(11):
        for j in range(6):
            sel=(dist==i)&(load==j)
            out[i,j]=ys[sel].sum(0)/max(1,(20.+train['load'][sel]).sum())
    return out


def metric(y,p):
    error=p-y
    return {'mean_actual':round(float(np.mean(y)),3),'mean_predicted':round(float(np.mean(p)),3),
            'mae':round(float(np.mean(abs(error))),3),'rmse':round(float(np.mean(error**2)**.5),3),
            'correlation':round(float(np.corrcoef(y,p)[0,1]),4)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('model',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();index=json.loads((a.cache/'index.json').read_text());rr=index['games']
    # SHA-based split independent of outcomes, policies and seats. Duplicate layouts were removed at extraction.
    train_rows=[r for r in rr if int(r['layout'][:8],16)%5!=0];valid_rows=[r for r in rr if int(r['layout'][:8],16)%5==0]
    assert not ({r['layout'] for r in train_rows}&{r['layout'] for r in valid_rows})
    train=prepare(train_rows,a.cache);valid=prepare(valid_rows,a.cache)
    print('maps',len(train_rows),len(valid_rows),'train workers',len(train['x']),flush=True)
    model,mu,sd,out=fit(train)
    with a.model.open('xb') as f:np.savez_compressed(f,**out)
    cash=individual(model,packed(valid,mu,sd))
    bg=out['background'][valid['band'],valid['phase']]
    cutoff=np.minimum(np.arange(1,11)[None,:],valid['left'][:,None]).clip(1,10)-1
    def restrict(mat):return np.take_along_axis(mat,cutoff,axis=1)
    projected=restrict(sum_workers(cash,valid)+bg)
    h=np.arange(1,11,dtype=float)[None,:]
    raw=valid['load'][:,None]*np.maximum(0.,1.-valid['d'][:,None]/(h+1.))
    crude=restrict(sum_workers(raw,valid))
    cell=calibrated_distance(train)
    trained=cell[buckets(valid)]*(20.+valid['load'][:,None])
    trained=np.where(valid["d"][:,None]<=h,trained,0.)
    cmp=restrict(sum_workers(trained,valid)+bg)
    allmetrics={}
    for k in (5,10):
        y=valid['y'][:,k-1]
        vs={'incoming':projected[:,k-1],'distance_discount':crude[:,k-1],'fitted_distance_load':cmp[:,k-1]}
        allmetrics[str(k)]={key:metric(y,v) for key,v in vs.items()}
        wins={base:0 for base in ('distance_discount','fitted_distance_load')}
        for i,g in enumerate(valid_rows):
            mask=valid['map']==i
            for b in wins:wins[b]+=float(np.mean(abs(y[mask]-vs['incoming'][mask])))<np.mean(abs(y[mask]-vs[b][mask]))
        allmetrics[str(k)]['map_mae_wins']=dict(wins)
    report={'sources':{d:sum(r['source']==d for r in rr) for d in sorted({r['source'] for r in rr})},
            'target_layout':index['target_layout'],'excluded_target':index['excluded_target'],'selected_stride':index['stride'],
            'train_maps':len(train_rows),'validation_maps':len(valid_rows),'train_workers':len(train['x']),
            'validation_team_snapshots':len(valid['y']),'metrics':allmetrics,
            'validation_layouts':[r['layout'] for r in valid_rows],'train_layout_set_sha256':hashlib.sha256(('\n'.join(sorted(r['layout'] for r in train_rows))).encode()).hexdigest(),
            'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest()}
    with a.report.open('x') as f: json.dump(report,f,indent=2,default=int)
    print(allmetrics,flush=True)

if __name__=='__main__':main()
