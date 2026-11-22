"""Train and compare terminal win probabilities with full-map partitioning and phase calibration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
for name in ("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ[name]="1"
import numpy as np
import torch
from forecast_win import win_pieces,win_sigmoid

HERE=Path(__file__).resolve().parent


def load(cache,labels):
    z=np.load(cache);l=np.load(labels);games=json.loads(str(z["games"]))
    if str(l["games_hash"])!=hashlib.sha256(json.dumps(games,sort_keys=True).encode()).hexdigest():raise ValueError("outcome map index changed")
    if not np.all((l["target"]>=0)&(l["target"]<=1)):raise ValueError("bad target")
    t=z["T"].astype("f");y=l["target"];cash=l["cash"];group=z["G"];raw=z["X"];b=z["B"]
    duplicate=lambda a:np.repeat(a[:,None],2,axis=1)
    scores=dict(cash=duplicate(cash),cash_asset=duplicate(cash+b[:,1]),cash_economic=duplicate(cash+b[:,0]),cash_territory=duplicate(cash)+b[:,2:4])
    scalars={k:win_pieces(v,t) for k,v in scores.items()}
    phase_risk=160.+1.6*(299-t);cash_signal=duplicate(cash/phase_risk);land_signal=b[:,2:4]/320.
    inputs=np.stack([np.concatenate((np.repeat(raw[:,s,None,:],2,axis=1),
                     (((1-2*s)*cash_signal)[:,:,None]).clip(-8,8),(((1-2*s)*land_signal)[:,:,None]).clip(-8,8)),axis=2) for s in (0,1)],axis=2).astype("f")
    split=np.array([g["split"] for g in games])[group]
    return dict(inputs=inputs,scalars=scalars,t=t,y=y,cash=cash,group=group,games=games,split=split)


def logloss(p,y):
    p=np.clip(p,.0001,.9999)
    return -(y*np.log(p)+(1-y)*np.log(1-p))


def brier(p,y):
    return (p-y)**2


def score(p,y,mask):
    return float(logloss(p[mask],y[mask,None]).mean())


def fit_scalar(feat,y,w,tr,penalty):
    x=feat[tr].reshape(-1,12).astype("d")
    out=np.repeat(y[tr],2).astype("d");weight=np.repeat(w[tr],2).astype("d")
    # Slopes on each absolute-score interval are nonnegative; opposite perspectives are exact complements.
    v=torch.nn.Parameter(torch.zeros(12,dtype=torch.float64))
    opt=torch.optim.LBFGS([v],max_iter=65,line_search_fn="strong_wolfe")
    xx=torch.from_numpy(x);yy=torch.from_numpy(out);ww=torch.from_numpy(weight)
    def closure():
        opt.zero_grad();beta=torch.nn.functional.softplus(v)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(xx@beta,yy,weight=ww,reduction="sum")
              + penalty*(beta**2).sum())/len(yy)
        loss.backward();return loss
    opt.step(closure)
    return torch.nn.functional.softplus(v).detach().numpy().astype("f")


def fit(data):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(742)
    y,t,g,split=(data[k] for k in ("y","t","group","split"));tr=split==0;va=split==1
    w=(1./np.bincount(g)[g]).astype("f");w/=w[tr].mean()
    result={};baselines={}
    for key,feat in data["scalars"].items():
        candidates=[]
        for lam in (2.,10.,35.):
            coef=fit_scalar(feat,y,w,tr,lam)
            p=win_sigmoid(feat@coef)
            candidates.append((score(p,y,va),lam,coef))
        _,lam,coef=min(candidates,key=lambda a:a[0])
        result["scalar_"+key]=coef
        baselines[key]=dict(penalty=lam)
    anchor=data["scalars"]["cash_territory"]@result["scalar_cash_territory"]
    inp=data["inputs"]
    mean=inp[tr].reshape(-1,inp.shape[-1]).mean(0);inv=1/np.maximum(.18,inp[tr].reshape(-1,inp.shape[-1]).std(0))
    x=torch.from_numpy(np.clip((inp-mean)*inv,-5,5))
    yy=torch.from_numpy(y);base=torch.from_numpy(anchor.astype("f"));ww=torch.from_numpy(w)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[-1],24),torch.nn.ReLU(),torch.nn.Dropout(.2),
                              torch.nn.Linear(24,12),torch.nn.ReLU(),torch.nn.Linear(12,1))
    opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.08)
    indices=np.flatnonzero(tr);rng=np.random.default_rng(742);best=(np.inf,)
    late=(t>=199);close=(np.abs(data["cash"])<=200)&(t>=75)
    masks=[va,va&late,va&close,va&late&close]
    for epoch in range(45):
        model.train()
        for idx in np.array_split(rng.permutation(indices),max(1,len(indices)//384)):
            out=model(x[idx]).squeeze(-1);logits=base[idx]+out[:,:,0]-out[:,:,1]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yy[idx,None].expand_as(logits),reduction="none")
            loss=(loss.mean(1)*ww[idx]).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.5);opt.step()
        model.eval()
        with torch.no_grad():
            h=model(x[va]).squeeze(-1);res=(h[:,:,0]-h[:,:,1]).numpy()
        for shrink in (0.,.15,.35,.6,1.):
            p=win_sigmoid(anchor[va]+shrink*res)
            # Set the selection rule before looking at either holdout.
            metric=sum(k*score(p,y[va],m[va]) for k,m in zip((.55,.2,.15,.1),masks))
            if metric<best[0]-1e-7:
                best=(metric,epoch,shrink,[p.detach().clone() for p in model.parameters()])
        if epoch>best[1]+10:break
    w1,b1,w2,b2,w3,b3=[p.numpy().T.copy() if p.ndim==2 else p.numpy().copy() for p in best[3]]
    result.update(mean=mean,inv=inv,w1=w1,b1=b1,w2=w2,b2=b2,w3=w3*best[2],b3=b3*best[2])
    return result,dict(epoch=best[1]+1,shrink=best[2],scalar_phase_calibrations=baselines)


def predictions(data,weights):
    result={k:win_sigmoid(x@weights["scalar_"+k]) for k,x in data["scalars"].items()}
    x=np.clip((data["inputs"]-weights["mean"])*weights["inv"],-5,5)
    h=np.maximum(x@weights["w1"]+weights["b1"],0)
    h=np.maximum(h@weights["w2"]+weights["b2"],0)
    out=(h@weights["w3"]+weights["b3"]).squeeze(-1)
    anchor=data["scalars"]["cash_territory"]@weights["scalar_cash_territory"]
    result["global"]=win_sigmoid(anchor+out[:,:,0]-out[:,:,1])
    result["constant"]=np.full_like(anchor,.5)
    return result


def evaluate(data,weights,partition,best_old):
    pred=predictions(data,weights);t=data['t'];cash=data['cash'];y=data['y'];g=data['group'];split=data['split']
    selected=split==partition;close=(np.abs(cash)<=200)&(t>=75);late=t>=199
    masks=dict(all=selected,late100=selected&late,close200=selected&close,late100_close200=selected&late&close)
    for i,(lo,hi) in enumerate(((0,74),(75,149),(150,224),(225,298))):masks[f"{lo}-{hi}"]=selected&(t>=lo)&(t<=hi)
    report={}
    for label,mask in masks.items():
        metrics={}
        for key,p in pred.items():
            pp=p[mask];yy=y[mask,None]
            ece=0
            confidence=np.maximum(pp,1-pp)
            correct=np.where(pp>=.5,yy,1-yy)
            for lo,hi in ((.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.01)):
                bin=(confidence>=lo)&(confidence<hi)
                if bin.any():ece+=bin.mean()*abs(confidence[bin].mean()-correct[bin].mean())
            metrics[key]=dict(log_loss=round(float(logloss(pp,yy).mean()),5),brier=round(float(brier(pp,yy).mean()),5),ece5=round(float(ece),5))
        report[label]=dict(boards=int(mask.sum()),games=len(np.unique(g[mask])),metrics=metrics)
        deltas=[];rows=[]
        for i in np.unique(g[mask]):
            m=mask&(g==i);f=pred["global"][m];o=pred[best_old][m];yy=y[m,None]
            a=float((logloss(f,yy)-logloss(o,yy)).mean());b=float((brier(f,yy)-brier(o,yy)).mean())
            deltas.append((a,b));rows.append([data["games"][i]["file"],int(m.sum()),round(a,5),round(b,5)])
        d=np.array(deltas);bootstrap=d[np.random.default_rng(742).integers(len(d),size=(4000,len(d)))].mean(1)
        report[label]["paired"]=dict(comparator=best_old,delta_logloss_brier=d.mean(0).round(5).tolist(),ci95=np.quantile(bootstrap,[.025,.975],axis=0).round(5).tolist(),games_logloss_improved=int((d[:,0]<0).sum()))
        if label in ("all","late100_close200"):report[label]["paired"]["games"]=rows
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("fit","external"));p.add_argument("cache",type=Path);p.add_argument("labels",type=Path)
    p.add_argument("--weights",type=Path,default=HERE/"forecast-win.npz");p.add_argument("--meta",type=Path,default=HERE/"forecast-win-calibration.json");p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();d=load(a.cache,a.labels)
    if a.mode=="fit":
        w,meta=fit(d);np.savez_compressed(a.weights,**w);pp=predictions(d,w);va=d['split']==1
        old=min(d['scalars'],key=lambda k:score(pp[k],d["y"],va));meta['best_single_on_calibration']=old
        a.meta.write_text(json.dumps(meta,indent=2)+"\n")
    else:w=np.load(a.weights);old=json.loads(a.meta.read_text())['best_single_on_calibration']
    res=evaluate(d,w,2,old);a.out.write_text(json.dumps(res,indent=2)+"\n")
    print(old,{k:(v['boards'],v['metrics']['global'],v['metrics'][old]) for k,v in res.items() if k in ('all','late100','close200','late100_close200')})
