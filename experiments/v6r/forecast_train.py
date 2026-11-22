"""Fit a symmetric residual MLP on game-separated realized changes in cash margin."""
import argparse
import json
import os
from pathlib import Path
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
H=(20,50)


def design(x,raw):
    d=x[:,0]-x[:,1]
    return np.concatenate((d,d*raw[:,0,-4,None],d*(1-raw[:,0,-2,None])),axis=1)


def score(pred, y, mask, horizon):
    errors=np.abs(pred[mask,:,horizon]-y[mask,None,horizon])
    return float(errors.mean())


def fit_ridge(x,y,train,weight,penalty):
    a=x[train];b=y[train];w=weight[train]
    return np.linalg.solve(a.T@(w[:,None]*a)+penalty*np.eye(a.shape[1]),a.T@(w[:,None]*b)).astype("f")


def baselines(z,y,train,val,xraw,time,weights):
    b=z["B"]
    native={"v5_economic":np.repeat(b[:,0,None],2,axis=1),"asset":np.repeat(b[:,1,None],2,axis=1),"v5_territory":b[:,2:4],
            "past24":np.repeat(((xraw[:,0,24]-xraw[:,1,24])*500)[:,None],2,axis=1)}
    out={"no_change":np.zeros((len(y),2,2))}
    # The frozen scorers are long-range valuations. Provide train-only per-horizon temporal rescaling too.
    phase=np.stack((np.ones(len(y)),time/299.,np.maximum(0.,1.-(299-time)/60.)),1)
    all_design=[]
    for name,pred in native.items():
        d=pred[:,:,None]*phase[:,None,:]
        if name!="past24":all_design.append(d)
        out[name+"_raw"]=np.repeat(pred[:,:,None],2,axis=2)
        select=np.repeat(train,2);ww=np.repeat(weights,2)
        coef=fit_ridge(d.reshape(-1,3),np.repeat(y,2,axis=0),select,ww,15000.)
        out[name+"_calibrated"]=d@coef
    ensemble=np.concatenate(all_design,2)
    coef=fit_ridge(ensemble.reshape(-1,ensemble.shape[2]),np.repeat(y,2,axis=0),np.repeat(train,2),np.repeat(weights,2),30000.)
    out["combined_calibrated"]=ensemble@coef
    return out


def train(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(713)
    z=np.load(args.cache);raw=z["X"];y=z["Y"];t=z["T"];g=z["G"];left=z["L"]
    games=json.loads(str(z["games"]));split=np.array([r["split"] for r in games])[g]
    tr,va,te=(split==i for i in range(3))
    counts=np.bincount(g);ww=np.asarray(1/np.sqrt(counts[g]),"f");ww/=ww[tr].mean()
    mean=raw[tr].reshape(-1,raw.shape[2]).mean(0);inv=1/np.maximum(.12,raw[tr].reshape(-1,raw.shape[2]).std(0))
    x=np.clip((raw-mean)*inv,-6,6).astype("f")
    d=design(x,raw)
    forecasts=baselines(z,y,tr,va,raw,t,ww)
    def selection(pred):
        return np.mean([score(pred,y,va&(left>=h),i)/(200 if i==0 else 350) for i,h in enumerate(H)])
    ridge_list=[]
    for penalty in (20,80,250,700):
        coef=fit_ridge(d,y/250,tr,ww,penalty)
        pr=np.repeat((d@coef*250)[:,None,:],2,axis=1)
        ridge_list.append((selection(pr),penalty,coef,pr))
    _,penalty,ridge,pr=min(ridge_list,key=lambda row:row[0]);forecasts["same_features_ridge"]=pr
    base=torch.from_numpy(pr[:,0,:].astype("f")/250)
    xx=torch.from_numpy(x);target=torch.from_numpy(y/250);w=torch.from_numpy(ww)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[2],40),torch.nn.ReLU(),torch.nn.Dropout(.13),
                              torch.nn.Linear(40,20),torch.nn.ReLU(),torch.nn.Linear(20,2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.03)
    best=(float("inf"),);rng=np.random.default_rng(713);indices=np.flatnonzero(tr)
    for epoch in range(90):
        model.train()
        for batch in np.array_split(rng.permutation(indices),max(1,len(indices)//512)):
            net=model(xx[batch]);pred=net[:,0]-net[:,1]+base[batch]
            loss=torch.nn.functional.smooth_l1_loss(pred,target[batch],reduction="none",beta=.75)
            loss=((loss[:,0]+.68*loss[:,1])*w[batch]).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
        model.eval()
        with torch.no_grad():
            nv=model(xx[va]);pval=(nv[:,0]-nv[:,1]).numpy()*250
        # Residual shrinkage is selected on whole calibration maps together with the epoch.
        for mix in (.25,.5,.8,1.):
            candidates=pr[va,0]+mix*pval
            metric=np.mean([np.abs(candidates[left[va]>=h,i]-y[va][left[va]>=h,i]).mean()/scale for i,(h,scale) in enumerate(zip(H,(200,350)))])
            if metric<best[0]-1e-7:best=(metric,epoch,mix,[p.detach().clone() for p in model.parameters()])
        if epoch>best[1]+16:break
    with torch.no_grad():
        for p,v in zip(model.parameters(),best[3]):p.copy_(v)
        model.eval();n=model(xx);f=base+(n[:,0]-n[:,1])*best[2]
    forecasts["forecast"]=np.repeat((f.numpy()*250)[:,None,:],2,axis=1)
    w1,b1,w2,b2,w3,b3=[v.numpy().T.copy() if v.ndim==2 else v.numpy().copy() for v in best[3]]
    weights=dict(mean=mean,inv=inv,ridge=ridge,w1=w1,b1=b1,w2=w2,b2=b2,w3=w3*best[2],b3=b3*best[2])
    np.savez_compressed(args.model,**weights)
    def metrics(mask):
        obj={}
        for j,h in enumerate(H):
            m=mask&(left>=h)
            if not m.any():continue
            methods={}
            for name,pred in forecasts.items():
                err=pred[m,:,j]-y[m,None,j]
                methods[name]=dict(mae=round(float(np.abs(err).mean()),3),rmse=round(float(np.sqrt((err**2).mean())),3))
            obj[str(h)]=dict(boards=int(m.sum()),games=int(len(np.unique(g[m]))),methods=methods)
        return obj
    valmetrics=metrics(va)
    old=["no_change","v5_economic_raw","v5_territory_raw","asset_raw","v5_economic_calibrated","v5_territory_calibrated","asset_calibrated","combined_calibrated","past24_calibrated"]
    paired={}
    for j,h in enumerate(H):
        methods=valmetrics[str(h)]["methods"]
        winner=min(old,key=lambda k:methods[k]["mae"])
        rows=[]
        for i,game in enumerate(games):
            mask=te&(g==i)&(left>=h)
            if not mask.any():continue
            er=np.abs(forecasts["forecast"][mask,:,j]-y[mask,None,j]).mean()
            olderr=np.abs(forecasts[winner][mask,:,j]-y[mask,None,j]).mean()
            rows.append([game["file"],round(float(er),4),round(float(olderr),4)])
        ds=[a-b for _,a,b in rows];bootstrap=np.random.default_rng(713).choice(ds,size=(5000,len(ds)),replace=True).mean(1)
        paired[str(h)]=dict(comparator=winner,games_improved=sum(v<0 for v in ds),games=len(ds),
                            macro_mae_difference=round(float(np.mean(ds)),3),paired_game_bootstrap_95=np.quantile(bootstrap,[.025,.975]).round(3).tolist(),rows=rows)
    report=dict(target="seat 0 cash margin at min(t+h, game end) minus margin at t; headline metrics require the full h future turns. Both observers of the v5 territory baseline receive only their own tracked belief. Other models use public players and tree metadata.",
            splits={k:dict(games=sum(r["split"]==i for r in games),boards=int((split==i).sum())) for i,k in enumerate(("train","calibration","test"))},
            ridge_penalty=penalty,mlp_epoch=best[1]+1,residual_weight=best[2],
            calibration=valmetrics,test=metrics(te),test_phases={str(lo)+"-"+str(hi):metrics(te&(t>=lo)&(t<=hi)) for lo,hi in ((0,74),(75,149),(150,224),(225,298))},
            paired=paired,games=games)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(report["splits"],{h:{k:v["mae"] for k,v in data["methods"].items()} for h,data in report["test"].items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("cache",type=Path);p.add_argument("--model",type=Path,default=HERE/"forecast-model.npz")
    p.add_argument("--report",type=Path,default=HERE/"forecast-validation.json")
    train(p.parse_args())
