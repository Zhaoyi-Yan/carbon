"""Train a low-dimensional horizon calibration of the public-state MLP and the frozen legal territory score."""
import argparse
import json
import os
from pathlib import Path
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
NAMES=("mean","inv","ridge","w1","b1","w2","b2","w3","b3")
H=(20,50)


def residual(raw,model):
    mean,inv,ridge,w1,b1,w2,b2,w3,b3=[model[k] for k in NAMES]
    x=np.clip((raw-mean)*inv,-6,6)
    d=x[:,0]-x[:,1]
    base=np.concatenate((d,d*raw[:,0,-4,None],d*(1-raw[:,0,-2,None])),axis=1)@ridge
    h=np.maximum(x@w1+b1,0);h=np.maximum(h@w2+b2,0);v=h@w3+b3
    return 250*(base+v[:,0]-v[:,1])


def frame_data(z,model):
    old=z["B"]
    public=residual(z["X"],model)
    canonical=lambda a:np.repeat(a[:,None],2,axis=1)
    features=np.stack((canonical(public[:,0]),canonical(public[:,1]),old[:,2:4]),axis=-1)
    combined=np.stack((canonical(old[:,0]),canonical(old[:,1]),old[:,2:4]),axis=-1)
    basis=np.stack((np.ones(len(public)),z["T"]/299.,np.maximum(0,1-(299-z["T"])/60.)),axis=1)
    expanded=lambda a:np.concatenate([a*basis[:,j,None,None] for j in range(3)],axis=2)/250
    return public,expanded(features),expanded(combined),expanded(old[:,2:4,None])


def coeff(x,y,w,mask,lam):
    a=x[mask].reshape(-1,x.shape[-1]);b=np.repeat(y[mask]/250,2,axis=0);w=np.repeat(w[mask],2)
    return np.linalg.solve(a.T@(w[:,None]*a)+np.eye(a.shape[1])*lam,a.T@(w[:,None]*b)).astype("f")


def mask_for(z,i):
    return np.array([r["split"] for r in json.loads(str(z["games"]))])[z["G"]]==i


def fit(z,model):
    _,ab,old,terr=frame_data(z,model);tr=mask_for(z,0);va=mask_for(z,1);y=z["Y"];n=z["L"]
    w=np.asarray(1/np.sqrt(np.bincount(z["G"])[z["G"]]),"f");w/=w[tr].mean()
    result={}
    for k,x in (("blend",ab),("legacy_combined",old),("legacy_territory",terr)):
        trials=[]
        for lam in (2.,16.,64.,200.):
            weights=coeff(x,y,w,tr,lam);pred=(x@weights)*250
            errors=[np.abs(pred[va&(n>=h),:,i]-y[va&(n>=h),i,None]).mean()/s for i,(h,s) in enumerate(zip(H,(200.,350.)))]
            trials.append((np.mean(errors),lam,weights))
        _,penalty,weights=min(trials,key=lambda row:row[0])
        result[k]=dict(penalty=penalty,coef=weights.tolist())
    joint=ab@np.asarray(result["blend"]["coef"])*250
    incumbent=terr@np.asarray(result["legacy_territory"]["coef"])*250
    alpha=[]
    for j,h in enumerate(H):
        m=va&(n>=h)
        options=(0.,.1,.25,.5,.75,1.)
        scores=[np.abs(a*joint[m,:,j]+(1.-a)*incumbent[m,:,j]-y[m,j,None]).mean() for a in options]
        alpha.append(options[int(np.argmin(scores))])
    result["horizon_mix"]=alpha
    return result


def evaluate(z,model,mix,mask):
    public,ab,old,terr=frame_data(z,model);y=z["Y"];g=z["G"];t=z["T"];l=z["L"];games=json.loads(str(z["games"]))
    canonical=lambda x:np.repeat(x[:,None,:],2,axis=1)
    static=lambda x:np.repeat(x[:,:,None],2,axis=2)
    raw=z["B"]
    methods={"zero":canonical(np.zeros_like(y)),"asset_raw":static(np.repeat(raw[:,1,None],2,axis=1)),
             "v5_economic_raw":static(np.repeat(raw[:,0,None],2,axis=1)),"v5_territory_raw":static(raw[:,2:4]),"public_mlp":canonical(public)}
    for k,x in (("blend",ab),("legacy_combined",old),("legacy_territory",terr)):
        methods[k]=x@np.asarray(mix[k]["coef"])*250
    alpha=np.asarray(mix["horizon_mix"])
    methods["hybrid"]=alpha*methods["blend"]+(1.-alpha)*methods["legacy_territory"]
    def stats(which,exact):
        out={}
        for j,h in enumerate(H):
            m=which&(l>=h) if exact else which
            if not m.any():continue
            vals={}
            for name,p in methods.items():
                err=p[m,:,j]-y[m,j,None]
                vals[name]=dict(mae=round(float(np.abs(err).mean()),3),rmse=round(float(np.sqrt(np.mean(err**2))),3))
            out[str(h)]=dict(boards=int(m.sum()),games=len(np.unique(g[m])),methods=vals)
        return out
    paired={}
    for j,h in enumerate(H):
        vals=[]
        for i,gm in enumerate(games):
            m=mask&(g==i)&(l>=h)
            if not m.any():continue
            vals.append(dict(file=gm["file"],**{k:round(float(np.abs(p[m,:,j]-y[m,j,None]).mean()),4) for k,p in methods.items()}))
        by_old={}
        for name in ("legacy_combined","legacy_territory","v5_territory_raw","asset_raw"):
            d=np.array([v["hybrid"]-v[name] for v in vals]);boot=np.random.default_rng(713).choice(d,size=(5000,len(d))).mean(1)
            by_old[name]=dict(games_better=int((d<0).sum()),n=len(d),macro_delta=round(float(d.mean()),3),ci95=np.quantile(boot,[.025,.975]).round(3).tolist())
        paired[str(h)]=dict(comparisons=by_old,rows=vals)
    return dict(exact=stats(mask,True),last_19_turns_clipped=stats(mask&(t>=280),False),
                phases={str(lo)+"-"+str(hi):stats(mask&(t>=lo)&(t<=hi),True) for lo,hi in ((0,74),(75,149),(150,224),(225,298))},
                paired=paired,games=[gm for i,gm in enumerate(games) if (mask&(g==i)).any()])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("fit","external"));p.add_argument("cache",type=Path)
    p.add_argument("--model",type=Path,default=HERE/"forecast-model.npz");p.add_argument("--calibration",type=Path,default=HERE/"forecast-blend.json")
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();z=np.load(a.cache);model=np.load(a.model)
    if a.mode=="fit":
        mix=fit(z,model);a.calibration.write_text(json.dumps(mix,indent=2)+"\n")
    else:
        mix=json.loads(a.calibration.read_text())
    result=evaluate(z,model,mix,mask_for(z,2));a.output.write_text(json.dumps(result,indent=2)+"\n")
    print({h:{k:v["mae"] for k,v in x["methods"].items()} for h,x in result["exact"].items()})
