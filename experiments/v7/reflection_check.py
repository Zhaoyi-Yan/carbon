"""Evaluate equal-cost rotations and reflections of the same policy against held-out moves."""
import argparse
import json
from pathlib import Path
import numpy as np
from branch_replay import ROOT,OUT,HERE
from benchmark import configure_engine,limit_math_threads,load_strategy


def transforms():
    r0,c0=np.indices((15,15)).reshape(2,225);rot=np.array([0,2,3,4,1])
    result=[]
    for reflected in (False,True):
        r,c=r0.copy(),(-c0 if reflected else c0).copy();act=np.array([0,1,4,3,2]) if reflected else np.arange(5)
        for i in range(4):
            cells=r%15*15+c%15;planes=np.arange(54)
            for op in (25,30):planes[op+act]=op+np.arange(5)
            result.append((cells,np.argsort(cells),act.copy(),planes));r,c=c,-r;act=rot[act]
    return result


def main():
    pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('cache',type=Path);args=pa.parse_args()
    limit_math_threads();configure_engine();m=load_strategy('tta_calibration',HERE/'checkpoints/soundings.py');m._torch.set_num_threads(1)
    maps=transforms()
    choices={'rotation4':[0,1,2,3],'reflection4':[0,2,4,6],'diagonal4':[0,2,5,7],'all8':list(range(8))};rows=[]
    for name in json.loads((OUT/'value-refit.json').read_text())['holdout']:
        with np.load(args.cache/(name+'.npz'),allow_pickle=False) as d:
            ix=np.flatnonzero(d['T']>=21)[::4]
            xx=d['X'][ix].astype('f')*.01;labels=d['A'][ix];tt=d['T'][ix].copy();ss=d['S'][ix];yy=d['Y'][ix]
        gaps=m._v7_shipping_gap(xx);special=np.abs(gaps)>=95;tt[special]=23+2*(tt[special]-21)+(gaps[special]>0)
        metrics={k:np.zeros(5) for k in choices}
        for i in range(0,len(xx),6):
            x=xx[i:i+6];a=labels[i:i+6];t=tt[i:i+6];sc=ss[i:i+6];y=yy[i:i+6]
            outs=[];n=len(t);buf=x.reshape(n,54,225)
            for cells,idx,act,channels in maps:
                v=buf[:,channels][:,:,idx].reshape(n,54,15,15)
                w,b,va=m._net()(np.ascontiguousarray(v),t,sc)
                outs.append((w[:,cells][:,:,act],b[:,cells],va))
            for k,group in choices.items():
                stats=metrics[k]
                for slot in (0,1):
                    z=sum(outs[j][slot] for j in group)/len(group)
                    valid=a[:,slot]>=0;target=a[:,slot][valid];scores=z[valid]
                    if len(target):
                        logp=scores-np.logaddexp.reduce(scores,axis=1)[:,None]
                        stats[0]+=-logp[np.arange(len(target)),target].sum();stats[1]+=(scores.argmax(1)==target).sum();stats[2]+=len(target)
                est=sum(outs[j][2] for j in group)/len(group);stats[3]+=np.abs(est-y).sum();stats[4]+=n
        rows.append(dict(layout=name,metrics={k:v.tolist() for k,v in metrics.items()}))
    summ={}
    for k in choices:
        z=np.sum([r['metrics'][k] for r in rows],axis=0);summ[k]=dict(nll=z[0]/z[2],accuracy=z[1]/z[2],actions=int(z[2]),value_mae=z[3]/z[4])
    (OUT/'reflection-validation.json').write_text(json.dumps(dict(results=summ,per_map=rows),indent=2)+'\n');print(summ)
if __name__=='__main__':main()
