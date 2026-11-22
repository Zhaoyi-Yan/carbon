"""Replay the live Lifeline style router and test original elite action retention outside Master training."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import types
from pathlib import Path
import numpy as np
import torch
from master_full import ROOT,PARENT,original,take_data,evaluate_holds,infer
from shipping_styles import _v7_shipping_gap
def routed(m,g,side):
    last=g['configuration']['episodeSteps']-2;avg=None;expert=22;out=[]
    for i,frame in enumerate(g['steps']):
        ob=frame[0]['observation'];pl=ob['players'];econ,_=m._v5_observe(pl,ob.get('trees') or {},i,last)
        dt=econ[side]-econ[1-side];p=.8*max(0.,-dt[5]/4.)+.7*max(0.,-dt[7])+.4*max(0.,dt[3]/3.)
        avg=p if avg is None else .84*avg+.16*p
        expert=21 if avg>(1.20 if expert==21 else 1.80) and i>32 and pl[side][0]<pl[1-side][0]+350 else 22
        out.append(expert)
    return np.asarray(out)
def old_elite(data,models,dev,exclude):
    folder=data.parent/'v6r-data/boards';rows=json.loads((ROOT/'experiments/v6r/elite-validation.json').read_text())['validation'];archive=[]
    for name in rows:
        with np.load(folder/(name+'.npz')) as f:
            group=str(f['layout'])
            if group in exclude:continue
            pick=np.flatnonzero(f['T']>=21)
            if not len(pick):continue
            x=f['X'][pick];ex=f['T'][pick].astype(int);gap=_v7_shipping_gap(x.astype('f')*.01)
            ids=np.where(np.abs(gap)<95,ex,23+2*(ex-21)+(gap>0)).astype('int64')
            d=dict(x=x,s=f['S'][pick],sid=ids);y=f['A'][pick];labels=f['Y'][pick]
        res=[]
        for model in models:
            z,v=infer(model,d,np.arange(len(x)),dev);out={}
            for role in (0,1):
                a=y[:,role];valid=a>=0;q=(z[:,:,:5] if role==0 else z[:,:,5:])[valid];a=a[valid]
                if not len(q):continue
                t=q-q.max(1,keepdims=True);logp=t-np.log(np.exp(t).sum(1,keepdims=True))
                out[['worker','base'][role]]=[len(a),float(-logp[np.arange(len(a)),a].sum()),int((q.argmax(1)==a).sum())]
            selected=y[:,1]>=0;truth=y[:,1][selected];guess=z[:,:,5:][selected].argmax(1)
            out['base_confusion']=np.bincount(truth*3+guess,minlength=9).reshape(3,3).tolist()
            out['value']=[len(v),float(np.abs(v-labels).sum())];res.append(out)
        archive.append(dict(name=name,layout=group,original=res[0],full=res[1]))
    combined={}
    for which in ('original','full'):
        combined[which]={}
        combined[which]['base_confusion']=np.sum([r[which]['base_confusion'] for r in archive],axis=0).tolist()
        for role in ('worker','base','value'):
            ar=np.sum([r[which][role] for r in archive if role in r[which]],axis=0)
            combined[which][role]=dict(n=int(ar[0]),mae=float(ar[1]/ar[0])) if role=='value' else dict(n=int(ar[0]),nll=float(ar[1]/ar[0]),hit=float(ar[2]/ar[0]))
    return dict(maps=archive,combined=combined,definition='Old fixed elite boards, one archived orientation per frame, only actual Artemis/Apollo actions, same legal inputs, role and shipping-context embedding for both networks. Maps sharing exact initial resource and base hash with any Master training/validation map excluded.')
def main(data,source):
    torch.set_num_threads(1);dev='mps' if torch.backends.mps.is_available() else 'cpu'
    models=[original(p).to(dev).eval() for p in (PARENT,source)];rows=json.loads((data/'master-worker/index.json').read_text())
    hold=[r for r in rows if r['fold']=='test'];d=take_data(data,hold,False)
    m=types.ModuleType('frozen_router');code=PARENT.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)');exec(compile(code,'<frozen_lifeline>','exec'),m.__dict__)
    targets=[];current=[]
    for nr,row in enumerate(hold):
        with gzip.open(row['file']) as f:g=json.load(f)
        ix=np.flatnonzero(d['map']==nr);t=d['t'][ix];base=routed(m,g,row['side'])[t]
        gap=_v7_shipping_gap(d['x'][ix].astype('f')*.01);d['sid'][ix]=np.where(np.abs(gap)<95,base,23+2*(base-21)+(gap>0))
        cash=np.array([(q[0]['observation']['players'][row['side']][0]-q[0]['observation']['players'][1-row['side']][0])/3000 for q in g['steps']])
        targets.extend(.28*cash[np.minimum(len(cash)-1,t+20)]+.52*cash[np.minimum(len(cash)-1,t+52)]+.20*cash[np.minimum(len(cash)-1,t+100)])
        current.extend(cash[t])
    result=evaluate_holds(*models,d,hold,dev)
    vals=[];targets=np.asarray(targets)[::4];current=np.asarray(current)[::4]
    for model in models:
        _,v=infer(model,d,np.arange(len(d['x'])),dev);v=v.reshape(-1,4).mean(1)
        vals.append(dict(future_cash_target_mae=float(np.mean(np.abs(v-targets))),
                         future_cash_change_correlation=float(np.corrcoef(v-current,targets-current)[0,1])))
    result.update(real_cash=dict(lifeline=vals[0],full=vals[1],meaning='Held-out observed future cash difference at +20/+52/+100, weighted .28/.52/.20 and divided by 3000. Correlation removes current bank differential from both prediction and observed future.'),
                  real_style_worker_snapshot=Counter(str(int(v)) for v in d['sid'][::4]),source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
                  old_elite=old_elite(data,models,dev,{r['group'] for r in rows}))
    (ROOT/'analysis/v7/master-full-preservation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['real_cash'],result['real_style_worker_snapshot'],result['old_elite']['combined'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('source',type=Path);a=p.parse_args();main(a.data,a.source)
