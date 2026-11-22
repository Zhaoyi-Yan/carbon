"""Refit the full policy on two elite teachers with separate embeddings and a cash critic."""
import argparse
import base64
import hashlib
import io
import json
import re
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/v5'))
from refit import load_full


def load_parent(path):
    text=path.read_text()
    arrays={}
    for name in ('_WEIGHTS','_VALUE'):
        encoded=re.search('^'+name+r' = "([^"]*)"$',text,re.M)[1]
        with np.load(io.BytesIO(base64.b64decode(encoded)),allow_pickle=False) as z:arrays.update(dict(z))
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'init.npz';np.savez(p,**arrays);return load_full(p)


def split(cache):
    index=json.loads((cache/'index.json').read_text())
    names=sorted(index,key=lambda n:index[n][0],reverse=True);used=set();tr=[];va=[];counts={21:0,22:0}
    for name in names:
        with np.load(cache/(name+'.npz')) as z:layout=str(z['layout']);elite=set(z['T'].tolist()) & {21,22}
        if layout in used: continue
        used.add(layout)
        dest=va if any(counts[k]<6 for k in elite) else tr
        dest.append(name)
        if dest is va:
            for k in elite:counts[k]+=1
    return tr,va


def load_rows(cache,names):
    agg={}
    for name in names:
        with np.load(cache/(name+'.npz'),allow_pickle=False) as z:
            for k in ('X','S','T','Y','A','W'):agg.setdefault(k,[]).append(z[k])
    return {k:torch.from_numpy(np.concatenate(parts)) for k,parts in agg.items()}


def symmetries():
    maps=[];rr,cc=np.indices((15,15)).reshape(2,225);directions=[(0,0),(-1,0),(0,1),(1,0),(0,-1)]
    for flip in (False,True):
        for rot in range(4):
            def change(r,c):
                if flip:c=-c
                for _ in range(rot):r,c=c,-r
                return r,c
            r,c=change(rr,cc);acts=np.array([directions.index(change(a,b)) for a,b in directions])
            planes=np.arange(54)
            for start in (25,30):planes[start+acts]=start+np.arange(5)
            maps.append((np.argsort((r%15)*15+c%15),planes,torch.tensor(np.r_[acts,-1])))
    return maps


def epoch(net,prior,data,device,rng,opt,batch,distill=.24,forecast=.50):
    n=len(data['X']);order=rng.permutation(n) if opt else np.arange(n);maps=symmetries();tot=np.zeros(5)
    for start in range(0,n,batch):
        ind=order[start:start+batch]
        x,s,t,y,a,w=[data[k][ind] for k in ('X','S','T','Y','A','W')]
        if opt:
            cells,planes,acts=maps[int(rng.integers(8))]
            x=x.flatten(2)[:,planes][:,:,cells].reshape(-1,54,15,15);a=a[:,:,cells].clone();w=w[:,:,cells]
            a[:,0]=acts[a[:,0].long()].to(torch.int8)
        x=x.to(device).float()*.01;s=s.to(device);t=t.to(device).long();y=y.to(device)
        a=a.to(device).long();w=w.to(device).float()
        with torch.no_grad():
            guide,prior_v=prior(x,torch.where(t>=21,0,t),s)
        z,v=net(x,t,s)
        elite=t>=21
        role_weight=torch.where(elite,1.,.16)[:,None]
        loss=0.;nll=hits=actions=0
        for slot,(lo,hi) in enumerate(((0,5),(5,8))):
            valid=a[:,slot]>=0;labels=a[:,slot].clamp_min(0)
            logp=z[:,:,lo:hi].log_softmax(-1);prev=guide[:,:,lo:hi].log_softmax(-1)
            ce=-logp.gather(-1,labels[:,:,None])[:,:,0]
            kl=(prev.exp()*(prev-logp)).sum(-1)
            weight=valid*role_weight*w[:,slot].clamp(.5,1.8)
            loss+=((ce+distill*kl)*weight).sum()/weight.sum().clamp_min(1)*(1. if slot==0 else .48)
            mask=valid & elite[:,None]
            nll=nll+(ce*mask).sum();hits=hits+((logp.argmax(-1)==labels)*mask).sum();actions=actions+mask.sum()
        goal=(1-forecast)*prior_v+forecast*y
        loss+=.65*F.smooth_l1_loss(v,goal,beta=.13)
        if opt:
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        tot+=torch.stack([nll,hits,actions,(v-y).abs().sum(),torch.ones((),device=device)*len(t)]).detach().cpu().numpy()
    return dict(nll=tot[0]/tot[2],accuracy=tot[1]/tot[2],value_mae=tot[3]/tot[4],actions=int(tot[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','source','weights','report'):p.add_argument(k,type=Path)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--warm',type=Path)
    p.add_argument('--learning-rate',type=float,default=.00015)
    p.add_argument('--distill',type=float,default=.24)
    p.add_argument('--forecast',type=float,default=.50)
    a=p.parse_args();torch.manual_seed(83117);rng=np.random.default_rng(83117);torch.set_num_threads(1)
    tr,va=split(a.cache);train=load_rows(a.cache,tr);valid=load_rows(a.cache,va)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    prior=load_parent(a.source).to(device).eval()
    import copy
    net=copy.deepcopy(prior);emb=nn.Embedding(23,96).to(device)
    with torch.no_grad():
        emb.weight[:21].copy_(net.style.weight);emb.weight[21:].copy_(net.style.weight[0].expand(2,-1))
    net.style=emb
    if a.warm:
        with np.load(a.warm,allow_pickle=False) as archive:raw=dict(archive)
        whole=raw['emb'];raw['emb']=whole[:21]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trunk.npz';np.savez(path,**raw);net=load_full(path)
        net.style=nn.Embedding.from_pretrained(torch.tensor(whole),freeze=False)
        net=net.to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=a.learning_rate,weight_decay=.00012)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=.000035)
    report={'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'training':tr,'validation':va,'epochs':[],'warm_sha256':hashlib.sha256(a.warm.read_bytes()).hexdigest() if a.warm else None,'learning_rate':a.learning_rate,'distill':a.distill,'forecast':a.forecast}
    best=float('inf')
    for ep in range(a.epochs+1):
        if ep:
            net.train();epoch(net,prior,train,device,rng,opt,a.batch,a.distill,a.forecast);schedule.step()
        net.eval()
        with torch.no_grad():m=epoch(net,prior,valid,device,rng,None,a.batch,a.distill,a.forecast)
        report['epochs'].append(m);print(ep,m,flush=True)
        if ep and m['nll']+.25*m['value_mae']<best:
            best=m['nll']+.25*m['value_mae'];net.export(a.weights);report['selected']=ep
        a.report.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
