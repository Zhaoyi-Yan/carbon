"""Refit only the cash critic after averaging two compatible action networks."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from interpolate import mixed,network
from shipping_styles import _v7_shipping_gap
import train


def prepare(cache,names):
    data=train.load_rows(cache,names)
    ids=data['T'].numpy().copy()
    gaps=np.concatenate([_v7_shipping_gap(x.numpy().astype('f')*.01) for x in data['X'].split(128)])
    hits=(ids>=21)&(np.abs(gaps)>=95.)
    ids[hits]=23+2*(ids[hits]-21)+(gaps[hits]>0)
    return data['X'],torch.from_numpy(ids).long(),data['S'],data['Y']


def frozen_features(net,group,device,selection):
    xx,tt,ss,yy=group
    # Capture the critic's projected spatial channels before its two dense layers.
    features=[];pred=[]
    def record(layer,args):features.append(args[0].detach().cpu())
    hook=net.v2.register_forward_pre_hook(record)
    with torch.no_grad():
        for i in range(0,len(selection),96):
            take=selection[i:i+96]
            _,v=net(xx[take].to(device).float()*.01,tt[take].to(device),ss[take].to(device))
            pred.append(v.cpu())
    hook.remove()
    return torch.cat(features).to(device),ss[selection,0].to(device),yy[selection].to(device),torch.cat(pred).to(device)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','out','report'):ap.add_argument(k,type=Path)
    a=ap.parse_args();rng=np.random.default_rng(18343);torch.manual_seed(18343);torch.set_num_threads(1)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    tr,va=train.split(a.cache);gt=prepare(a.cache,tr);gv=prepare(a.cache,va)
    n=network(mixed(.65)).to(device)
    tx,ts,ty,tp=frozen_features(n,gt,device,np.flatnonzero((gt[1].numpy()>=21)|(rng.random(len(gt[1]))<.22)))
    vx,vs,vy,vp=frozen_features(n,gv,device,np.arange(len(gv[1])))
    opt=torch.optim.AdamW([*n.v2.parameters(),*n.v3.parameters()],lr=.0004,weight_decay=.00025)
    target=.85*ty+.15*tp
    def estimate(x,s):return n.v3(n.v2(x).relu())[:,0]+s
    scores=[];best=float('inf')
    for ep in range(9):
        if ep:
            for ix in np.array_split(rng.permutation(len(tx)),max(1,len(tx)//256)):
                v=estimate(tx[ix],ts[ix]);loss=F.smooth_l1_loss(v,target[ix],beta=.1)
                opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad(): err=float((estimate(vx,vs)-vy).abs().mean().cpu())
        scores.append(err);print(ep,err,flush=True)
        if err<best:
            best=err;chosen=ep;n.export(a.out)
        a.report.write_text(json.dumps(dict(training=tr,holdout=va,head_rows=len(tx),alpha=.65,criterion='held-out cash MAE; worker/base logits fixed',mae=scores,selected=chosen),indent=2)+'\n')


if __name__=='__main__':main()
