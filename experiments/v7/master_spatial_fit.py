"""Train a small toroidal, style-conditioned residual policy with short-trajectory supervision."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dual_weights import encoded,arrays
from master_data import NAMES
from master_worker_data import ROOT
HERE=Path(__file__).resolve().parent
PARENT=ROOT/'experiments/v7/checkpoints/harbormaster.py'

class Spatial(nn.Module):
    def __init__(self,mean,inv,head):
        super().__init__();C=24
        self.register_buffer('mean',torch.as_tensor(mean).reshape(1,-1,1,1))
        self.register_buffer('inv',torch.as_tensor(inv).reshape(1,-1,1,1))
        self.register_buffer('prior',torch.as_tensor(np.concatenate([head['hw.w'],head['hb.w']],axis=0)))
        self.register_buffer('bias',torch.as_tensor(np.r_[head['hw.b'],head['hb.b']]))
        self.stem=nn.Conv2d(180,C,1);self.spans=nn.ModuleList(nn.Conv2d(C,C,3) for _ in range(2))
        self.mix=nn.ModuleList(nn.Conv2d(C,C,1) for _ in range(2));self.world=nn.ModuleList(nn.Conv2d(C,C,1) for _ in range(2))
        self.heads=nn.Conv2d(C,10,1);nn.init.zeros_(self.heads.weight);nn.init.zeros_(self.heads.bias)
        self.style=nn.Parameter(torch.randn(3,C)*.05);self.temperature=nn.Parameter(torch.tensor([.82,.72]))
    def forward(self,x,sty):
        raw=F.conv2d(x[:,:96],self.prior,self.bias)
        z=F.relu(self.stem((x-self.mean)*self.inv)+self.style[sty,:,None,None])
        for j,d in enumerate((1,2)):
            h=F.relu(F.conv2d(F.pad(z,(d,d,d,d),mode='circular'),self.spans[j].weight,self.spans[j].bias,dilation=d))
            z=F.relu(z+self.mix[j](h)+self.world[j](z.mean((2,3),keepdim=True)))
        y=self.heads(z);return torch.cat((raw[:,:5]*self.temperature[0]+y[:,:5],
                        raw[:,5:]*self.temperature[1]+y[:,5:8],y[:,8:]),1).reshape(len(x),10,225)

def contents(path):
    with np.load(path) as z:
        n=len(z['h']);h=z['h'].transpose(0,2,1).reshape(n,96,15,15)
        x=z['x'].astype(np.float16)*np.float16(.01)
        trace=np.unpackbits(z['trace'],axis=-1,count=15)
        return np.concatenate((h,x,trace.astype(np.float16)),axis=1),z['y'],z['goal'],z['mask'],z['t'],z['rot']

def pool(rows,folder):
    data=defaultdict(list);groups=[]
    for r in rows:
        x,y,goal,mask,t,rot=contents(folder/r['cache']);n=len(x)
        for k,v in dict(x=x,y=y,goal=goal,mask=mask,style=np.full(n,NAMES.index(r['teacher']),np.int8)).items():data[k].append(v)
        groups.extend([r['group']]*n)
    return {k:torch.from_numpy(np.concatenate(v)) for k,v in data.items()},np.asarray(groups)

def calc(net,items,ids,device):
    x,y,goal,mask,style=[items[k][ids].to(device) for k in ('x','y','goal','mask','style')]
    y=y.long();style=style.long();q=net(x.float(),style);mask=mask.long()
    wt=torch.tensor([1.,.40,1.85],device=device)[style,None]
    weights=wt*(1.+.75*((mask&2)>0).float()+.65*((mask&4)>0).float())*(y[:,0]>=0)
    worker=F.cross_entropy(q[:,:5],y[:,0],ignore_index=-1,reduction='none')
    wl=(worker*weights).sum()/weights.sum().clamp_min(1)
    bf=(y[:,1]>=0)*wt;bc=F.cross_entropy(q[:,5:8],y[:,1],ignore_index=-1,reduction='none')
    bl=(bc*bf).sum()/bf.sum().clamp_min(1)
    track=(goal[:,2]>0)*wt
    gl=(F.smooth_l1_loss(q[:,8:],goal[:,:2].float(),reduction='none').sum(1)*track).sum()/track.sum().clamp_min(1)
    return wl+.24*bl+.18*gl,q

def inner_loss(m,items,ix,dev):
    nll=[]
    with torch.inference_mode():
        for b in np.array_split(ix,max(1,len(ix)//40)):
            _,q=calc(m,items,b,dev);z=q[:,:5].cpu()
            y=items['y'][b,0].long();c=F.cross_entropy(z,y,ignore_index=-1,reduction='none').numpy()
            kind=items['style'][b].numpy();mask=items['mask'][b].numpy()
            for j in range(len(b)):
                sel=y[j].numpy()>=0
                if not sel.any():continue
                w=(1+.6*((mask[j]&2)>0)+.6*((mask[j]&4)>0))*sel
                nll.append((kind[j],float((c[j]*w).sum()/w.sum())))
    return .55*np.mean([v for k,v in nll if k==2])+.45*np.mean([v for k,v in nll if k==0])
def train(maker,data,idx,inner,epochs,dev):
    torch.manual_seed(81723);rng=np.random.RandomState(81723);model=maker().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=.0016,weight_decay=.002)
    history=[]
    for ep in range(epochs):
        model.train()
        for ii in np.array_split(rng.permutation(idx),max(1,len(idx)//48)):
            opt.zero_grad(set_to_none=True);loss,_=calc(model,data,ii,dev);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if len(inner):
            model.eval();value=inner_loss(model,data,inner,dev);history.append(float(value))
        print('epoch',ep+1,'inner' if len(inner) else 'refit',history[-1] if len(inner) else '',flush=True)
    return model,history
def main(data,output,reference=None):
    torch.set_num_threads(1);dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    folder=data/'master-spatial';rows=json.loads((data/'master-worker/index.json').read_text())
    rows=[r for r in rows if r['fold']=='train'];assert len({r['group'] for r in rows})==42
    items,grp=pool(rows,folder);idx=np.arange(len(grp));inner=set()
    for style in NAMES:
        same=[r for r in rows if r['teacher']==style]
        same.sort(key=lambda r:hashlib.sha256(('spatial-inner-1'+r['group']).encode()).hexdigest());inner.update(r['group'] for r in same[:2])
    train_idx=idx[~np.isin(grp,list(inner))];val_idx=idx[np.isin(grp,list(inner))]
    # Standardization uses only the inner training group.
    norms=items['x'][train_idx[::12],:,:,::2].float();mean=norms.mean((0,2,3));sd=norms.std((0,2,3));del norms
    floor=torch.tensor([.25]*96+[.18]*84);inv=1./torch.maximum(sd,floor)
    head=arrays(encoded(PARENT));maker=lambda:Spatial(mean,inv,head)
    first,h=train(maker,items,train_idx,val_idx,12,dev);selected=int(np.argmin(h))+1
    del first
    model,_=train(maker,items,idx,[],selected,dev)
    weights={k:v.detach().cpu().numpy() for k,v in model.state_dict().items() if k not in ('prior','bias')}
    if reference is not None:np.savez_compressed(reference,**weights)
    np.savez_compressed(output,**{k:v.astype(np.float16) for k,v in weights.items()})
    meta=dict(epochs=selected,inner_nll=h,fit_boards=len(idx),inner_boards=len(val_idx),inner_maps=sorted(inner),train_maps=sorted(set(grp)),device=str(dev),source_sha=hashlib.sha256(PARENT.read_bytes()).hexdigest())
    (ROOT/'analysis/v7/master-spatial-fit.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path);p.add_argument('--reference',type=Path);a=p.parse_args();main(a.data,a.output,a.reference)
