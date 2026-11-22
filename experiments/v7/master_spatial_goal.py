"""Toroidal destination attention, trained to recover the real cell ten steps in the future."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import master_spatial_fit as base
from master_spatial_fit import ROOT,HERE,PARENT,NAMES
from dual_weights import arrays,encoded
from distill_data import load

def geometry():
    p=np.arange(225);dy=(p[None,:]//15-p[:,None]//15+7)%15-7;dx=(p[None,:]%15-p[:,None]%15+7)%15-7
    dist=np.abs(dy)+np.abs(dx)
    regions=np.stack([dy/7.,dx/7.,dy<0,dx>0,dy>0,dx<0,dist==0,dist/14.],axis=-1).astype('f')
    return dist.astype('f'),regions

class GoalNet(base.Spatial):
    def __init__(self,mean,inv,head):
        super().__init__(mean,inv,head);C=24
        self.query=nn.Conv2d(C,12,1);self.key=nn.Conv2d(C,12,1);self.token=nn.Conv2d(C,8,1)
        self.combine=nn.Conv2d(C+16,C,1);self.out=nn.Conv2d(C,8,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        dist,regions=geometry();self.register_buffer('distance',torch.from_numpy(dist));self.register_buffer('regions',torch.from_numpy(regions))
    def forward(self,x,sty):
        raw=F.conv2d(x[:,:96],self.prior,self.bias)
        z=F.relu(self.stem((x-self.mean)*self.inv)+self.style[sty,:,None,None])
        for j,d in enumerate((1,2)):
            h=F.relu(F.conv2d(F.pad(z,(d,d,d,d),mode='circular'),self.spans[j].weight,self.spans[j].bias,dilation=d))
            z=F.relu(z+self.mix[j](h)+self.world[j](z.mean((2,3),keepdim=True)))
        ask=self.query(z).flatten(2).transpose(1,2);key=self.key(z).flatten(2)
        self.goal_scores=torch.bmm(ask,key)*.288675-.40*self.distance
        mass=F.softmax(self.goal_scores,dim=-1)
        tokens=torch.bmm(mass,self.token(z).flatten(2).transpose(1,2))
        dirs=torch.einsum('bij,ijf->bif',mass,self.regions)
        ctx=torch.cat((tokens,dirs),dim=2).transpose(1,2).reshape(len(x),16,15,15)
        correction=self.out(F.relu(self.combine(torch.cat((z,ctx),1))))
        y=self.heads(z)
        return torch.cat((raw[:,:5]*self.temperature[0]+y[:,:5]+correction[:,:5],
                     raw[:,5:]*self.temperature[1]+y[:,5:8]+correction[:,5:8],y[:,8:]),1).reshape(len(x),10,225)

def ten_labels(rows,folder):
    # Independent of carbon, cached frozen activations and optimizer state.
    m=load(PARENT);outputs=[]
    for r in rows:
        with gzip.open(r['file']) as f:g=json.load(f)
        with np.load(folder/r['cache']) as z:t,rot=z['t'],z['rot']
        yy=np.full((len(t),225),-1,np.int16);s=r['side'];st=g['steps']
        for i,(step,j) in enumerate(zip(t,rot)):
            if step+10>=len(st):continue
            cells=m._V5_ROTATIONS[j][0];own=st[step][0]['observation']['players'][s][2];after=st[step+10][0]['observation']['players'][s][2]
            for u,(q,v,k) in own.items():
                if u in after:yy[i,cells[q]]=cells[after[u][0]]
        outputs.append(yy)
    return torch.from_numpy(np.concatenate(outputs))
def main(data,output):
    torch.set_num_threads(1);dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    folder=data/'master-spatial';rows=[r for r in json.loads((data/'master-worker/index.json').read_text()) if r['fold']=='train']
    ref=json.loads((ROOT/'analysis/v7/master-spatial-fit.json').read_text());items,grp=base.pool(rows,folder);idx=np.arange(len(grp))
    items['end10']=ten_labels(rows,folder)
    inner=np.isin(grp,ref['inner_maps']);train_idx=idx[~inner];val_idx=idx[inner]
    v=items['x'][train_idx[::12],:,:,::2].float();mean=v.mean((0,2,3));sd=v.std((0,2,3));del v
    inv=1./torch.maximum(sd,torch.tensor([.25]*96+[.18]*84));head=arrays(encoded(PARENT))
    previous=base.calc
    def trajectory(net,items,ii,device):
        loss,q=previous(net,items,ii,device)
        labels=items['end10'][ii].to(device).long()
        per=F.cross_entropy(net.goal_scores.transpose(1,2),labels,ignore_index=-1,reduction='none')
        weights=((labels>=0)*(1.+.60*((items['mask'][ii].to(device).long()&6)>0))).float()
        return loss+.09*(per*weights).sum()/weights.sum().clamp_min(1),q
    base.calc=trajectory
    maker=lambda:GoalNet(mean,inv,head)
    first,history=base.train(maker,items,train_idx,val_idx,8,dev);epochs=int(np.argmin(history))+1;del first
    model,_=base.train(maker,items,idx,[],epochs,dev)
    weights={k:v.detach().cpu().numpy() for k,v in model.state_dict().items() if k not in ('prior','bias','distance','regions')}
    np.savez_compressed(output,**weights)
    (ROOT/'analysis/v7/master-spatial-goal-fit.json').write_text(json.dumps(dict(epochs=epochs,inner_nll=history,inner_maps=ref['inner_maps'],
          train_maps=ref['train_maps'],boards=len(idx),second_head='Cell occupied ten turns later, conditional on survival. Labels are never inputs.'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.data,a.output)
