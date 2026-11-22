"""Train a small circular CNN on real actions, v4 soft targets and future game scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Student(nn.Module):
    def __init__(self,width=32,blocks=3):
        super().__init__()
        self.stem=nn.Conv2d(54,width,3)
        self.style=nn.Embedding(21,width)
        nn.init.zeros_(self.style.weight)
        self.res=nn.ModuleList([nn.ModuleList([nn.Conv2d(width,width,3),nn.Linear(width,width),
                                                nn.Conv2d(width,width,3)]) for _ in range(blocks)])
        self.workers=nn.Conv2d(width,5,1);self.bases=nn.Conv2d(width,3,1)
        self.pool=nn.Conv2d(width,24,1);self.v2=nn.Linear(53,64);self.v3=nn.Linear(64,1)
        for layers in self.res:
            nn.init.zeros_(layers[-1].bias)
            nn.init.normal_(layers[-1].weight,std=.012)

    def forward(self,x,t,s):
        def pad(v):
            v=torch.cat((v[:,:,:,-1:],v,v[:,:,:,:1]),dim=3)
            return torch.cat((v[:,:,-1:,:],v,v[:,:,:1,:]),dim=2).contiguous()
        h=(self.stem(pad(x))+self.style(t)[:,:,None,None]).relu()
        for first,global_layer,second in self.res:
            q=first(pad(h)).relu()
            q=q+global_layer(q.mean((2,3)))[:,:,None,None]
            h=(h+second(pad(q))).relu()
        cells=h.flatten(2).transpose(1,2).contiguous()
        project=lambda layer:F.linear(cells,layer.weight.flatten(1),layer.bias)
        v=project(self.pool).relu()
        y=self.v3(self.v2(torch.cat((v.mean(1),v.amax(1),s),1)).relu())[:,0]+s[:,0]
        return torch.cat((project(self.workers),project(self.bases)),2),y

    def export(self,path):
        result={'meta':np.array([len(self.res)]),'snames':np.array(['teacher%d'%i for i in range(20)])}
        def put(prefix,layer):
            result[prefix+'.w']=layer.weight.detach().cpu().numpy()
            result[prefix+'.b']=layer.bias.detach().cpu().numpy()
        put('stem',self.stem);put('hw',self.workers);put('hb',self.bases)
        for i,(first,gate,second) in enumerate(self.res):
            for name,layer in (('c1',first),('g',gate),('c2',second)):put('b%d.%s'%(i,name),layer)
        result['emb']=self.style.weight.detach().cpu().numpy()
        for name,layer in (('v1',self.pool),('v2',self.v2),('v3',self.v3)):
            result[name+'.weight']=layer.weight.detach().cpu().numpy();result[name+'.bias']=layer.bias.detach().cpu().numpy()
        np.savez_compressed(path,**result)


def data(root,names):
    fields={k:[] for k in ('X','S','L','T','A','Y','V')}
    with np.load(root/(names[0]+'.npz')) as check:
        if 'W' in check:fields['W']=[]
    for name in names:
        with np.load(root/(name+'.npz')) as z:
            for k in fields:fields[k].append(z[k])
    return {k:torch.from_numpy(np.concatenate(v)) for k,v in fields.items()}


def epoch(model,rows,opt,device,rng,batch):
    n=len(rows['X'])
    order=rng.permutation(n) if opt else np.arange(n)
    metrics=torch.zeros(5,device=device)
    for start in range(0,n,batch):
        ids=order[start:start+batch]
        fetch=lambda k:rows[k][ids].to(device)
        x=fetch('X').float()*.01;t=fetch('T').long();s=fetch('S').float()
        labels=fetch('A').long();guide=fetch('L').float();target=fetch('Y');old=fetch('V')
        # Embed the population policy in the spare row as dropout on teacher identity.
        if opt:t=torch.where(torch.rand(len(t),device=device)<.22,20,t)
        out,score=model(x,t,s)
        mask=labels>=0
        losses=[]; sums=[];hits=torch.zeros((),device=device)
        for which,(i,j) in enumerate(((0,5),(5,8))):
            pred=out[:,:,i:j];teacher=guide[:,:,i:j];active=mask[:,which]
            y=labels[:,which].clamp_min(0)
            logp=pred.log_softmax(-1)
            ce=-logp.gather(-1,y[:,:,None])[:,:,0]
            soft=-(teacher.div(1.55).softmax(-1)*pred.div(1.55).log_softmax(-1)).sum(-1)
            count=active.sum().clamp_min(1)
            losses.append(((ce+.52*soft)*(active)).sum()/count)
            sums.append((ce*active).sum())
            hits+=((pred.argmax(-1)==y)*active).sum()
        goal=.30*target[:,0]+.45*target[:,1]+.25*old
        val=F.smooth_l1_loss(score,goal,beta=.15)
        loss=losses[0]+.40*losses[1]+.95*val
        if opt:
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step()
        metrics+=torch.stack([sums[0]+sums[1],hits,mask.sum(),(score-goal).abs().sum(),torch.tensor(len(ids),device=device)]).detach()
    ce,ok,count,error,states=metrics.cpu().tolist()
    return dict(nll=ce/count,accuracy=ok/count,value_mae=error/states)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('cache','split','weights','report'):p.add_argument(arg,type=Path)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch',type=int,default=128)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(4452);rng=np.random.default_rng(4452)
    dev='mps' if torch.backends.mps.is_available() else 'cpu'
    report=json.loads(a.split.read_text())
    tr=data(a.cache,report['training_matches']);va=data(a.cache,report['validation_matches'])
    model=Student().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=.0018,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=.00020)
    history=[];best=float('inf')
    print(dev,'states',len(tr['X']),len(va['X']),flush=True)
    for i in range(a.epochs):
        model.train();train=epoch(model,tr,opt,dev,rng,a.batch);schedule.step()
        model.eval()
        with torch.no_grad(): valid=epoch(model,va,None,dev,rng,a.batch)
        history.append(dict(epoch=i+1,train=train,validation=valid))
        criterion=valid['nll']+.5*valid['value_mae']
        if criterion<best:
            best=criterion;model.export(a.weights)
        a.report.write_text(json.dumps(history,indent=2)+'\n');print(history[-1],flush=True)


if __name__=='__main__':
    main()
