"""Fine-tune the full v4 trunk with twenty style embeddings and a real-return value target."""
import argparse
import base64
import io
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from backbone import load
from distill import data, Student


def load_full():
    baseline=load()
    policy=baseline._POLICY
    v=np.load(io.BytesIO(base64.b64decode(baseline._VALUE)))
    model=Student(width=96,blocks=policy.nb)
    model.pool=nn.Conv2d(96,32,1);model.v2=nn.Linear(69,128);model.v3=nn.Linear(128,1)
    def fill(layer,w,b):
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
    def policy_layer(layer,name):fill(layer,policy.p[name+'.w'],policy.p[name+'.b'])
    policy_layer(model.stem,'stem');policy_layer(model.workers,'hw');policy_layer(model.bases,'hb')
    for i,(c,g,d) in enumerate(model.res):
        for layer,key in ((c,'c1'),(g,'g'),(d,'c2')):policy_layer(layer,'b%d.%s'%(i,key))
    for layer,key in ((model.pool,'v1'),(model.v2,'v2'),(model.v3,'v3')):fill(layer,v[key+'.weight'],v[key+'.bias'])
    ranking=json.loads((Path(__file__).parent/'design.json').read_text())['ranking']
    indices=[policy.snames.index('CL-'+r['name'])+1 if 'CL-'+r['name'] in policy.snames else baseline._SID for r in ranking]
    with torch.no_grad():model.style.weight.copy_(torch.from_numpy(policy.p['emb'][indices+[baseline._SID]]))
    return model


def transform(rows,t,r,c):
    dr=np.array([0,-1,0,1,0]);dc=np.array([0,0,1,0,-1])
    rr,cc=np.indices((15,15)).reshape(2,225)
    if t:rr,cc=cc,rr;dr,dc=dc,dr
    if r:rr=-rr;dr=-dr
    if c:cc=-cc;dc=-dc
    mapping=[(0,0),(-1,0),(0,1),(1,0),(0,-1)]
    actions=np.array([mapping.index(x) for x in zip(dr,dc)])
    cells=np.argsort((rr%15)*15+cc%15)
    planes=np.arange(54)
    for start in (25,30):planes[start+actions]=start+np.arange(5)
    x=rows['X'].flatten(2)[:,planes][:,:,cells].reshape(-1,54,15,15)
    lab=rows['A'][:,:,cells].clone()
    swap=torch.tensor(np.r_[actions,-1],dtype=torch.long)
    lab[:,0]=swap[lab[:,0].long()].to(torch.int8)
    mapped=rows['L'][:,cells][:,:,np.r_[np.argsort(actions),[5,6,7]]]
    return x,lab,mapped


def run(model,data,opt,dev,rng,batch):
    n=len(data['X']);ordering=rng.permutation(n) if opt else np.arange(n)
    metrics=torch.zeros(5,device=dev)
    for st in range(0,n,batch):
        ix=ordering[st:st+batch];rows={k:v[ix] for k,v in data.items()}
        x,labels,guide=(transform(rows,*rng.integers(0,2,3)) if opt and rng.random()<.59
                         else (rows['X'],rows['A'],rows['L']))
        x=x.to(dev).float()*.01;labels=labels.to(dev).long();guide=guide.to(dev).float()
        s=rows['S'].to(dev);t=rows['T'].to(dev).long()
        if opt:t=torch.where(torch.rand(len(t),device=dev)<.21,20,t)
        z,estimate=model(x,t,s)
        mask=labels>=0;losses=[];ce_sum=0;hits=0
        targets=rows['Y'].to(dev);old=rows['V'].to(dev)
        # Advantage weights are detached observed returns. Clipping limits one-sided games.
        weight=(.6*(targets[:,0]-old)+.4*(targets[:,1]-old)).mul(2.).clamp(-.65,.65).exp()[:,None]
        for slot,(i,j) in enumerate(((0,5),(5,8))):
            zz=z[:,:,i:j];orig=guide[:,:,i:j];sel=mask[:,slot];y=labels[:,slot].clamp_min(0)
            logp=zz.log_softmax(-1)
            ce=-logp.gather(-1,y[:,:,None])[:,:,0];pr=orig.softmax(-1)
            kl=(pr*(orig.log_softmax(-1)-logp)).sum(-1)
            losses.append(((weight*ce+.42*kl)*sel).sum()/(sel*weight).sum().clamp_min(1.))
            ce_sum+=(ce*sel).sum();hits+=((zz.argmax(-1)==y)*sel).sum()
        goal=.55*old+.18*targets[:,0]+.27*targets[:,1]
        loss=losses[0]+.45*losses[1]+.85*F.smooth_l1_loss(estimate,goal,beta=.12)
        if opt:
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        metrics+=torch.stack((ce_sum,hits,mask.sum(),(estimate-goal).abs().sum(),torch.tensor(len(ix),device=dev))).detach()
    ce,h,c,err,num=metrics.cpu().tolist()
    return dict(nll=ce/c,accuracy=h/c,value_mae=err/num)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','split','weights','report'):p.add_argument(k,type=Path)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch',type=int,default=64)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(7003);rng=np.random.default_rng(7003)
    dev='mps' if torch.backends.mps.is_available() else 'cpu'
    split=json.loads(a.split.read_text());tr=data(a.cache,split['training_matches']);va=data(a.cache,split['validation_matches'])
    net=load_full().to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=.00009,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=.00002)
    with torch.no_grad():hist=[dict(epoch=0,validation=run(net,va,None,dev,rng,a.batch))]
    print(hist[-1],flush=True);best=float('inf')
    for ep in range(a.epochs):
        net.train();train=run(net,tr,opt,dev,rng,a.batch);schedule.step()
        net.eval()
        with torch.no_grad():valid=run(net,va,None,dev,rng,a.batch)
        if valid['nll']+.4*valid['value_mae']<best:
            best=valid['nll']+.4*valid['value_mae'];net.export(a.weights)
        hist.append(dict(epoch=ep+1,train=train,validation=valid));print(hist[-1],flush=True)
        a.report.write_text(json.dumps(hist,indent=2)+'\n')


if __name__=='__main__':
    main()
