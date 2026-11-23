"""Matched-budget full-network behavior cloning, small rollout refit and value calibration."""
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from network import INITIAL,PARENT,START,ROOT,make,pack
from collect import write

sys.path.insert(0,str(ROOT/'experiments/v6r'))
from train import symmetries

STEPS=192
BATCH=32
EPOCHS=3
SEED=271124
FIELDS=('X','A','S','Y','T','M')


def sample(out,maps,arm,epoch):
    rng=np.random.default_rng(SEED+epoch)
    subset=[m for m in maps if m['fold']=='train']
    # Equalize teachers first, then games within each teacher. Each row is an entire team.
    weights=np.array([1/6]*6+[0]*4) if arm=='top6' else np.array([.83/6]*6+[.05,.05,.035,.035])
    n=STEPS*BATCH
    counts=np.floor(weights*n).astype(int)
    counts[np.argsort(-(weights*n-counts))[:n-counts.sum()]]+=1
    want=defaultdict(dict)
    for t,count in enumerate(counts):
        names=[m['game'] for m in subset if t in m['teachers']]
        for i,total in enumerate(rng.multinomial(count,np.full(len(names),1/len(names)))):
            if total:want[names[i]][t]=total
    return assemble(out,want,rng)


def assemble(out,want,rng):
    rows=defaultdict(list)
    for name,targets in want.items():
        with np.load(out/'encoded'/(name+'.npz')) as z:
            tids=z['T'];ind=[]
            for t,count in targets.items():
                possible=np.flatnonzero(tids==t)
                if count is None:ind.extend(possible[::6])
                else:ind.extend(rng.choice(possible,count,replace=count>len(possible)))
            for k in FIELDS:rows[k].append(z[k][ind])
    return {k:torch.from_numpy(np.concatenate(v)) for k,v in rows.items()}


def evaldata(out,maps,fold):
    return assemble(out,{m['game']:{t:None for t in m['teachers']} for m in maps if m['fold']==fold},None)


def assess(net,rows,device):
    net.eval();stats=defaultdict(lambda:np.zeros(4))
    with torch.inference_mode():
        for chunk in range(0,len(rows['X']),64):
            x,a,s,y,t,m=[rows[k][chunk:chunk+64].to(device) for k in FIELDS]
            logits,v=net(x.float()*.01,t.long()+START,s)
            err=(v-y).abs().cpu().numpy()
            for i,(lo,hi) in enumerate(((0,5),(5,8))):
                choice=a[:,i].long();valid=choice>=0
                scores=logits[:,:,lo:hi].log_softmax(-1)
                ce=-scores.gather(-1,choice.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                hits=(scores.argmax(-1)==choice)
                tt=t.cpu().numpy()
                masks={'worker' if i==0 else 'base':valid}
                if i==0:
                    masks.update(loaded_return=valid & ((m&1)!=0),plant_or_seize=valid & ((m&2)!=0),low_cash=valid & ((m&4)!=0))
                else:masks['recruit']=valid & (choice>0)
                for key,mask in masks.items():
                    counts=mask.sum(1).cpu().numpy();nll=(ce*mask).sum(1).cpu().numpy();ok=(hits*mask).sum(1).cpu().numpy()
                    for teacher in np.unique(tt):
                        which=tt==teacher
                        stats[f'{teacher}/{key}']+=np.array([nll[which].sum(),ok[which].sum(),counts[which].sum(),0])
                if i==0:
                    for teacher in np.unique(tt):
                        which=tt==teacher
                        stats[f'{teacher}/value']+=np.array([err[which].sum(),0,which.sum(),0])
    return {k:dict(loss=float(v[0]/v[2]),accuracy=float(v[1]/v[2]),count=int(v[2])) for k,v in stats.items() if v[2]}


def equal_teacher(metrics,role):
    return float(np.mean([metrics[f'{i}/{role}']['loss'] for i in range(6)]))


def update(net,prior,rows,dev,opt,rng):
    net.train()
    transforms=symmetries()
    for ids in np.array_split(rng.permutation(len(rows['X'])),STEPS):
        x,a,s,y,t=[rows[k][ids] for k in ('X','A','S','Y','T')]
        cells,planes,acts=transforms[rng.integers(8)]
        x=x.flatten(2)[:,planes][:,:,cells].reshape(-1,54,15,15).to(dev).float()*.01
        a=a[:,:,cells].long();a[:,0]=acts[a[:,0]];a=a.to(dev);s=s.to(dev);y=y.to(dev)
        t=t.long().to(dev)
        old=torch.as_tensor(INITIAL,device=dev)[t]
        # Keep a general style active when the opponent identity is uncertain.
        generic=torch.from_numpy(rng.random(len(ids))<.18).to(dev)
        sid=torch.where(generic,20,t+START)
        old=torch.where(generic,20,old)
        with torch.no_grad():guide,past=prior(x,old,s)
        z,v=net(x,sid,s)
        loss=F.smooth_l1_loss(v,.65*past+.35*y,beta=.15)*.7
        for i,(lo,hi,strength) in enumerate(((0,5,1.),(5,8,.45))):
            label=a[:,i];valid=label>=0;count=valid.sum(1)
            logp=z[:,:,lo:hi].log_softmax(-1);prev=guide[:,:,lo:hi].log_softmax(-1)
            ce=-logp.gather(-1,label.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            kl=(prev.exp()*(prev-logp)).sum(-1)
            team=((ce+.27*kl)*valid).sum(1)/count.clamp_min(1)
            loss+=strength*team.sum()/(count>0).sum().clamp_min(1)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()


def main(out):
    torch.set_num_threads(1)
    dev='mps' if torch.backends.mps.is_available() else 'cpu'
    maps=json.loads((out/'split.json').read_text())
    val,test=[evaldata(out,maps,fold) for fold in ('validation','test')]
    # Set both arms' route prototypes from the same train-only, teacher-balanced sample.
    core=sample(out,maps,'top6',0)
    s=core['S'].numpy();t=core['T'].numpy()
    centers=np.stack([s[t==i].mean(0) for i in range(6)]);scale=np.maximum(s.std(0),.05)
    del core
    source=PARENT.read_text()
    report=dict(device=dev,steps=STEPS,batch=BATCH,epochs=EPOCHS,arms={},baseline={})
    for arm in ('top6','top6-plus'):
        report['arms'][arm]={}
        for size in ('main','small'):
            torch.manual_seed(SEED)
            prior=make(source,size=='small').to(dev).eval()
            if size not in report['baseline']:report['baseline'][size]=assess(prior,test,dev)
            net=copy.deepcopy(prior)
            opt=torch.optim.AdamW(net.parameters(),lr=.000075,weight_decay=.0001)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.000025)
            rng=np.random.default_rng(SEED)
            best=float('inf')
            record=dict(validation=[])
            for epoch in range(1,EPOCHS+1):
                rows=sample(out,maps,arm,epoch)
                update(net,prior,rows,dev,opt,rng);sched.step()
                del rows
                metrics=assess(net,val,dev)
                objective=equal_teacher(metrics,'worker')+.45*equal_teacher(metrics,'base')+.3*equal_teacher(metrics,'value')
                record['validation'].append(float(objective))
                if objective<best:
                    best=objective;record['selected']=epoch
                    selected=copy.deepcopy(net.state_dict())
            net.load_state_dict(selected)
            record['test']=assess(net,test,dev)
            net.export(out/(arm+'-'+size+'.npz'))
            report['arms'][arm][size]=record
            write(out/'fit.json',report)
            del selected,net,prior,opt,sched
        pack(out,arm,centers,scale)
    write(out/'router.json',dict(centers=centers.tolist(),scale=scale.tolist()))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
