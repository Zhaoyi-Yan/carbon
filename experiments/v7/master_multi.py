"""Single Lifeline trunk with three Master shipment embeddings and replay from original elite domains."""
import argparse
import base64
import copy
import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from master_full import HERE,ROOT,PARENT,SHA,original,take_data,infer,evaluate_holds,metrics
from dual_weights import encoded,arrays
from receipt_manifest import semantic_layout
from shipping_styles import _v7_shipping_gap


def _mm_switch(obs,normal,gap,state):
    step=obs['step'];side=obs['player'];own,enemy=obs['players'][side],obs['players'][1-side]
    if state.get('mm_tick')==step:return state['mm_id']
    if state.get('mm_tick')!=step-1:state['mm_signal']=0.;state['mm_active']=False
    depots=tuple(own[1].values())
    campers=[p for p,c,k in enemy[2].values() if k=='COLLECTOR' and c<=35 and any(dist(p,b)<=2 for b in depots)]
    threatened=any(k=='COLLECTOR' and cargo>=55 and any(dist(pos,b)<=4 for b in depots) and any(dist(pos,p)<=3 for p in campers) for pos,cargo,k in own[2].values())
    depleted=len(own[2])<=5 and len(enemy[2])>=len(own[2])+2 and own[0]<105
    under=depleted or (bool(campers) and (threatened or len(own[2])+2<=len(enemy[2])))
    score=.75*state.get('mm_signal',0.)+.25*under
    active=16<=step<=275 and score>=(.26 if state.get('mm_active') else .64)
    state['mm_signal']=score;state['mm_active']=active;state['mm_tick']=step
    state['mm_id']=(27 if abs(gap)<95 else 28+int(gap>0)) if active else normal
    return state['mm_id']

def build(code,weights=HERE/'master_multi_weights.npz'):
    if hashlib.sha256(code.encode()).hexdigest()!=SHA:raise ValueError('Expected frozen Lifeline')
    with np.load(weights) as arc:
        for key in ('_WEIGHTS','_VALUE'):
            raw=arrays(encoded(code,key));a={k:(arc[k] if k in arc and k not in ('meta','snames') else v) for k,v in raw.items()}
            f=io.BytesIO();np.savez_compressed(f,**a);prefix=base64.b64encode(f.getvalue()).decode()
            code=re.sub('^'+key+r' = "[^"\n]+"$',key+' = "'+prefix+'"',code,count=1,flags=re.M)
    marker='def _v5_root(obs, est, x, previous):';spot='    active_expert=_v7_shipping_id(expert,ship_gap)'
    original_tiny='    return np.where(ids>=23,21+(ids-23)//2,ids)'
    for s in (marker,spot,original_tiny):
        if code.count(s)!=1:raise ValueError(s)
    return code.replace(marker,inspect.getsource(_mm_switch)+'\n\n'+marker).replace(spot,spot+'\n    active_expert=_mm_switch(obs,active_expert,ship_gap,state)').replace(original_tiny,'    return np.where(ids>=27,22,np.where(ids>=23,21+(ids-23)//2,ids))')

def archives(data,rows):
    prior=json.loads((ROOT/'experiments/v6r/elite-validation.json').read_text())
    check=json.loads((ROOT/'analysis/v7/master-full-preservation.json').read_text())['old_elite']['maps']
    valid=[q['name'] for q in check];assert len(valid)==6
    paths={}
    for folder in (ROOT.parent/'v6r-data/artemis',ROOT.parent/'v5-data',ROOT.parent/'v6-data',ROOT/'replays',ROOT.parent/'v6r-data/latest-followup',data/'distill-public'):
        for p in folder.rglob('*.json.gz'):paths.setdefault(p.name.split('.')[0],p)
    def canonical(p):
        with gzip.open(p) as f:return semantic_layout(json.load(f))
    deny={r['group'] for r in rows}|set(json.loads((data/'distill-cache/index.json').read_text())['excluded_layouts'])
    canon={canonical(Path(r['file'])) for r in rows};folder=ROOT.parent/'v6r-data/boards'
    holding=[]
    for name in valid:
        with np.load(folder/(name+'.npz')) as z:key=str(z['layout'])
        sig=canonical(paths[name]);assert key not in deny and sig not in canon
        deny.add(key);canon.add(sig);holding.append(dict(name=name,layout=key,canonical=sig))
    sample=[]
    for name in sorted(prior['training'],key=lambda s:hashlib.sha256(('multi-source-'+s).encode()).digest()):
        with np.load(folder/(name+'.npz')) as z:
            key=str(z['layout']);elite=set(int(x) for x in z['T'] if x>=21)
        if key in deny or not elite:continue
        canonical_id=canonical(paths[name])
        if canonical_id in canon:continue
        t=next((v for v in sorted(elite) if sum(v in r['styles'] for r in sample)<27),None)
        if t is None:continue
        sample.append(dict(name=name,layout=key,canonical=canonical_id,styles=sorted(elite)));deny.add(key);canon.add(canonical_id)
        if all(sum(v in r['styles'] for r in sample)>=27 for v in (21,22)):break
    assert len(sample)>=35
    return sample,holding

def elite_data(records):
    folder=ROOT.parent/'v6r-data/boards';parts={k:[] for k in ('x','y','s','sid','mask','style','group','expert','domain')}
    for r in records:
        with np.load(folder/(r['name']+'.npz')) as z:
            ix=np.flatnonzero(z['T']>=21);rng=np.random.default_rng(int(r['layout'][:8],16))
            if len(ix)>78:ix=np.sort(rng.choice(ix,78,replace=False))
            x=z['X'][ix];y=z['A'][ix];sc=z['S'][ix];ex=z['T'][ix].astype(int)
        gap=_v7_shipping_gap(x.astype('f')*.01);sid=np.where(np.abs(gap)<95,ex,23+2*(ex-21)+(gap>0)).astype('int64')
        for k,v in dict(x=x,y=y,s=sc,sid=sid,mask=np.zeros((len(ix),225),np.uint8),style=np.zeros(len(ix),int),group=np.full(len(ix),r['layout']),expert=sid,domain=np.zeros(len(ix),bool)).items():parts[k].append(v)
    return {k:np.concatenate(v) for k,v in parts.items()}

def master_domain(data):
    delta=_v7_shipping_gap(data['x'].astype('f')*.01)
    data['expert']=np.where(np.abs(delta)<95,27,28+(delta>0)).astype('int64')
    data['domain']=np.ones(len(delta),bool)
    return data

def joint_train(old,full,d,device):
    net=copy.deepcopy(old);embed=torch.cat((old.style.weight.detach(),old.style.weight[[22,25,26]].detach().clone()))
    net.style=nn.Embedding.from_pretrained(embed,freeze=False).to(device)
    trunk=[p for n,p in net.named_parameters() if n!='style.weight']
    opt=torch.optim.AdamW([{'params':trunk,'lr':.000026},{'params':[net.style.weight],'lr':.00115}],weight_decay=.00003)
    rng=np.random.default_rng(991465);anchors=net.style.weight[:27].detach().clone()
    for ep in range(4):
        loss_total=0.;net.train()
        for ix in np.array_split(rng.permutation(len(d['x'])),max(1,len(d['x'])//38)):
            x=torch.tensor(d['x'][ix],device=device).float()*.01;sc=torch.tensor(d['s'][ix],device=device);y=torch.tensor(d['y'][ix],device=device).long()
            m=torch.tensor(d['mask'][ix],device=device).long();where=d['domain'][ix];ex=d['expert'][ix].copy();teacher_ids=d['sid'][ix].copy()
            # Distill original actions on a second mix of nonelite styles without labelling those styles as Master.
            generic=(~where)&(rng.random(len(ix))<.13);teacher_ids[generic]=rng.integers(0,21,sum(generic));ex[generic]=teacher_ids[generic]
            idx=torch.tensor(teacher_ids,device=device);active=torch.tensor(where,device=device);oldlabel=torch.tensor(~where&~generic,device=device)
            scores=np.choose(d['style'][ix],[1.,.75,1.22]).astype('f');master_weight=torch.tensor(scores,device=device)[:,None]
            with torch.no_grad():
                q0,v0=old(x,idx,sc);q1,_=full(x,idx,sc)
            z,v=net(x,torch.tensor(ex,device=device),sc);loss=0.
            for role,(lo,hi) in enumerate(((0,5),(5,8))):
                lp=z[:,:,lo:hi].log_softmax(-1);p0=q0[:,:,lo:hi].log_softmax(-1);p1=q1[:,:,lo:hi].log_softmax(-1)
                eligible=y[:,role]>=0;ce=-lp.gather(2,y[:,role].clamp_min(0)[:,:,None]).squeeze(2)
                priority=(1+.4*((m&2)>0)+.45*((m&4)>0)) if role==0 else 1.
                sup=eligible*(master_weight*active[:,None]*priority+.11*oldlabel[:,None])
                kl0=(p0.exp()*(p0-lp)).sum(-1);kl1=(p1.exp()*(p1-lp)).sum(-1)
                qw=(.018+eligible.float())
                wm=qw*active[:,None];wo=qw*(~active)[:,None]
                loss+=(1. if role==0 else .50)*((ce*sup).sum()/sup.sum().clamp_min(1)+
                      7.0*(kl0*wo).sum()/wo.sum().clamp_min(1)+1.2*(kl1*wm).sum()/wm.sum().clamp_min(1))
            loss+=2.*F.smooth_l1_loss(v,v0,beta=.09)+.005*(net.style.weight[:27]-anchors).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),.8);opt.step();loss_total+=float(loss.detach().cpu())
        print('epoch',ep+1,'loss',round(loss_total,2),flush=True)
    return net.eval()

def main(data,out,weight_path=None):
    torch.set_num_threads(1);torch.manual_seed(991465);dev='mps' if torch.backends.mps.is_available() else 'cpu'
    rows=json.loads((data/'master-worker/index.json').read_text());training=[r for r in rows if r['fold']=='train'];held=[r for r in rows if r['fold']=='test']
    oldtrain,oldhold=archives(data,rows)
    md=master_domain(take_data(data,training,True));olddata=elite_data(oldtrain)
    d={k:np.concatenate([md[k],olddata[k]]) for k in olddata}
    teacher=original().to(dev).eval();guide=original(data/'master-full.py').to(dev).eval();print('training',dev,len(md['x']),len(olddata['x']),len(oldtrain),flush=True)
    model=joint_train(teacher,guide,d,dev);weights=weight_path or HERE/'master_multi_weights.npz';package=io.BytesIO();model.export(package)
    payload=package.getvalue()
    if weights.exists() and weights.read_bytes()!=payload:raise ValueError('Choose a new --weights path for a new fit')
    if not weights.exists():weights.write_bytes(payload)
    code=build(PARENT.read_text(),weights);compile(code,str(out),'exec')
    if out.exists() and out.read_text()!=code:raise ValueError('Choose a new candidate path for a different model')
    if not out.exists():out.write_text(code)
    print('published',out,hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
    meta=dict(source=SHA,candidate=hashlib.sha256(code.encode()).hexdigest(),guidance=hashlib.sha256((data/'master-full.py').read_bytes()).hexdigest(),master_train=[r['group'] for r in training],master_hold=[r['group'] for r in held],old_train=oldtrain,old_hold=oldhold,master_rows=len(md['x']),old_rows=len(olddata['x']),epochs=4,
              design='Joint replay from 42 Master maps and original Artemis/Apollo training-only maps. No Master validation or six held ordinary layouts enter training in either exact or dihedrally equivalent form. Original style anchors use output KL 7.0 and weak demonstration loss, Master rows 27/28/29 receive real behavior and fixed MasterFull output prior. Hyperparameters selected before either holdout. Frozen smooth gate checks real base campers, loaded allies or production disadvantage and reads only public positions and economy.')
    (ROOT/'analysis/v7/master-multi-fit.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path);p.add_argument('--weights',type=Path);a=p.parse_args();main(a.data,a.output,a.weights)
