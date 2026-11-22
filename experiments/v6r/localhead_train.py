"""Select residual-head epochs on train-only maps, refit and evaluate the untouched elite map panel."""
import argparse,base64,hashlib,json,re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[2]


class LocalHead(nn.Module):
    def __init__(self,channels,extra):
        super().__init__()
        self.a=nn.Linear(channels,24);self.b=nn.Linear(channels,24,bias=False)
        self.c=nn.Linear(extra,24,bias=False);self.out=nn.Linear(24,1,bias=False)
        nn.init.zeros_(self.out.weight)
        self.scale=nn.Parameter(torch.ones(()))

    def forward(self,a,b,c,z):
        h=self.a(a)[:,None,:]+self.b(b)+self.c(c)
        d=self.out(h.tanh())[:,:,0]
        return z*self.scale.clamp(.35,1.8)+d-d.mean(1,keepdim=True)


def metric(logits,y):
    l=torch.from_numpy(np.ascontiguousarray(logits)).float();y=torch.from_numpy(np.ascontiguousarray(y)).long()
    return {'count':len(y),'nll':round(float(F.cross_entropy(l,y)),6),
            'accuracy':round(float((l.argmax(1)==y).float().mean()),6)}


def norm(train,mask,rng):
    ix=np.flatnonzero(mask);ix=rng.choice(ix,size=min(6000,len(ix)),replace=False)
    means=[];stds=[]
    for k in 'abc':
        a=train[k][ix].astype(np.float32).reshape(-1,train[k].shape[-1])
        means.append(a.mean(0,dtype=np.float64).astype(np.float32))
        stds.append(np.maximum(.16,a.std(0,dtype=np.float64)).astype(np.float32))
    return means,stds


def as_tensors(data,mask,scaler):
    mu,sd=scaler
    raw={k:torch.from_numpy(((data[k][mask].astype(np.float32)-mu[i])/sd[i])) for i,k in enumerate('abc')}
    raw['z']=torch.from_numpy(data['z'][mask].astype(np.float32))
    raw['y']=torch.from_numpy(data['y'][mask].astype(np.int64))
    return raw


def fitting(x,indices,max_epochs,internal=None):
    torch.manual_seed(58943)
    model=LocalHead(x['a'].shape[-1],x['c'].shape[-1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.003)
    rng=np.random.default_rng(80523);quality=[]
    for ep in range(1,max_epochs+1):
        model.train();order=rng.permutation(indices)
        for start in range(0,len(order),512):
            ids=order[start:start+512]
            score=model(*[x[k][ids] for k in 'abcz'])
            loss=F.cross_entropy(score,x['y'][ids])+.004*((score-x['z'][ids])**2).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if internal is not None:
            with torch.inference_mode():
                out=predict(model,x,internal)
                quality.append(metric(out,x['y'][internal].numpy()))
    return model,quality


def predict(model,x,ix):
    with torch.inference_mode():
        return np.concatenate([model(*[x[k][ind] for k in 'abcz']).numpy() for ind in np.array_split(ix,max(1,len(ix)//1024))])


def export(model,scaler):
    mu,sd=scaler
    p=[layer.weight.detach().numpy().T.copy()/std[:,None] for layer,std in zip((model.a,model.b,model.c),sd)]
    bias=model.a.bias.detach().numpy().copy()
    for mean,w in zip(mu,p):bias-=mean@w
    return dict(wa=p[0],wb=p[1],wc=p[2],bias=bias,wo=model.out.weight.detach().numpy().T.copy(),scale=float(model.scale.detach().clamp(.35,1.8)))


def read(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)


def train_main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('cache',type=Path)
    p.add_argument('model',type=Path);p.add_argument('report',type=Path);p.add_argument('--max-epochs',type=int,default=12)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);rng=np.random.default_rng(86201)
    trunk=base64.b64decode(re.search(r'^_WEIGHTS = "([^"]+)"',a.parent.read_text(),re.M)[1])
    fingerprint=hashlib.sha256(trunk).hexdigest();train=read(a.cache/'train0.npz')
    tests=[read(a.cache/f'test{i}.npz') for i in range(4)]
    result={'backbone_sha256':fingerprint,'parent_sha256':hashlib.sha256(a.parent.read_bytes()).hexdigest(),
            'split':json.loads((ROOT/'experiments/v6r/elite-validation.json').read_text())['validation'],
            'fit_styles':sorted(set(train['style'].tolist())),'training_maps':len(set(train['game'].tolist())),
            'roles':{}}
    collected=[]
    for kind,name in enumerate(('collector','planter')):
        mask=train['role']==kind;games=train['game'][mask];ix=np.arange(len(games));inner=games%7==0
        interim=norm(train,mask & (train['game']%7!=0),rng)
        x=as_tensors(train,mask,interim)
        _,qe=fitting(x,ix[~inner],a.max_epochs,ix[inner]);chosen=1+np.argmin([v['nll'] for v in qe])
        del x
        normal=norm(train,mask,rng);x=as_tensors(train,mask,normal)
        model,_=fitting(x,ix,int(chosen));state=export(model,normal);collected.append(state)
        # The temperature comparator uses the same train-only data as the fitted local head.
        fitted=minimize_scalar(lambda q:float(F.cross_entropy(q*x['z'],x['y'])),bounds=(.35,1.8),method='bounded').x
        del x
        raw=[];refined=[];single={};tmask=tests[0]['role']==kind;y=tests[0]['y'][tmask]
        for i,part in enumerate(tests):
            assert np.array_equal(part['y'][tmask],y)
            tb=as_tensors(part,tmask,normal)
            out=predict(model,tb,np.arange(len(y)))[:,part['acts']]
            baseline=part['z'][tmask].astype(np.float32)[:,part['acts']]
            if i==0:
                single={'frozen':metric(baseline,y),'temperature':metric(baseline*fitted,y),'head':metric(out,y)}
            raw.append(baseline);refined.append(out)
        raw=np.mean(raw,axis=0);refined=np.mean(refined,axis=0)
        fg=tests[0]['game'][tmask];panel=[]
        for g in np.unique(fg):
            sel=fg==g
            panel.append({'game':result['split'][g],'frozen':metric(raw[sel],y[sel]),'head':metric(refined[sel],y[sel])})
        result['roles'][name]=dict(training_actions=int(mask.sum()),epoch=int(chosen),inner=qe,training_calibration_scale=float(fitted),
                                   native=single,root_rotation_mean={'frozen':metric(raw,y),'temperature':metric(raw*fitted,y),'head':metric(refined,y)},
                                   validation_games=panel)
        print(name,chosen,result['roles'][name]['root_rotation_mean'],flush=True)
    packed={k:np.asarray([s[k] for s in collected],dtype=np.float32) for k in collected[0]}
    with a.model.open('xb') as f:np.savez_compressed(f,**packed,backbone=fingerprint)
    with a.report.open('x') as f:json.dump(result,f,indent=2)


if __name__=='__main__':train_main()
