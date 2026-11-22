"""Train small orientation-preserving adapters without changing the frozen convolutional trunk."""
import argparse,hashlib,json,os,time
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
import master_worker as mw
from master_fit import predict as old_predict
from master_data import NAMES
from master_worker_data import ROOT
HERE=Path(__file__).resolve().parent
ROT=np.stack([[0]+[1+(i+j)%4 for i in range(4)] for j in range(4)])
mw._V5_ROTATIONS=[(None,None,a,None) for a in ROT]
KEYS=('h','l','k','r','y','base','t','heavy','pressure','rich','home','press')


def load_rows(rows,d,cache):
    cohort=[]
    for r in rows:
        with np.load(d/r['cache']) as f:c={k:f[k] for k in KEYS+('gs','tgs')}
        teacher=NAMES.index(r['teacher'])
        with np.load(cache/r['cache']) as z:
            assert np.array_equal(c['y'],z['wy']) and np.array_equal(c['t'],z['wt'])
            old=np.zeros_like(c['base'],dtype='f')
            for role in (0,1):
                ii=c['r']==role
                if ii.any():old[ii]=old_predict((OLD['w'+str(role)],OLD['s'+str(role)]),z['wx'][ii].astype('f'),np.full(ii.sum(),teacher))
        c.update(row=r,style=np.full(len(c['r']),teacher),old=old)
        cohort.append(c)
    return cohort


def normals(train):
    center=np.concatenate([v['h'][::3,:,0].astype('f').reshape(-1,96) for v in train])
    diff=np.concatenate([(v['h'][::5].astype('f')-v['h'][::5,:,0:1]).reshape(-1,96) for v in train])
    local=np.concatenate([v['l'] for v in train]).astype('f')
    return dict(cmean=center.mean(0),cscale=center.std(0).clip(.25,None),dscale=diff.std(0).clip(.3,None),
                lmean=local.mean(0),lscale=local.std(0).clip(.20,None))


class Head(nn.Module):
    def __init__(self,normal):
        super().__init__()
        for k,v in normal.items():self.register_buffer(k,torch.from_numpy(v))
        for k,shape in dict(cw=(96,24),lw=(20,24),hb=(24,),dirs=(24,5),dw=(96,10),kw=(10,10),db=(10,),dv=(10,),
                            sty_dirs=(24,15),role=(2,10),temperature=(2,)).items():
            p=nn.Parameter(torch.zeros(shape));setattr(self,k,p)
            if k in ('cw','lw','dw','kw'):nn.init.normal_(p,std=.05)
            if k=='temperature':nn.init.ones_(p)
    def forward(self,raw,local,context,role,base,style):
        raw=raw.float();local=local.float();context=context.float();base=base.float()
        center=raw[:,:,0]
        h=torch.tanh((center-self.cmean)/self.cscale@self.cw+((local-self.lmean)/self.lscale)[:,None]@self.lw+self.hb)
        direct=h@self.dirs
        direct=torch.stack([direct[:,j,a] for j,a in enumerate(ROT)],1)
        d=torch.tanh((raw-center[:,:,None])/self.dscale@self.dw+context[:,None]@self.kw+self.db)
        spec=(h@self.sty_dirs).reshape(len(h),4,3,5)
        spec=torch.stack([spec[:,j][:,:,a] for j,a in enumerate(ROT)],1)
        corr=(direct+d@self.dv+(spec*style[:,None,:,None]).sum(2)).mean(1)
        corr+=torch.einsum('naf,nf->na',context,self.role[role])
        return base*self.temperature[role,None]+corr.clamp(-4.,4.)


def packed(ds,keys):
    return {k:torch.from_numpy(np.concatenate([v[k] for v in ds])) for k in keys}


def learn(train,dev,epochs):
    torch.manual_seed(86371);net=Head(normals(train));optim=torch.optim.AdamW(net.parameters(),lr=.0025,weight_decay=.035)
    order=('h','l','k','r','base')
    allk=order+('style','y','heavy','pressure','rich')
    data=packed(train,allk);valid=packed(dev,allk) if dev else None
    best=(1.e9,0);history=[]
    for ep in range(epochs):
        net.train();rng=torch.Generator().manual_seed(209+ep)
        for ii in torch.randperm(len(data['y']),generator=rng).split(256):
            style=F.one_hot(data['style'][ii],3).float()
            # Shrink source labels toward a common style so the nine 7sE training maps do not dominate.
            style=.9*style+.1*torch.tensor([.45,.20,.35])
            logits=net(*(data[k][ii] for k in order),style)
            weight=(1.+.45*data['heavy'][ii]+.42*data['pressure'][ii]+.30*data['rich'][ii])*torch.tensor([1.0,.65,1.30])[data['style'][ii]]
            loss=(F.cross_entropy(logits,data['y'][ii],reduction='none')*weight).sum()/weight.sum()
            loss+=.035*net.sty_dirs.square().mean()+.012*net.role.square().mean()+.008*net.dirs.square().mean()
            optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step()
        if valid:
            with torch.inference_mode():
                losses=[]
                for ii in torch.arange(len(valid['y'])).split(512):
                    z=net(*(valid[k][ii] for k in order),F.one_hot(valid['style'][ii],3).float())
                    losses.extend(F.cross_entropy(z,valid['y'][ii],reduction='none').tolist())
                score=np.mean(np.array(losses)[valid['style'].numpy()!=1]);history.append(round(float(score),5))
            if score<best[0]:best=score,ep+1
    return {k:v.detach().numpy() for k,v in net.state_dict().items()},best[1],history


def router(train):
    X=np.concatenate([c['gs'] for c in train if c['style'][0]!=1]).astype('f')
    ys=np.concatenate([np.full(len(c['gs']),c['style'][0]==2) for c in train if c['style'][0]!=1])
    w=np.concatenate([np.full(len(c['gs']),50/len(c['gs'])) for c in train if c['style'][0]!=1])
    mu=X.mean(0);sd=X.std(0).clip(.1,None);model=LogisticRegression(C=.06,class_weight='balanced').fit((X-mu)/sd,ys,sample_weight=w)
    return dict(gate_mean=mu,gate_scale=sd,gate_w=model.coef_[0].astype('f'),gate_b=np.float32(model.intercept_[0]))


def metrics(v,z,mask):
    if not mask.any():return None
    pred=z.argmax(1);y=v['y'];n=np.arange(len(y));lp=log_softmax(z,axis=1)
    return [int(mask.sum()),float(-lp[n,y][mask].sum()),int((pred[mask]==y[mask]).sum())]


def evaluate(v,model,style_id):
    r=v['r'];ys=v['y'];base=v['base'].astype('f')
    g=v['gs'].astype('f');p=mw._mw_router(g,model);indices=np.searchsorted(v['tgs'],v['t'])
    mix=np.column_stack([1-p,np.zeros_like(p),p])[indices].astype('f')
    fixed=np.eye(3,dtype='f')[np.full(len(r),style_id)]
    out={}
    for name,styles in [('worker_routed',mix),('worker_oracle',fixed)]:
        by=[]
        for i in range(0,len(r),192):
            sl=slice(i,i+192)
            by.append(mw._mw_logits(v['h'][sl].astype('f'),v['l'][sl].astype('f'),v['k'][sl].astype('f'),r[sl],base[sl],styles[sl],model))
        out[name]=np.concatenate(by)
    out['frozen_root']=base
    if 'old' in v:out['prior_linear_oracle']=v['old'].astype('f')
    masks={'all':r>=0,'collectors':r==0,'planters':r==1,'heavy_near_home':v['heavy'],'light_pressure':v['pressure'],'rich_ore':v['rich']}
    scopes={}
    for scope,mask in masks.items():
        if not mask.any():continue
        label=v['home'] if scope=='heavy_near_home' else (v['press'] if scope=='light_pressure' else np.tile([True,False,False,False,False],(len(r),1)))
        scope_out={}
        for name,z in out.items():
            data=metrics(v,z,mask)
            if scope in ('heavy_near_home','light_pressure','rich_ore'):
                n=np.arange(len(r));truth=label[n,ys][mask];guess=label[n,z.argmax(1)][mask]
                data+=[int(truth.sum()),int(guess.sum()),int((truth&guess).sum())]
            scope_out[name]=data
        scopes[scope]=scope_out
    root=out['frozen_root'].argmax(1);adapt=out['worker_routed'].argmax(1)
    turns=[]
    for t in v['tgs']:
        ii=v['t']==t
        turns.append([int(t),int((root[ii]!=adapt[ii]).sum()),int((adapt[ii]==ys[ii]).sum()-(root[ii]==ys[ii]).sum())])
    return scopes,dict(mean_E=float(np.mean(p)),D_E_correct=int(((p>=.5)==bool(style_id==2)).sum()),n=len(p)),turns


def combine(rows):
    merged={}
    for row in rows:
        scope=row['raw']
        for k,models in scope.items():
            for name,vals in models.items():
                group=merged.setdefault(k,{});group[name]=np.add(group.get(name,np.zeros(len(vals))),vals)
    out={}
    for k,models in merged.items():
        out[k]={}
        for name,values in models.items():
            n,loss,ok,*task=values
            metrics_=dict(units=int(n),nll=round(float(loss/n),5),hit=round(float(ok/n),5))
            if task:
                truth,pred,overlap=task
                metrics_.update(action_frequency=round(truth/n,4),predicted_frequency=round(pred/n,4),
                                precision=round(overlap/max(1,pred),4),recall=round(overlap/max(1,truth),4))
            out[k][name]=metrics_
    return out


def main():
    global OLD
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);OLD=dict(np.load(HERE/'master-model.npz'))
    rows=json.loads((a.data/'index.json').read_text());train=load_rows([r for r in rows if r['fold']=='train'],a.data,a.cache)
    assert len(train)==42
    dev=[]
    for style in NAMES:
        same=sorted([r for r in train if r['row']['teacher']==style],key=lambda c:hashlib.sha256(('directional-v1'+c['row']['group']).encode()).digest())
        dev+=same[:max(2,len(same)//5)]
    devkeys={v['row']['group'] for v in dev}
    _,epochs,history=learn([v for v in train if v['row']['group'] not in devkeys],dev,18);print('selected',epochs,history,flush=True)
    model,_,_=learn(train,[],epochs);model.update(router(train))
    np.savez_compressed(a.model,**model)
    test=load_rows([r for r in rows if r['fold']=='test'],a.data,a.cache);assert len(test)==13
    assert {r['row']['group'] for r in train}.isdisjoint(r['row']['group'] for r in test)
    output=[]
    for v in test:
        r=v['row'];raw,gate,turns=evaluate(v,model,NAMES.index(r['teacher']))
        output.append(dict(match=r['match'],seat=r['side'],seed=r['seed'],teacher=r['teacher'],layout=r['group'],raw=raw,gate=gate,turns=turns))
    with np.load(a.data/'target.npz') as z:target={k:z[k] for k in z}
    r,g,t=evaluate(target,model,2);target_row=dict(raw=r,gate=g,turns=t)
    summary={key:combine([v for v in output if key=='all' or v['teacher'] in key.split('/')]) for key in ('all','Master7sD','Master7sE','Master7sB2','Master7sD/Master7sE')}
    archive=dict(train_maps=42,inner_development_maps=sorted(devkeys),validation_maps=13,
                 training_ids=[v['row']['match'] for v in train],epochs=epochs,inner_dev_nll=history,weights_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),model_bytes=a.model.stat().st_size,
                 scopes='Heavy: seated collector load>=65 within four of own home; action moves closer. Pressure: load<=15 within four of rival depot or three of a collector with at least 35 more cargo; action moves closer to either objective. Rich: estimated soil>=45, no tree/depot; action is stay. Precision/recall in each applicable group are separate from exact move.',
                 metrics=summary,validation=output,target=dict(match='4063a34afedb6d4767a9118e1dee9e5d',teacher='Master7sE',seat=0,gate=g,metrics=combine([target_row]),turns=t),
                 limitations='Original Master cache shares the exact same 42/13 split. The trunk used older public pretraining so independence is for the new adapter. Oracle supplies true D/B2/E identity; routed mode does not. No win rate inferred from imitation.')
    a.output.write_text(json.dumps(archive,indent=1)+'\n')
    print('all',summary['all']['all'],'DE',summary['Master7sD/Master7sE']['all'],flush=True)

if __name__=='__main__':main()
