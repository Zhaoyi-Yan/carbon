"""Fine-tune Lifeline's entire existing action/value backbone on public Master games."""
import argparse
import base64
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import re
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dual_weights import arrays,encoded
from shipping_styles import _v7_shipping_gap

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE/'checkpoints/lifeline.py'
SHA='9c5bdda7b8d1a9cc33af4cd0fe03edcb4a34a2d42e027d9b06fed519ff6b949d'
SPEC=importlib.util.spec_from_file_location('frozen_carbon_student',ROOT/'experiments/v5/distill.py')
STUDENT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(STUDENT)
STRENGTH=(.40,.70,1.0)

def original(path=PARENT):
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==SHA
    w=arrays(encoded(path));v=arrays(encoded(path,'_VALUE'));c=w['emb'].shape[1]
    model=STUDENT.Student(c,int(w['meta'][0]));model.style=nn.Embedding.from_pretrained(torch.tensor(w['emb']),freeze=False)
    model.pool=nn.Conv2d(c,32,1);model.v2=nn.Linear(69,128);model.v3=nn.Linear(128,1)
    def put(layer,a,b):
        with torch.no_grad():layer.weight.copy_(torch.from_numpy(a));layer.bias.copy_(torch.from_numpy(b))
    for layer,key in [(model.stem,'stem'),(model.workers,'hw'),(model.bases,'hb')]:
        put(layer,w[key+'.w'],w[key+'.b'])
    for j,(a,b,d) in enumerate(model.res):
        for layer,k in [(a,'c1'),(b,'g'),(d,'c2')]:
            stem=f'b{j}.{k}';put(layer,w[stem+'.w'],w[stem+'.b'])
    for layer,k in [(model.pool,'v1'),(model.v2,'v2'),(model.v3,'v3')]:put(layer,v[k+'.weight'],v[k+'.bias'])
    return model

def scalars(g,side):
    result=[]
    for t,f in enumerate(g['steps']):
        a,b=(f[0]['observation']['players'][j] for j in (side,1-side))
        cargo=lambda p:sum(u[1] for u in p[2].values())
        result.append([(a[0]-b[0])/3000,(cargo(a)-cargo(b))/3000,t/300,(len(a[3])-len(b[3]))/10,(len(a[2])-len(b[2]))/10])
    return np.array(result,np.float32)

def take_data(folder,rows,training):
    parts={k:[] for k in ('x','y','mask','s','style','sid','group','t','rot','map')}
    for nr,r in enumerate(rows):
        with np.load(folder/'master-spatial'/r['cache']) as z:
            t=z['t'];ix=np.arange(len(t));seed=int(r['group'][:8],16);rng=np.random.default_rng(seed)
            if training and len(ix)>100:ix=np.sort(rng.choice(ix,100,replace=False))
            x=z['x'][ix];y=z['y'][ix];m=z['mask'][ix];rot=z['rot'][ix];t=t[ix]
        with gzip.open(r['file']) as fr:g=json.load(fr)
        s=scalars(g,r['side'])[t];delta=_v7_shipping_gap(x.astype('f')*.01)
        base=np.where(rng.random(len(ix))<.28,21,22) if training else np.full(len(ix),22)
        sid=np.where(np.abs(delta)<95,base,23+2*(base-21)+(delta>0)).astype('int64')
        name=r['teacher'][-1] if 'B2' not in r['teacher'] else 'B2'
        for k,v in dict(x=x,y=y,mask=m,s=s,style=np.full(len(ix),{'D':0,'B2':1,'E':2}[name]),sid=sid,group=np.full(len(ix),r['group']),map=np.full(len(ix),nr),t=t,rot=rot).items():parts[k].append(v)
    return {k:np.concatenate(v) for k,v in parts.items()}

def train(guide,rows,ix,dev,epochs=2):
    net=copy.deepcopy(guide);net.train();rng=np.random.default_rng(15611)
    opt=torch.optim.AdamW(net.parameters(),lr=.000034,weight_decay=.00005)
    for ep in range(epochs):
        tic=time.monotonic();seen=0
        for jj in np.array_split(rng.permutation(ix),max(1,len(ix)//34)):
            x=torch.tensor(rows['x'][jj],device=dev).float()*.01
            a=torch.tensor(rows['y'][jj],device=dev).long();sc=torch.tensor(rows['s'][jj],device=dev)
            m=torch.tensor(rows['mask'][jj],device=dev).long()
            ids=rows['sid'][jj].copy();anchor=rng.random(len(jj))<.17;ids[anchor]=rng.integers(0,21,sum(anchor))
            te=torch.tensor(ids,device=dev);labeled=torch.tensor(~anchor,device=dev)[:,None]
            cohorts=torch.tensor(np.choose(rows['style'][jj],[1.,.75,1.22]).astype('f'),device=dev)[:,None]
            with torch.no_grad():old,v0=guide(x,te,sc)
            logits,v=net(x,te,sc);loss=0
            for role,(lo,hi) in enumerate(((0,5),(5,8))):
                valid=a[:,role]>=0;target=a[:,role].clamp_min(0)
                logp=F.log_softmax(logits[:,:,lo:hi],-1);logq=F.log_softmax(old[:,:,lo:hi],-1)
                ce=-logp.gather(-1,target[:,:,None]).squeeze(2)
                importance=(1+.40*((m&2)>0)+.45*((m&4)>0)) if role==0 else 1.
                weights=valid*labeled*cohorts*importance
                kl=(logq.exp()*(logq-logp)).sum(-1)
                kweight=(valid.float()+.025)*torch.where(labeled,1.,4.)
                loss+=(1.0 if role==0 else .42)*((weights*ce).sum()/weights.sum().clamp_min(1)+1.35*(kl*kweight).sum()/kweight.sum().clamp_min(1))
            loss+=1.7*F.smooth_l1_loss(v,v0,beta=.09)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),.80);opt.step();seen+=len(jj)
        if dev=='mps':torch.mps.synchronize()
        print('trained',len(ix),ep+1,seen,round(time.monotonic()-tic,2),flush=True)
    return net

def mix(base,net,share):
    out=copy.deepcopy(base);d={k:torch.lerp(v,net.state_dict()[k],share) for k,v in base.state_dict().items()};out.load_state_dict(d);return out.eval()
def infer(model,d,ix,dev):
    out=[];values=[]
    with torch.inference_mode():
        for ids in np.array_split(ix,max(1,len(ix)//48)):
            x=torch.tensor(d['x'][ids],device=dev).float()*.01;s=torch.tensor(d['s'][ids],device=dev);sid=torch.tensor(d['sid'][ids],device=dev)
            z,v=model(x,sid,s);out.extend(z.cpu().numpy());values.extend(v.cpu().numpy())
    return np.asarray(out),np.asarray(values)
def metrics(d,ix,z,v,basev):
    a=d['y'][ix];m=d['mask'][ix];result={}
    for label,select,role in [('worker',a[:,0]>=0,0),('collector',(a[:,0]>=0)&((m&1)>0),0),('planter',(a[:,0]>=0)&((m&1)==0),0),
                                ('light_enemy_depot',(m&4)>0,0),('loaded_home',(m&2)>0,0),('affordable_base',a[:,1]>=0,1),('actual_hire',a[:,1]>0,1)]:
        if not np.any(select):continue
        q=z[:,:,:5] if role==0 else z[:,:,5:];q=q[select];truth=a[:,role][select]
        maximum=q.max(1,keepdims=True);lp=q-maximum-np.log(np.exp(q-maximum).sum(1,keepdims=True))
        result[label]=dict(n=len(q),nll=round(float(-lp[np.arange(len(q)),truth].mean()),6),hit=round(float(np.mean(q.argmax(1)==truth)),6))
    result['value_drift_mae']=float(np.abs(v-basev).mean());return result
def build(base,weights=HERE/'master_full_weights.npz'):
    if hashlib.sha256(base.encode()).hexdigest()!=SHA:raise ValueError('Expected the frozen Lifeline candidate')
    with np.load(weights) as arc:
        text=base
        for key in ('_WEIGHTS','_VALUE'):
            old=arrays(encoded(base,key));a={k:(arc[k] if k in arc and k not in ('meta','snames') else v) for k,v in old.items()}
            f=io.BytesIO();np.savez_compressed(f,**a);enc=base64.b64encode(f.getvalue()).decode()
            text=re.sub(r'^'+key+r' = "[^"\n]+"$',key+' = "'+enc+'"',text,count=1,flags=re.M)
    return text

def publish(base,model,output,weights):
    model.export(weights)
    text=build(base,weights);compile(text,str(output),'exec');output.write_text(text)
def main(folder,output):
    torch.set_num_threads(1);torch.manual_seed(15611);dev='mps' if torch.backends.mps.is_available() else 'cpu'
    guide=original().to(dev).eval();allrows=json.loads((folder/'master-worker/index.json').read_text())
    trainrows=[r for r in allrows if r['fold']=='train'];hold=[r for r in allrows if r['fold']=='test']
    inner=set(json.loads((ROOT/'analysis/v7/master-spatial-fit.json').read_text())['inner_maps']);assert len(inner)==6
    assert len(trainrows)==42 and len(hold)==13 and not {r['group'] for r in trainrows}&{r['group'] for r in hold}
    d=take_data(folder,trainrows,True);ind=np.arange(len(d['x']));fit=ind[~np.isin(d['group'],list(inner))];val=ind[np.isin(d['group'],list(inner))]
    print('prepared',dev,len(fit),len(val),flush=True)
    student=train(guide,d,fit,dev);_,vbase=infer(guide,d,val,dev);scores=[]
    for share in STRENGTH:
        out=mix(guide,student,share);z,v=infer(out,d,val,dev);r=metrics(d,val,z,v,vbase)
        score=.62*r['worker']['nll']+.16*r['light_enemy_depot']['nll']+.12*r['loaded_home']['nll']+.10*r['affordable_base']['nll']+.3*r['value_drift_mae']
        scores.append(dict(share=share,score=score,metrics=r));del out
    share=min(scores,key=lambda r:r['score'])['share'];print('selected',share,[(r['share'],r['score']) for r in scores],flush=True)
    del student
    net=mix(guide,train(guide,d,ind,dev),share);weight=folder/'master-full-weights.npz';publish(PARENT.read_text(),net,output,weight)
    print('published',output,hashlib.sha256(output.read_bytes()).hexdigest(),flush=True)
    calibration=dict(source_sha=SHA,student_sha=hashlib.sha256(output.read_bytes()).hexdigest(),device=dev,development_rows=len(ind),train_groups=sorted(set(d['group'])),inner_groups=sorted(inner),strength_selection=scores,share=share,train_steps='Two passes, up to 100 legal boards per map. Independent 36/6 calibration on whole maps, then 42-map refit.')
    (ROOT/'analysis/v7/master-full-fit.json').write_text(json.dumps(calibration,indent=2)+'\n')
    del d
    h=take_data(folder,hold,False);report=evaluate_holds(guide,net,h,hold,dev)
    report.update(calibration);(ROOT/'analysis/v7/master-full-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('held',report['cohorts']['all'],flush=True)

def evaluate_holds(guide,net,d,rows,dev):
    # Collapse four legal rotations before measuring engineer-facing, native-cell decisions.
    ix=np.arange(len(d['x']));b,v0=infer(guide,d,ix,dev);q,v=infer(net,d,ix,dev)
    # Use the same authoritative rotation definition as cache preparation.
    src=PARENT.read_text();a=src.index('def _v5_symmetry_maps():');e=src.index('_V5_ROTATIONS = _v5_symmetry_maps()',a)
    scope={'np':np,'NPL':54,'SIZE':15,'N':225};exec(src[a:e],scope);maps=scope['_v5_symmetry_maps']()
    index=[];base=[];pred=[];bv=[];pv=[]
    for i in range(0,len(ix),4):
        ii=np.arange(i,i+4);assert list(d['rot'][ii])==[0,1,2,3] and len(set(d['t'][ii]))==1
        for source,dst in ((b,base),(q,pred)):
            frames=[]
            for j,(cells,_,acts,_) in enumerate(maps):
                raw=source[ii[j],cells];frames.append(np.concatenate((raw[:,:5][:,acts],raw[:,5:]),axis=1))
            dst.append(np.mean(frames,axis=0))
        index.append(i);bv.append(np.mean(v0[ii]));pv.append(np.mean(v[ii]))
    index=np.array(index);base=np.array(base);pred=np.array(pred);bv=np.array(bv);pv=np.array(pv)
    def compare(mask):
        i=index[mask];r0=metrics(d,i,base[mask],bv[mask],bv[mask]);r1=metrics(d,i,pred[mask],pv[mask],bv[mask])
        wt=d['y'][i,0]>=0;bt=d['y'][i,1]>=0
        changes={k:float(np.mean(a[sel].argmax(1)!=b[sel].argmax(1))) for k,a,b,sel in [('worker',base[mask,:,:5],pred[mask,:,:5],wt),('base',base[mask,:,5:],pred[mask,:,5:],bt)] if sel.any()}
        confusion={}
        truth=d['y'][i,1][bt]
        for tag,logits in [('lifeline',base[mask,:,5:]),('full',pred[mask,:,5:])]:
            guessed=logits[bt].argmax(1)
            confusion[tag]=np.bincount(truth*3+guessed,minlength=9).reshape(3,3).tolist()
        return {'lifeline':r0,'full':r1,'decision_change':changes,'base_confusion':confusion}
    out={key:compare(np.ones(len(index),bool) if n is None else d['style'][index]==n) for key,n in [('all',None),('D',0),('B2',1),('E',2)]}
    return dict(cohorts=out,held=[dict(match=r['match'],group=r['group'],teacher=r['teacher'],seat=r['side'],seed=r['seed'],scores=compare(d['map'][index]==i)) for i,r in enumerate(rows)],states=len(index))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.data,a.output)
