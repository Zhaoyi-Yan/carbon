"""Fit a residual multinomial reply model and grade public actions on disjoint maps."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from reply_features import _reply_features

ROOT=Path(__file__).resolve().parents[2]


def get_policy():
    path=ROOT/'yan-v5.py'
    spec=importlib.util.spec_from_file_location('reply_teacher',path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def collect(model,replay,stride):
    frames=replay['steps']
    output=[]
    for observer in range(2):
        target=1-observer
        tracker=model.BeliefTracker(model._Bel,observer)
        model._ST.clear()
        prev=None
        for t,(before,after) in enumerate(zip(frames,frames[1:])):
            full=before[0]['observation']
            pl=full['players']
            carbon=before[observer]['observation']['carbon']
            est,known=tracker.step(pl,t,carbon)
            if t>=10 and (t-10)%stride==0 and pl[target][2]:
                contact=[]
                for uid,unit in pl[target][2].items():
                    features=_reply_features(pl,target,uid,prev,model.FEAT_DIST,model._ROLLOUT_LANDING)
                    if features is not None:
                        contact.append((uid,unit,features))
                if contact:
                    obs={'step':t,'player':observer,'players':pl,'trees':full.get('trees',{}),'carbon':carbon}
                    x=model._quant(model.encode(pl,t,observer,est,known,prev))
                    model._v5_root(obs,est,x,prev)
                    w,_=model._ST['v5_history']['opponent']
                    for uid,(pos,_,kind),features in contact:
                        logit=w[pos]/model._TEMP
                        label=model.WIDX[(after[target]['action'] or {}).get(uid) or 'NONE']
                        feat=np.column_stack((logit-logit.mean(),features-features.mean(0)))
                        output.append((feat,label,int(kind=='PLANTER')))
            tracker.set_actions(pl,after[observer]['action'] or {})
            prev=pl
    return output


def train(x,y):
    count=len(x)
    def objective(w):
        z=x@w
        z-=z.max(1,keepdims=True)
        p=np.exp(z);p/=p.sum(1,keepdims=True)
        loss=-np.log(p[np.arange(count),y]).mean()+.012*(w[1:]@w[1:])
        p[np.arange(count),y]-=1
        grad=(x*p[:,:,None]).sum((0,1))/count
        grad[1:]+=.024*w[1:]
        return loss,grad
    w=np.zeros(x.shape[-1]);w[0]=1.
    return minimize(objective,w,jac=True,method='L-BFGS-B',bounds=[(.15,1.4)]+[(None,None)]*(len(w)-1)).x


def metrics(x,y,w):
    z=x@w
    z-=z.max(1,keepdims=True)
    p=np.exp(z);p/=p.sum(1,keepdims=True)
    return {'n':len(y),'accuracy':float(np.mean(p.argmax(1)==y)),'nll':float(-np.log(p[np.arange(len(y)),y]).mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',nargs='+',type=Path);p.add_argument('--per-source',type=int,default=14)
    p.add_argument('--stride',type=int,default=19);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=get_policy();m._torch.set_num_threads(1)
    seen=set();chosen={};metadata=[]
    for folder in a.source:
        selected=0
        for path in sorted((folder/'replays').glob('*.json.gz'),key=lambda p:hashlib.sha256(p.name.encode()).digest()):
            if path.name in seen:continue
            seen.add(path.name)
            with gzip.open(path) as f:r=json.load(f)
            if len(r['steps'])<45:continue
            first=r['steps'][0][0]['observation']
            fingerprint=hashlib.sha256(json.dumps([first['full_carbon'],[pl[1] for pl in first['players']]],sort_keys=True).encode()).hexdigest()
            if fingerprint in chosen:continue
            chosen[fingerprint]=collect(m,r,a.stride)
            metadata.append({'file':path.name,'seed':r['configuration']['randomSeed'],'map':fingerprint,'source':folder.name})
            selected+=1
            if selected>=a.per_source:break
    partition={k:('test' if i%4==0 else 'train') for i,k in enumerate(sorted(chosen))}
    def arrays(part,role):
        rows=[(x,y) for k,vs in chosen.items() if partition[k]==part for x,y,r in vs if r==role]
        return np.stack([v[0] for v in rows]),np.array([v[1] for v in rows])
    result={'maps':[dict(g,split=partition[g['map']]) for g in metadata],'stride':a.stride,
            'teacher_sha256':hashlib.sha256((ROOT/'yan-v5.py').read_bytes()).hexdigest(),'roles':[]}
    for role in range(2):
        x,y=arrays('train',role); xt,yt=arrays('test',role)
        w=train(x,y);base=np.eye(1,len(w))[0];temp=base.copy()
        fitted=train(x[:,:,:1],y)
        temp[0]=fitted[0]
        result['roles'].append({'weights':w.tolist(),'train':len(y),'heldout':{'frozen':metrics(xt,yt,base),
                        'temperature':metrics(xt,yt,temp),'tactical':metrics(xt,yt,w)}})
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print([(r['train'],r['heldout']) for r in result['roles']],flush=True)


if __name__=='__main__':main()
