"""Fit residual logits for the online v5's searched moves, using the other player's visibility."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine,limit_math_threads,load_strategy

V5='ec91ff24ea3af3510fe28c948b5d7da3'


def collect(model,game,target):
    frames=game['steps'];observer=1-target;tracker=model.BeliefTracker(model._Bel,observer)
    last=None;inputs=[];positions=[]
    for i,(current,following) in enumerate(zip(frames,frames[1:])):
        board=current[0]['observation'];pl=board['players']
        est,known=tracker.step(pl,i,current[observer]['observation']['carbon'])
        if i%3==0 or i<12:
            inputs.append(model._quant(model.encode(pl,i,target,est,[1.]*225,last)))
            actions=following[target]['action'] or {}
            rows=[(pos,int(kind=='PLANTER'),model.WIDX[actions.get(u,'NONE')]) for u,(pos,_,kind) in pl[target][2].items()]
            if pl[target][0]>=30 and len(pl[target][2])<10:
                rows.extend((pos,2,model.BIDX[actions.get(u,'NONE')]) for u,pos in pl[target][1].items())
            positions.append(rows)
        tracker.set_actions(pl,following[observer]['action']);last=pl
    samples=[]
    for k in range(0,len(inputs),16):
        h=model._net()(np.asarray(inputs[k:k+16]),20,hidden=True)
        for ix,t in enumerate(positions[k:k+16]):
            for p,role,y in t:samples.append((h[ix,p],role,y))
    return samples


def fit(train,valid,base):
    def unpack(rows):
        return torch.from_numpy(np.stack([r[0] for r in rows])).float(),torch.tensor([r[2] for r in rows],dtype=torch.long)
    x,y=unpack(train);u,v=unpack(valid)
    w0,b0=map(lambda z:torch.from_numpy(z.copy()).float(),base)
    # Penalize the change in predicted logits; the memory only has 68 games.
    scale=x.square().mean(0).sqrt().clamp_min(.2)
    x0=x@w0.T+b0;u0=u@w0.T+b0
    w=torch.zeros_like(w0,requires_grad=True);b=torch.zeros_like(b0,requires_grad=True)
    opt=torch.optim.LBFGS([w,b],lr=.7,max_iter=35,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();d=(x/scale)@w.T+b
        loss=F.cross_entropy(x0+d,y)+.024*d.square().mean()+.013*w.square().sum()/w.shape[0]
        loss.backward();return loss
    opt.step(closure)
    wf=(w/scale).detach();bf=b.detach();uf=u0+u@wf.T+bf
    def metrics(z):return dict(nll=float(F.cross_entropy(z,v)),actions=len(v),accuracy=float((z.argmax(-1)==v).float().mean()))
    return wf.numpy(),bf.numpy(),dict(original=metrics(u0),fitted=metrics(uf))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','raw','model','report'):p.add_argument(name,type=Path)
    a=p.parse_args();configure_engine();limit_math_threads();torch.set_num_threads(1)
    m=load_strategy('r6_prediction_trunk',a.source)
    data=[r for r in json.loads((a.raw/'index.json').read_text()) if any(z['id']==V5 for z in r['strategies'].values())]
    data.sort(key=lambda r:r['finished'],reverse=True)
    tr=[];va=[];names=[[],[]]
    for idx,r in enumerate(data):
        key=next(i for i,k in enumerate(r['order']) if r['strategies'][k]['id']==V5)
        g=json.loads(gzip.decompress((a.raw/r['file']).read_bytes()))
        held=idx<14;names[int(held)].append(r['file'])
        (va if held else tr).extend(collect(m,g,key))
        if idx%15==0:print(idx,flush=True)
    weights=np.zeros((3,5,96),dtype='f');biases=np.zeros((3,5),dtype='f')
    result={}
    for role in range(3):
        k='hw' if role<2 else 'hb';n=5 if role<2 else 3
        w,b,metrics=fit([r for r in tr if r[1]==role],[r for r in va if r[1]==role],(m._POLICY.p[k+'.w'].reshape(n,-1),m._POLICY.p[k+'.b']))
        weights[role,:n]=w;biases[role,:n]=b
        result[['collector','planter','base'][role]]=metrics
    np.savez_compressed(a.model,weights=weights,biases=biases)
    a.report.write_text(json.dumps(dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),training=names[0],validation=names[1],metrics=result),indent=2)+'\n')
    print(result,flush=True)

if __name__=='__main__':main()
