"""Refit the small closed-loop simulator on actual actions and cash of two elite players."""
import argparse
import base64
import copy
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'experiments/v6r'),str(ROOT/'experiments/v5')]
from train import epoch,load_rows,split
from distill import Student


def student():
    src=(ROOT/'yan-v6.py').read_text()
    blob=re.search(r"_V6R_BYTES = '([^']+)'",src)[1]
    with np.load(io.BytesIO(base64.b64decode(blob)),allow_pickle=False) as f: z=dict(f)
    model=Student(z['stem.b'].size,int(z['meta'][0]))
    def assign(layer,prefix,weight='w',bias='b'):
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(z[prefix+'.'+weight]))
            layer.bias.copy_(torch.tensor(z[prefix+'.'+bias]))
    for n,layer in [('stem',model.stem),('hw',model.workers),('hb',model.bases)]:assign(layer,n)
    for i,layers in enumerate(model.res):
        for n,l in zip(('c1','g','c2'),layers):assign(l,'b%d.%s'%(i,n))
    for n,l in [('v1',model.pool),('v2',model.v2),('v3',model.v3)]:assign(l,n,'weight','bias')
    extended=z['emb'][[*range(21),0,0]]
    model.style=nn.Embedding.from_pretrained(torch.tensor(extended),freeze=False)
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path);p.add_argument('model',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();torch.manual_seed(75119);torch.set_num_threads(1)
    rng=np.random.default_rng(75119);dev='mps' if torch.backends.mps.is_available() else 'cpu'
    tr,va=split(a.cache);dt=load_rows(a.cache,tr);dv=load_rows(a.cache,va)
    keep=(dt['T'].numpy()>=21)|(rng.random(len(dt['T']))<.30)
    dt={k:x[keep] for k,x in dt.items()}
    model=student().to(dev);prior=copy.deepcopy(model).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4,eta_min=.000045)
    scores=[];best=float('inf')
    for i in range(5):
        if i:
            model.train();epoch(model,prior,dt,dev,rng,opt,112,.22,.68);sched.step()
        model.eval()
        with torch.no_grad():m=epoch(model,prior,dv,dev,rng,None,112)
        scores.append(m);print(i,m,flush=True)
        score=m['nll']+.20*m['value_mae']
        if i==0 or score<best:
            best=score;best_ep=i
            if i:model.export(a.model)
        a.report.write_text(json.dumps(dict(training=tr,holdout=va,training_rows=len(dt['T']),scores=scores,selected=best_ep,
            scope='Small rollout model only. The full action network and branch referee are separate.'),indent=2)+'\n')


if __name__=='__main__':main()
