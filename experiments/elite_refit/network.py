"""Unpack and refit the two networks used by frozen v7, preserving their native formats."""
import base64
import io
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/v5'))
from distill import Student

PARENT=ROOT/'yan-v7.py'
# New IDs follow the frozen rank order; retain all native style embeddings.
INITIAL=(22,21,20,21,20,20,20,20,20,20)
START=27


def read_blob(source,key):
    pat=r'''(?m)^(\s*'''+key+r'''\s*=\s*)(['"])([^'"\n]*)\2$'''
    found=re.search(pat,source)
    if not found:raise ValueError(key)
    with np.load(io.BytesIO(base64.b64decode(found[3])),allow_pickle=False) as z:
        return dict(z)


def make(source,small,extended=False):
    raw=read_blob(source,'_V6R_BYTES' if small else '_WEIGHTS')
    if not small:raw.update(read_blob(source,'_VALUE'))
    w=raw['stem.b'].size
    m=Student(w,int(raw['meta'][0]))
    m.pool=nn.Conv2d(w,len(raw['v1.bias']),1)
    m.v2=nn.Linear(raw['v2.weight'].shape[1],len(raw['v2.bias']))
    m.v3=nn.Linear(len(raw['v2.bias']),1)
    def assign(layer,key,k='w',b='b'):
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(raw[key+'.'+k]))
            layer.bias.copy_(torch.from_numpy(raw[key+'.'+b]))
    for layer,k in ((m.stem,'stem'),(m.workers,'hw'),(m.bases,'hb')):assign(layer,k)
    for i,layers in enumerate(m.res):
        for l,k in zip(layers,('c1','g','c2')):assign(l,f'b{i}.{k}')
    for l,k in ((m.pool,'v1'),(m.v2,'v2'),(m.v3,'v3')):assign(l,k,'weight','bias')
    emb=raw['emb']
    if not extended:
        if small:emb=emb[[*range(23),21,21,22,22]]
        assert len(emb)==START
        emb=np.concatenate([emb,emb[list(INITIAL)]])
    assert len(emb)==START+len(INITIAL)
    m.style=nn.Embedding.from_pretrained(torch.from_numpy(emb),freeze=False)
    return m


def replace_blob(text,key,data):
    buf=io.BytesIO()
    np.savez_compressed(buf,**data)
    encoded=base64.b64encode(buf.getvalue()).decode()
    pat=r'''(?m)^(\s*'''+key+r'''\s*=\s*)(['"])([^'"\n]*)\2$'''
    text,n=re.subn(pat,lambda m:m[1]+m[2]+encoded+m[2],text)
    assert n==1
    return text


def pack(out,arm,centers,scale):
    src=PARENT.read_text()
    with np.load(out/(arm+'-main.npz')) as z:
        for key in ('_WEIGHTS','_VALUE'):
            keys=read_blob(src,key)
            src=replace_blob(src,key,{k:z[k] if k!='snames' else keys[k] for k in keys})
    with np.load(out/(arm+'-small.npz')) as z:src=replace_blob(src,'_V6R_BYTES',dict(z))
    old='    active_expert=_v7_shipping_id(expert,ship_gap)'
    new='''    # Select the two teacher profiles closest to the current public economic state.
    descriptor=_scalars(players,step,me)
    distances=(((descriptor-_ELITE_CENTERS)/_ELITE_SCALE)**2).sum(axis=1)
    order_teachers=np.argsort(distances)
    active_expert=27+int(order_teachers[0])'''
    alt='_r6_second_style(x, players, me, _v7_shipping_id(43-expert,ship_gap))'
    small='return np.where(ids>=23,21+(ids-23)//2,ids)'
    for anchor in (old,alt,small):assert src.count(anchor)==1
    src=src.replace(old,new).replace(alt,'_r6_second_style(x, players, me, 27+int(order_teachers[1]))')
    src=src.replace(small,'return ids')
    src=src.replace('def _v7_shipping_gap(x):',f'_ELITE_CENTERS=np.array({centers.tolist()!r},dtype=np.float32)\n_ELITE_SCALE=np.array({scale.tolist()!r},dtype=np.float32)\n\ndef _v7_shipping_gap(x):')
    path=out/(arm+'.py')
    compile(src,str(path),'exec')
    assert len(src.encode())<=10_000_000
    path.write_text(src)
    return path


if __name__=='__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description='Rebuild a standalone candidate from exported weights.')
    parser.add_argument('directory',type=Path)
    parser.add_argument('arm',choices=('top6','top6-plus'))
    args=parser.parse_args()
    params=json.loads((args.directory/'router.json').read_text())
    print(pack(args.directory,args.arm,np.array(params['centers']),np.array(params['scale'])))
