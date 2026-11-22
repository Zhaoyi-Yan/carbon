"""Test interpolation between shipped v6 and its four-epoch shipment-conditioned refit."""
import argparse
import copy
import json
from pathlib import Path
import re,base64,io,sys,tempfile
import numpy as np
import torch
from torch import nn
from shipping_styles import _v7_shipping_gap

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/v6r'))
import train
from refit import load_full


def base_arrays():
    source=(ROOT/'yan-v6.py').read_text()
    out={}
    for key in ('_WEIGHTS','_VALUE'):
        encoded=re.search(r'^'+key+r' = "([^"]+)"$',source,re.M)[1]
        with np.load(io.BytesIO(base64.b64decode(encoded)),allow_pickle=False) as z:out.update(dict(z))
    out['emb']=out['emb'][[*range(23),21,21,22,22]]
    return out


def mixed(alpha):
    a=base_arrays()
    with np.load(HERE/'shipping.npz',allow_pickle=False) as f:
        return {k:((a[k].astype('f')*(1.-alpha)+f[k].astype('f')*alpha).astype(f[k].dtype)
                   if np.issubdtype(f[k].dtype,np.floating) else f[k]) for k in a}


def network(fields):
    emb=fields['emb'];fields=dict(fields,emb=emb[:21])
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'mixed.npz';np.savez(p,**fields);model=load_full(p)
    model.style=nn.Embedding.from_pretrained(torch.tensor(emb),freeze=True)
    return model.eval()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    torch.set_num_threads(1);device='mps' if torch.backends.mps.is_available() else 'cpu'
    _,va=train.split(a.cache);d=train.load_rows(a.cache,va);t=d['T'].numpy().copy()
    gaps=np.concatenate([_v7_shipping_gap(x.numpy().astype('f')*.01) for x in d['X'].split(128)])
    fresh=(t>=21)&(abs(gaps)>=95.);t[fresh]=23+2*(t[fresh]-21)+(gaps[fresh]>0);d['T']=torch.tensor(t)
    prior=network(mixed(0.)).to(device);rng=np.random.default_rng(42);result=[]
    for alpha in (0.,.35,.65,1.):
        m=network(mixed(alpha)).to(device)
        with torch.no_grad():metrics=train.epoch(m,prior,d,device,rng,None,96,.09,.70)
        result.append(dict(weight=alpha,**metrics));print(alpha,metrics,flush=True)
    a.out.write_text(json.dumps(dict(heldout=va,scores=result),indent=2)+'\n')


if __name__=='__main__':main()
