"""Learn separate elite policy embeddings for positions with a large near-home shipment."""
import argparse
import inspect
import json
from pathlib import Path
import re
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def _v7_shipping_gap(x):
    """Distance-discounted existing cargo from the two legal observation planes."""
    shape=x.shape
    cells=x.reshape(-1,54,225)
    own=np.expm1(7*cells[:,3])*np.maximum(0.,1.-14*cells[:,39]/9.)**1.4
    rival=np.expm1(7*cells[:,6])*np.maximum(0.,1.-14*cells[:,40]/9.)**1.4
    return (own.sum(1)-rival.sum(1)).reshape(shape[:-3])


def _v7_shipping_id(style,gap):
    if style not in (21,22) or abs(gap)<95.:return style
    return 23+2*(style-21)+int(gap>0)


def apply(code,weights):
    location=str(ROOT/'experiments/v5')
    sys.path.insert(0,location)
    try:
        from model_export import split
    finally:
        sys.path.remove(location)
    for key,blob in zip(('_WEIGHTS','_VALUE'),split(weights)):
        code,n=re.subn('^'+key+r' = "[^"]*"$',lambda _:key+' = "'+blob+'"',code,flags=re.M)
        assert n==1
    anchor='def _v5_root(obs,'
    code=code.replace(anchor,inspect.getsource(_v7_shipping_gap)+'\n'+inspect.getsource(_v7_shipping_id)+'\n'+anchor)
    anchor="    state['root_ids'] = [expert,enemy_style]"
    assert code.count(anchor)==1
    code=code.replace(anchor,"    ship_gap=float(_v7_shipping_gap(x))\n    active_expert=_v7_shipping_id(expert,ship_gap)\n    state['root_ids'] = [active_expert,enemy_style]")
    code=code.replace("ids[me],ids[op] = expert,enemy_style","ids[me],ids[op] = active_expert,enemy_style")
    return code.replace('_r6_second_style(x, players, me, 43-expert)','_r6_second_style(x, players, me, _v7_shipping_id(43-expert,ship_gap))')


def main():
    import torch
    from torch import nn
    import copy
    sys.path.insert(0,str(ROOT/'experiments/v6r'))
    import train
    from refit import load_full
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','weights','out','report'):p.add_argument(k,type=Path)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(1729);rng=np.random.default_rng(1729)
    with np.load(a.weights,allow_pickle=False) as z:raw=dict(z)
    emb=raw['emb'];raw['emb']=emb[:21]
    with tempfile.TemporaryDirectory() as tmp:
        file=Path(tmp)/'init.npz';np.savez(file,**raw);model=load_full(file)
    for_names=[*range(23),21,21,22,22]
    model.style=nn.Embedding.from_pretrained(torch.tensor(emb[for_names]),freeze=False)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=model.to(device);prior=copy.deepcopy(model).eval()
    tr,va=train.split(a.cache)
    old=json.loads((ROOT/'experiments/v6r/deep-validation.json').read_text())
    assert tr==old['training'] and va==old['validation']
    def get(names,training):
        data=train.load_rows(a.cache,names)
        ids=data['T'].numpy().copy();elite=ids>=21
        gaps=np.concatenate([_v7_shipping_gap(x.numpy().astype('f')*.01) for x in data['X'].split(128)])
        active=elite & (np.abs(gaps)>=95.)
        ids[active]=23+2*(ids[active]-21)+(gaps[active]>0)
        data['T']=torch.tensor(ids)
        if training:
            include=elite|(rng.random(len(ids))<.14);data={k:v[include] for k,v in data.items()}
        return data
    training=get(tr,True);valid=get(va,False)
    opt=torch.optim.AdamW(model.parameters(),lr=.000060,weight_decay=.00009)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=4,eta_min=.000025)
    report=dict(training=tr,validation=va,training_rows=len(training['T']),heldout_styles={str(t):int((valid['T']==t).sum()) for t in range(21,27)},epochs=[])
    def validate():
        with torch.no_grad():return train.epoch(model,prior,valid,device,rng,None,96,.09,.70)
    best=float('inf')
    for ep in range(5):
        if ep:
            model.train();train.epoch(model,prior,training,device,rng,opt,96,.09,.70);sch.step()
        model.eval();m=validate();report['epochs'].append(m);print(ep,m,flush=True)
        metric=m['nll']+.25*m['value_mae']
        if ep and metric<best:best=metric;model.export(a.out);report['selected']=ep
        a.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
