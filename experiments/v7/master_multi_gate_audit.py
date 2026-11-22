"""Measure the frozen blockaded-carrier gate on the independent original-elite games."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import types
import numpy as np
import torch
from master_full import ROOT,HERE,PARENT,original,infer
from master_multi_film import load_film
from shipping_styles import _v7_shipping_gap

def main(data,source):
    torch.set_num_threads(1);dev='mps' if torch.backends.mps.is_available() else 'cpu'
    ordinary=original().to(dev);new=load_film(HERE/'master_multi_film_weights.npz',dev)
    q=types.ModuleType('frozen_mfi_gate');exec(compile(source.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)'),str(source),'exec'),q.__dict__)
    required=json.loads((ROOT/'analysis/v7/master-multi-fit.json').read_text())['old_hold'];names={r['name'] for r in required};files={}
    for folder in (ROOT.parent/'v6r-data/artemis',ROOT.parent/'v5-data',ROOT.parent/'v6-data',ROOT/'replays',ROOT.parent/'v6r-data/latest-followup'):
        for p in folder.rglob('*.json.gz'):
            n=p.name.split('.')[0]
            if n in names:files.setdefault(n,p)
    archive=[];allstats={}
    for r in required:
        name=r['name']
        with gzip.open(files[name]) as f:g=json.load(f)
        fp=ROOT.parent/'v6r-data/boards'/(name+'.npz')
        with np.load(fp) as z:
            keep=np.flatnonzero(z['T']>=21);X=z['X'][keep];S=z['S'][keep];Y=z['A'][keep];t=np.rint(S[:,2]*300).astype(int);st=z['T'][keep].astype(int)
        depots=[tuple(sorted(p[1].values())) for p in g['steps'][0][0]['observation']['players']]
        sides=np.array([depots.index(tuple(np.flatnonzero(x[12].ravel()))) for x in X])
        active=np.zeros(len(X),bool)
        for side in np.unique(sides):
            hist={};flags=[]
            for step,f in enumerate(g['steps']):
                ob=f[0]['observation'];flags.append(q._mm_switch(dict(player=int(side),players=ob['players'],step=step),22,0.,hist)>=27)
            idx=np.flatnonzero(sides==side);active[idx]=np.asarray(flags)[t[idx]]
        gap=_v7_shipping_gap(X.astype('f')*.01);normal=np.where(np.abs(gap)<95,st,23+2*(st-21)+(gap>0));master=np.where(np.abs(gap)<95,27,28+(gap>0))
        for key,ids,net in [('lifeline',normal,ordinary),('gated',np.where(active,master,normal),new)]:
            p,_=infer(net,dict(x=X,s=S,sid=ids),np.arange(len(X)),dev)
            for slice_name,sel in [('all',np.ones(len(X),bool)),('activated',active)]:
                for slot in (0,1):
                    z=p[:,:,:5] if slot==0 else p[:,:,5:];a=Y[:,slot];valid=sel[:,None]&(a>=0);z=z[valid];a=a[valid]
                    if not len(a):continue
                    u=z-z.max(1,keepdims=True);nll=np.log(np.exp(u).sum(1))-u[np.arange(len(a)),a]
                    measure=allstats.setdefault((key,slice_name,slot),[0,0.,0]);measure[0]+=len(a);measure[1]+=nll.sum();measure[2]+=(z.argmax(1)==a).sum()
        archive.append(dict(name=name,layout=r['layout'],timesteps=len(active),active=int(active.sum()),teacher_rows=dict(Counter(map(int,st)))))
    result={}
    for (key,sc,slot),(n,ll,ac) in allstats.items():result.setdefault(key,{}).setdefault(sc,{})[['worker','base'][slot]]=dict(n=n,nll=float(ll/n),hit=float(ac/n))
    result.update(maps=archive,meaning='Six original Artemis/Apollo holdout layouts outside all Master maps. Uses the previously archived elite teacher row 21 or 22, natural routing for transport amounts and the actual blockade gate applied from turn zero to both seats. Report separately for all boards and the gate-active subset.')
    (ROOT/'analysis/v7/master-multi-film-gate.json').write_text(json.dumps(result,indent=2)+'\n');print(result['lifeline'],result['gated'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('source',type=Path);a=p.parse_args();main(a.data,a.source)
