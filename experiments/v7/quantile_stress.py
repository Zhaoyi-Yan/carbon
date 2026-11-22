"""Evaluate the opening gate on an independent, deliberately difficult cohort of delayed-spawn elite opponents."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import quantile as qt
from distill_fit import load as cached
from distill_data import load as agent,public,configure_engine,ACTIVE
from quantile_fit import HERE,ROOT,LAND,_ds_logprob
from quantile_open import first,signature
from receipt_data import ACT

def main(data):
    configure_engine()
    source=json.loads((data/'receipts-cache/index.json').read_text())
    index=json.loads((data/'quantile-cache/index.json').read_text())
    block=set(index['excluded'])|{r['group'] for r in index['records']}
    roster=defaultdict(list);indexes={};seen=set()
    for r in source['games']:
        if r['layout'] in block:continue
        folder=ROOT/r['source']
        if folder not in indexes:indexes[folder]={Path(m['file']).name:m for m in json.loads((folder/'index.json').read_text()) if m.get('file')}
        row=indexes[folder].get(r['file'])
        if row is None:continue
        for side,k in enumerate(row['order']):
            label=row['strategies'][k]['name']
            if label not in ('CarbonEdge_Artemis','CarbonEdge_Prometheus'):continue
            path=folder/'replays'/r['file']
            if not signature(first(path),side)[1]:continue
            roster[label].append((r,side,path))
    tasks=[]
    for name,pool in roster.items():
        tasks.extend((name,*p) for p in sorted(pool,key=lambda p:hashlib.sha256(('quantile-controls-v1'+p[0]['layout']).encode()).hexdigest())[:10])
    scratch=data/'quantile-stress';scratch.mkdir(exist_ok=True);manifest=[]
    m=agent(ROOT/'experiments/v7/checkpoints/fairlead.py')
    native=m._ds_class_reply;capture={}
    def trace(state,players,op,step,opponent_x,prev):
        capture['bare']=state['opponent'][0]
        return native(state,players,op,step,opponent_x,prev)
    m._ds_class_reply=trace
    for j,(name,row,side,path) in enumerate(tasks):
        dest=scratch/(row['layout']+'.npz')
        manifest.append(dict(name=name,layout=row['layout'],side=side,file=dest.name,replay=row['file'],source=row['source']))
        if dest.exists():continue
        m._ST.clear();tr=m.BeliefTracker(m._Bel,1-side);prior=None;stream=np.random.RandomState(37);arr=defaultdict(list)
        for t,o,after,actions,cfg in public(path,1-side):
            if t>max(ACTIVE):break
            p=o['players'];w=p[side][2];nbase=len(p[side][1]);u=stream.random_sample(nbase+len(w))[nbase:]
            est,known=tr.step(p,t,o['carbon'])
            est,known=m._sense_tree_carbon(tr.bel,o,cfg,prior,est,known)
            if t in ACTIVE:
                board=m._quant(m.encode(p,t,1-side,est,known,prior))
                m._v5_root(o,est,board,prior)
                if w:
                    state=m._ST['v5_history'];pos=[v[0] for v in w.values()]
                    a=state['ds_class'];arr['base'].extend(_ds_logprob(state['opponent'][0][pos]/.4))
                    arr['bare'].extend(capture['bare'][pos]);arr['u'].extend(u[:,None])
                    arr['y'].extend(ACT[actions[side].get(k) or 'NONE'] for k in w)
                    arr['uid'].extend(int(k.split('-')[-1]) for k in w)
                    arr['pos'].extend(pos);arr['t'].extend([t]*len(pos))
                    arr['evidence'].extend([a['evidence']]*len(w));arr['count'].extend([a['count']]*len(w))
                    arr['role'].extend(int(v[2]=='PLANTER') for v in w.values());arr['cargo'].extend(v[1] for v in w.values())
            tr.set_actions(p,actions[1-side]);prior=p;m._ST['prev']=p
        np.savez_compressed(dest,**arr)
        print(j+1,len(tasks),name,flush=True)
    (scratch/'index.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
