"""Evaluate both training domains separately and replay the real joint-policy deployment gate."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import types
import numpy as np
import torch
from master_full import ROOT,PARENT,original,take_data,evaluate_holds
from master_full_verify import old_elite,routed
from shipping_styles import _v7_shipping_gap

class Mapping:
    def __init__(self,model,high):
        self.model=model;self.high=high
    def __call__(self,x,ids,s):return self.model(x,ids//32 if self.high else ids%32,s)

def main(data,source,film=False):
    torch.set_num_threads(1);device='mps' if torch.backends.mps.is_available() else 'cpu'
    models=[original(PARENT).to(device).eval()]
    if film:
        from master_multi_film import load_film,HERE
        models.append(load_film(HERE/'master_multi_film_weights.npz',device))
    else:models.append(original(source).to(device).eval())
    rows=json.loads((data/'master-worker/index.json').read_text())
    hold=[r for r in rows if r['fold']=='test'];d=take_data(data,hold,False)
    spec=types.ModuleType('deployment');code=source.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)')
    exec(compile(code,str(source),'exec'),spec.__dict__)
    original_ids=np.zeros(len(d['x']),dtype=np.int64);selected=original_ids.copy();forced=original_ids.copy()
    gate=[];active_states=0
    for nr,row in enumerate(hold):
        with gzip.open(row['file']) as f:g=json.load(f)
        s=row['side'];base=routed(spec,g,s);ix=np.flatnonzero(d['map']==nr);t=d['t'][ix]
        gap=_v7_shipping_gap(d['x'][ix].astype('f')*.01);original_ids[ix]=np.where(np.abs(gap)<95,base[t],23+2*(base[t]-21)+(gap>0))
        forced[ix]=np.where(np.abs(gap)<95,27,28+(gap>0))
        st={};flags=[]
        for step,f in enumerate(g['steps']):
            ob=f[0]['observation'];c={'step':step,'player':s,'players':ob['players']}
            flags.append(spec._mm_switch(c,22,0.,st)>=27)
        flags=np.asarray(flags);selected[ix]=np.where(flags[t],forced[ix],original_ids[ix])
        gate.append(dict(match=row['match'],layout=row['group'],teacher=row['teacher'],seen=len(flags),active=int(flags.sum()),
                          samples=len(ix)//4,active_samples=int(flags[t[::4]].sum())));active_states+=gate[-1]['active_samples']
    wrappers=[Mapping(models[i],i==1) for i in (0,1)];results={}
    for key,ids in [('own_master',forced),('gated',selected),('original_styles',original_ids)]:
        d['sid']=original_ids+32*ids
        results[key]=evaluate_holds(*wrappers,d,hold,device)
    results['older']=old_elite(data,models,device,{r['group'] for r in rows})
    results['master_gate']=dict(replay_frames=gate,active_states=active_states,states=len(d['x'])//4)
    # Explicit per-group preservation as well as finding whether the frozen activation is too sparse.
    out=ROOT/('analysis/v7/master-multi-film-validation.json' if film else 'analysis/v7/master-multi-validation.json');out.write_text(json.dumps(results,indent=2)+'\n')
    for k in ('own_master','gated','original_styles'):
        q=results[k]['cohorts']['all'];print(k,{s:[q['lifeline'][s]['hit'],q['full'][s]['hit'],q['lifeline'][s]['nll'],q['full'][s]['nll']] for s in ('worker','loaded_home','light_enemy_depot','affordable_base')},flush=True)
    print('original elite',results['older']['combined'],'gate',active_states,len(d['x'])//4,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--film',action='store_true');a=p.parse_args();main(a.data,a.candidate,a.film)
