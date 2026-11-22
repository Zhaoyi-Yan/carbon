"""Test a prespecified ambiguity fallback on all 13 held-out maps, with native and old-head controls."""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from scipy.special import softmax
from master_spatial_check import load_net,HERE,load,PARENT,contents,ROOT,NAMES,mw

def main(data):
    torch.set_num_threads(1);model=load_net(HERE/'master-spatial.npz');m=load(PARENT)
    oldweights=dict(np.load(HERE/'master-worker-model.npz'));mw._V5_ROTATIONS=m._V5_ROTATIONS
    records=[];rows=json.loads((data/'master-worker/index.json').read_text())
    for row in (q for q in rows if q['fold']=='test'):
        x,y,_,mask,ts,rots=contents(data/'master-spatial'/row['cache']);sid=NAMES.index(row['teacher']);out=[]
        with torch.inference_mode():
            for v in np.array_split(x,max(1,len(x)//32)):
                out.extend(model(torch.from_numpy(v.astype('f')),torch.full((len(v),),sid,dtype=torch.long)).numpy())
        out=np.stack(out).transpose(0,2,1)
        with np.load(data/'master-worker'/row['cache']) as f:
            base=f['base'].astype('f');targets=f['y'];n=len(targets);w=f['p'] if 'p' in f else None
            st=np.eye(3,dtype='f')[np.full(n,sid)]
            old=mw._mw_logits(*[f[k].astype('f') for k in ('h','l','k')],f['r'],base,st,oldweights)
            wt=f['t'];relevant=f['heavy']|f['pressure']
        from starvation_replay import read_replay
        env=read_replay(Path(row['file']));stats=defaultdict(int)
        for t in np.unique(ts):
            ids=np.flatnonzero(ts==t);order=list(env['steps'][int(t)][0]['observation']['players'][row['side']][2].values())
            if not order:continue
            pos=[u[0] for u in order];idx=np.flatnonzero(wt==t);assert len(idx)==len(pos)
            z=[]
            for j,i in enumerate(ids):
                c,_,a,_=m._V5_ROTATIONS[j];z.append(out[i,c[pos],:5][:,a])
            probs=softmax(np.mean(z,axis=0),axis=1)
            original=base[idx].argmax(1);rank=np.argsort(-probs,axis=1,kind='stable');r0,r1=rank[:,0],rank[:,1]
            prev=probs[np.arange(len(pos)),r0];alt=probs[np.arange(len(pos)),r1]
            allowed=relevant[idx]&(r0==original)&(alt>=.18)&(alt>=.45*prev)
            if not allowed.any():continue
            k=np.argmax(np.where(allowed,alt/prev,-1));target=targets[idx[k]]
            top=int(r0[k]);second=int(r1[k])
            control=[]
            for alternative in (base[idx[k]],old[idx[k]]):
                valid=alternative.copy();valid[top]=-np.inf;control.append(valid.argmax())
            for name,yes in dict(eligible=True,original=target==top,expert_second=target==second,native_second=target==control[0],older_second=target==control[1],diff_native=second!=control[0]).items():
                stats[name]+=int(yes)
        records.append(dict(match=row['match'],teacher=row['teacher'],stats=dict(stats)))
    groups={}
    for label in tuple(NAMES)+('all',):
        taken=[r for r in records if label=='all' or r['teacher']==label]
        groups[label]={k:sum(r['stats'].get(k,0) for r in taken) for k in ('eligible','original','expert_second','native_second','older_second','diff_native')}
    result=dict(groups=groups,maps=records,method='One optional distinct worker action per snapshot, conditional on expert/native top agreement, P(second)>=.18 and ratio>=.45, highest ratio over eligible workers; no collision reservation in the label-based approximation. Scores count matching the demonstrated action on identical states. This diagnoses extra action coverage and cannot establish a win advantage.')
    (ROOT/'analysis/v7/master-spatial-runnerup.json').write_text(json.dumps(result,indent=2)+'\n');print(groups,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);main(p.parse_args().data)
