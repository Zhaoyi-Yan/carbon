"""Test simulated visibility against both real trackers and actual next worker moves on separate games."""
import argparse
import json
from pathlib import Path
import numpy as np
from branch_replay import ROOT,OUT,HERE
from benchmark import configure_engine,limit_math_threads,load_strategy
from starvation_replay import read_replay
from rollout_fog import _v7_next_known


def assess(m,game,root_side):
    f=game['steps'];track=[m.BeliefTracker(m._Bel,i) for i in (0,1)];prev=None;masks={};raw=[];rows=[]
    def predict():
        if not raw:return
        x=np.array([u[1] for u in raw]);ids=np.asarray([22 if u[5]==root_side else 20 for u in raw]);w,_,_=m._V6R_TINY(x,ids)
        for idx,(kind,_,pos,labels,err,s) in enumerate(raw):
            z=w[idx,pos];prob=z-np.logaddexp.reduce(z,axis=1)[:,None]
            rows.append((kind,float(-prob[np.arange(len(pos)),labels].sum()),int((z.argmax(1)==labels).sum()),len(pos),err))
        raw.clear()
    for t in range(min(181,len(f)-1)):
        p=f[t][0]['observation']['players'];action=[z['action'] or {} for z in f[t+1]]
        estimate=[];gold=[]
        for s in (0,1):
            stock,known=track[s].step(p,t,f[t][s]['observation']['carbon'])
            estimate.append(stock);gold.append(known)
        gold=np.asarray(gold,dtype=bool)
        if t%15==0:masks={k:gold.copy() for k in ('root','increment','reflect')}
        else:
            for k in ('increment','reflect'):masks[k]=_v7_next_known(masks[k],p,k=='reflect')
        if t%15 in (4,9,14):
            mask={'all':np.ones_like(gold),'actual':gold,**masks}
            obs=m._v5_pair(p,t,estimate[root_side],prev)
            for s in (0,1):
                units=p[s][2]
                if not units:continue
                pos=[u[0] for u in units.values()];a=np.array([m.WIDX.get(action[s].get(k),0) for k in units])
                for k,v in mask.items():
                    x=obs[s].copy();x[1]=v[s].reshape(15,15);raw.append((k,x,pos,a,float(np.mean(v[s]!=gold[s])),s))
            if len(raw)>=25:predict()
        for s in (0,1):track[s].set_actions(p,action[s])
        prev=p
    predict();out={}
    for k in ('all','actual','root','increment','reflect'):
        r=[v for v in rows if v[0]==k];out[k]=[sum(v[i] for v in r) for i in range(1,5)]+[len(r)]
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args()
    limit_math_threads();configure_engine();m=load_strategy('fog_data',HERE/'checkpoints/pilotage.py');m._torch.set_num_threads(1)
    _v7_next_known.__globals__.update({n:getattr(m,n) for n in ('NB','CLS','CLS_MEMBERS')})
    meta=json.loads((a.folder/'index.json').read_text());rows=[]
    for r in meta[7:17]:
        rows.append(dict(file=r['file'],scores=assess(m,read_replay(a.folder/r['file']),r['seat'])))
    summary={}
    for k in rows[0]['scores']:
        loss,ok,n,err,count=np.sum([r['scores'][k] for r in rows],axis=0)
        summary[k]=dict(nll=loss/n,accuracy=ok/n,mask_error=err/count,actions=int(n))
    print(summary);(OUT/'rollout-fog-validation.json').write_text(json.dumps(dict(summary=summary,games=rows),indent=2)+'\n')
if __name__=='__main__':main()
