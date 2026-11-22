"""Time frozen own, exact dual, student dual, and one-reference value inference on real leaf tensors."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from antisym_data import PARENT,ROOT,configure_engine,load,public,stockroot

def main(data):
    configure_engine();m=load(PARENT);row=json.loads((data/'distill-public/index.json').read_text())[0]
    side=row['seat'];me=1-side;tr=m.BeliefTracker(m._Bel,me);prev=None;views=[];sc=[];ids=[];m._ST.clear()
    for t,o,after,acts,cfg in public(data/'distill-public'/row['file'],me):
        pos=stockroot(m,o,cfg,prev,tr,t>=28);pl=o['players']
        if pos:
            root,w,b=pos;future=m._turn(root,cfg,acts)
            views.append(m._v7_visible_views(future,pl)[[me,side]])
            sc.append([m._scalars(future['players'],t+1,s) for s in (me,side)])
            ids.append(m._v5_ids([me,side]))
        tr.set_actions(pl,acts[me]);prev=pl
        if len(views)==8:break
    X=np.asarray(views);S=np.asarray(sc);I=np.asarray(ids);net=m._net()
    result={}
    for n in (2,4,8):
        x=np.ascontiguousarray(X[:n,0]);s=np.ascontiguousarray(S[:n,0]);i=I[:n,0]
        ex=np.ascontiguousarray(X[:n,1]);es=np.ascontiguousarray(S[:n,1]);ei=I[:n,1]
        xx=np.ascontiguousarray(X[:n].reshape(2*n,m.NPL,15,15));ss=S[:n].reshape(2*n,5);ii=I[:n].ravel()
        ax=np.concatenate((x,ex[:1]));ass=np.concatenate((s,es[:1]));ai=np.r_[i,ei[0]]
        fs=dict(single=lambda:net(x,i,s),full=lambda:net(xx,ii,ss),
                student=lambda:(net(x,i,s),m._V6R_TINY(ex,ei,es)),
                reference=lambda:(net(ax,ai,ass),m._V6R_TINY(ex,ei,es)))
        ts={k:[] for k in fs};rng=np.random.RandomState(5303)
        for rep in range(12):
            for k in rng.permutation(list(fs)):
                begin=time.perf_counter_ns();fs[k]();elapsed=(time.perf_counter_ns()-begin)/1e6
                if rep>=2:ts[k].append(elapsed)
        result[n]={k:np.quantile(v,[.5,.9]).round(4).tolist() for k,v in ts.items()}
    report=dict(ms_p50_p90=result,cpu_threads=1,source=row['match'],note='Neural passes only, on real leaf tensors at eight consecutive turns, 10 measured shuffled repetitions. Small-network adaptation and archive simulation are excluded from timings.')
    (ROOT/'analysis/v7/antisym-timing.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
