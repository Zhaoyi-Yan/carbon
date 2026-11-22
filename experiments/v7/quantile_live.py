"""Check the assembled module, live sequential recognition and additive CPU cost on public observations."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import quantile
from quantile_fit import HERE,ROOT,_ds_logprob,summary
from distill_data import configure_engine,load,load_manifest,local,public
from receipt_data import ACT
from benchmark import validate_actions

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main(data):
    configure_engine()
    src=ROOT/'experiments/v7/checkpoints/fairlead.py';file=data/'quantile-fairlead.py'
    built=quantile.build(src.read_text());compile(built,str(file),'exec')
    if file.exists():assert file.read_text()==built
    else:file.write_text(built)
    m=load(file);rows,fold,ban=load_manifest(data/'distill-public');maps=set();jobs=[]
    for r in sorted(rows,key=lambda r:(r['source'],r['group'],r['id'])):
        srcname=r['source'];g=r['group']
        if fold[g]!='test' or g in maps or sum(c[0]==srcname+'-v6' for c in jobs)>=2:continue
        maps.add(g);jobs.append((srcname+'-v6',g,r['seat'],srcname,r['payload'],r['id']))
        if srcname=='local' and not any(j[0]=='v5' for j in jobs):jobs.append(('v5',g,1-r['seat'],srcname,r['payload'],r['id']))
    hard=json.loads((data/'quantile-stress/index.json').read_text())
    for c,n in [('CarbonEdge_Artemis',4),('CarbonEdge_Prometheus',1)]:
        for r in [v for v in hard if v['name']==c][:n]:
            jobs.append((c,r['layout'],r['side'],'public',ROOT/r['source']/'replays'/r['replay'],r['replay']))
    stats=defaultdict(list);tim=[];reports=[];updates=0
    upd=m._qt_update;search=m._search
    def trace(state,players,side,step,prior):
        nonlocal updates
        workers=players[side][2];sites=[w[0] for w in workers.values()];old,bases=state['opponent']
        tick=time.perf_counter_ns();upd(state,players,side,step,prior);tim.append((time.perf_counter_ns()-tick)/1e6)
        check=state['quantile_reply'];stream.random_sample(len(players[side][1])+len(workers))
        a=stream.get_state();b=check['rng'].get_state()
        assert np.array_equal(a[1],b[1]) and a[2:]==b[2:]
        assert state['opponent'][1] is bases
        if step>=2:assert check.get('matched',False)==(name!='v5')
        updates+=1
        if workers:
            start=_ds_logprob(old[sites]/.4);end=_ds_logprob(state['opponent'][0][sites]/.4)
            idx=np.arange(len(sites));y=np.array([ACT[commands[side].get(k) or 'NONE'] for k in workers])
            nll=-end[idx,y];mask=np.array([w[1]>=65 and w[2]=='COLLECTOR' for w in workers.values()])
            alt=end.copy();alt[idx,start.argmax(1)]=-np.inf
            changed=np.any(abs(end-start)>1e-6,axis=1)
            scores=np.column_stack((nll+start[idx,y],nll,end.argmax(1)==y,np.exp(-nll),alt.argmax(1)==y,changed))
            for typ,use in [('all',idx>=0),('loaded',mask)]:
                if use.any():
                    stats[name,typ].append((g,scores[use]))
                    if name.endswith('-v6'):stats['combined-v6',typ].append((g,scores[use]))
    m._qt_update=trace
    for name,g,side,source,payload,mid in jobs:
        observer=1-side;m._ST.clear();stream=np.random.RandomState(37)
        it=local(payload,observer) if source=='local' else public(payload,observer)
        for t,o,_,commands,cfg in it:
            if t>=128:break
            if t==34 and name=='local-v6' and not reports:
                m._search=search;m._S_K=4;m._S_D2=2;m._S_COST[:]=[.0004,.0003,.0004,.0003]
                action=m.agent(o,cfg);assert m._S_STATS['searched']>0
            else:
                m._search=lambda o,c,e,a,*args:a
                action=m._act(o,cfg,None)
            validate_actions(action,o['players'][observer])
            m._ST['tr'].set_actions(o['players'],commands[observer])
        reports.append(dict(cohort=name,map=g,replay=mid,observing=observer))
        print(len(reports),len(jobs),name,flush=True)
    result={}
    for (co,typ),vals in stats.items():result.setdefault(co,{})[typ]=summary(vals)
    report=dict(parent_sha=sha(src),candidate_sha=sha(file),matched_sequential_rng_states=updates,total_searched=m._S_STATS['searched'],
      module_ms_p50_p95_p99=np.percentile(tim,[50,95,99]).round(4).tolist(),initial_window=[0,128],decisions=reports,metrics=result,
      caveat='Entire previous history of both behavior evidence and quantile evidence carried forward. One actual search; other observations teacher-forced and only legal public root was evaluated. gate field here measures fraction of rows actually altered.')
    (ROOT/'analysis/v7/quantile-live.json').write_text(json.dumps(report,indent=2)+'\n')
    print(updates,report['module_ms_p50_p95_p99'],{k:v['all']['delta_nll'] for k,v in result.items()})
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
