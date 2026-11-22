"""Check true compiled root integration on one held-out public match, including a live search call."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from distill_data import load,ROOT
from benchmark import validate_actions
import distill
import distill_fast
import distill_id

def main(a):
    manifest=json.loads((a.cache/'index.json').read_text())['records']
    r=next(v for v in manifest if v['source']=='public' and v['fold']=='test' and not v['negative'])
    src=next(x for x in json.loads((a.public/'index.json').read_text()) if x['match']==r['id'])
    with gzip.open(a.public/src['file']) as f:g=json.load(f)
    observer=1-r['seat'];m=load(a.candidate);cfg=g['configuration']
    builder={'initial':distill,'class':distill_id,'fast':distill_fast}[a.kind]
    assert a.candidate.read_text()==builder.build(a.parent.read_text())
    times=[];blends=[];magnitude=[];rootsearch=m._search
    hook='_ds_update' if a.kind=='initial' else '_ds_class_reply'
    updater=getattr(m,hook)
    def audit(state,team,side,t,x,prev):
        start=time.perf_counter();prior=state['opponent'][0]
        updater(state,team,side,t,x,prev)
        times.append((time.perf_counter()-start)*1000)
        if a.kind=='initial':
            ms=state['distilled_reply'];count=ms['count'];gain=ms['evidence']/max(1.,count)
            alpha=float(np.clip((gain-.008)*4.,0,.78)) if count>=12 else 0.
        else:alpha=state['ds_class']['weight']
        blends.append(alpha)
        change=np.abs(state['opponent'][0]-prior).max()
        if alpha==0:assert change==0.
        magnitude.append(float(change))
    setattr(m,hook,audit)
    for t,frame in enumerate(g['steps'][:40]):
        obs={k:v for k,v in frame[0]['observation'].items() if k!='full_carbon'}
        obs.update(frame[observer]['observation']);obs.pop('full_carbon',None);obs['player']=observer
        m._search=lambda obs,cfg,stock,act,*args:act
        if t==39:
            m._search=rootsearch;m._S_K=4;m._S_D2=2;m._S_COST[:]=[.0004,.0003,.0004,.0003]
            act=m.agent(obs,cfg)
        else:act=m._act(obs,cfg,None)
        validate_actions(act,obs['players'][observer])
        m._ST['tr'].set_actions(obs['players'],g['steps'][t+1][observer]['action'] or {})
    assert len(times)==40 and m._S_STATS['searched']>0
    out={'frozen_parent_sha':hashlib.sha256(a.parent.read_bytes()).hexdigest(),'candidate_sha':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
         'match':r['id'],'active_root_frames':int(np.sum(np.array(blends)>0)),'max_change':round(max(magnitude),4),
         'timing_ms_p50_p95':np.percentile(times,[50,95]).round(3).tolist(),'total_root_frames':len(times),
         'true_searches':int(m._S_STATS['searched']),'seat_observing':observer,'scope':'Cold initial forty public observations. Search invoked once; other decisions only checked root prediction.'}
    a.report.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('public',type=Path);p.add_argument('parent',type=Path);p.add_argument('candidate',type=Path);p.add_argument('report',type=Path)
    p.add_argument('--kind',choices=('initial','class','fast'),default='initial')
    main(p.parse_args())
