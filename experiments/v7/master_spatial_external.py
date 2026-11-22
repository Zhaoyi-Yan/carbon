"""Independent local archive replication on layouts outside both the tuning and 13-map evaluation."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import master_spatial as mx
import master_worker as mw
from master_worker_data import load,ROOT
from master_data import NAMES,SAMPLES
from receipt_data import layout
from receipt_manifest import semantic_layout
from master_spatial_fit import PARENT,HERE
from master_spatial_check import scored,metric

def roster(data):
    studied=json.loads((data/'master-worker/index.json').read_text());used={r['group'] for r in studied}
    ban=set(json.loads((data/'distill-cache/index.json').read_text())['excluded_layouts']);used|=ban
    canonical=set()
    for r in studied:
        with gzip.open(r['file']) as f:canonical.add(semantic_layout(json.load(f)))
    metadata={}
    for folder in (ROOT/'../v5-data',ROOT/'../v6-data',ROOT/'../v6r-data',ROOT/'replays'):
        for ix in folder.glob('**/index.json'):
            try:rows=json.loads(ix.read_text())
            except ValueError:continue
            if not isinstance(rows,list):continue
            for r in rows:
                if not isinstance(r,dict) or 'strategies' not in r:continue
                for k,p in r['strategies'].items():
                    name=p['name'].split('-')[-1]
                    if name in ('Master7sD','Master7sE'):metadata[r['id']]=(r['order'].index(k),name)
    for r in json.loads((ROOT/'analysis/v7/distill-public.json').read_text())['matches']:
        match=r['id']
        source=next((v for v in json.loads((data/'distill-public/index.json').read_text()) if v['match']==match),None)
        if source is None:continue
        teacher=(r['name_b'] if r['name_a']=='yan-QQ-v6' else r['name_a']).split('-')[-1]
        if teacher in ('Master7sD','Master7sE'):metadata[match]=1-source['seat'],teacher
    paths={}
    for root in (ROOT/'../v5-data',ROOT/'../v6-data',ROOT/'../v6r-data',ROOT/'replays',data/'distill-public'):
        for p in root.glob('**/*.json.gz'):
            mid=p.name[:32]
            if mid in metadata and p.name[32:34]=='-1':paths.setdefault(mid,p)
    excludeids={r['match'] for r in studied}|{'03100a95e8647c51a4d642fe08d856dd','4063a34afedb6d4767a9118e1dee9e5d'}
    cap={'Master7sE':8,'Master7sD':6};selected=[]
    for mid in sorted(set(paths)-excludeids,key=lambda s:hashlib.sha256(('private-confirm-spatial-'+s).encode()).digest()):
        side,name=metadata[mid]
        if sum(r['name']==name for r in selected)>=cap[name]:continue
        p=paths[mid]
        with gzip.open(p) as f:g=json.load(f)
        group=layout(g);sig=semantic_layout(g)
        if group in used or sig in canonical:continue
        used.add(group);canonical.add(sig);selected.append(dict(mid=mid,path=str(p),name=name,s=side,layout=group,canonical=sig))
    return selected

def main(data,weights):
    selected=roster(data);selected.sort(key=lambda r:r['name'],reverse=True)
    code=mx.build(PARENT.read_text(),weights);candidate=data/'master-spatial-worker-library.py';candidate.write_text(code)
    m=load(candidate);old=dict(np.load(HERE/'master-worker-model.npz'));sums=defaultdict(list);events=[];tim=[]
    for r in selected:
        s=r['s'];sid=NAMES.index(r['name']);te=r['name'].replace('Master7s','')
        with gzip.open(r['path']) as f:g=json.load(f)
        hist=[];m._ST.clear();tr=m.BeliefTracker(m._Bel,s);per=defaultdict(list)
        for t,f in enumerate(g['steps'][:-1]):
            obs={**f[0]['observation'],**f[s]['observation']};obs.pop('full_carbon',None);obs['player']=s
            pl=obs['players'];prev=hist[0] if hist else None
            est,known=tr.step(pl,t,obs['carbon']);est,known=m._sense_tree_carbon(tr.bel,obs,g['configuration'],prev,est,known)
            cmd=g['steps'][t+1][s]['action'] or {}
            if t in SAMPLES:
                x=m._quant(m.encode(pl,t,s,est,known,prev));root=m._v5_root(obs,est,x,prev)
                uu,role,h,l,k,scene=mw._mw_inputs(obs,x,prev,m._DIRECTIONAL[0])
                if uu:
                    loc=[pl[s][2][u][0] for u in uu];frozen=root[0][loc]
                    ref=mw._mw_logits(h,l,k,role,frozen,np.tile(np.eye(3)[sid],(len(uu),1)),old)
                    t0=time.perf_counter();new,_=m._mx_heads(obs,x,hist,te);tim.append(1000*(time.perf_counter()-t0))
                    labels=np.array([m.WIDX.get(cmd.get(u),0) for u in uu])
                    for name,mask in [('all',role>=0),('heavy_home',(role==0)&(l[:,1]*170>=65)&(l[:,2]*7<=4.01)),
                                      ('light_enemy_base',(role==0)&(l[:,1]*170<=15)&(l[:,3]*7<=4.01))]:
                        for model,z in [('directional_known',ref),('spatial4',new[loc]),('frozen_root',frozen)]:
                            vals=scored(z[mask],labels[mask])
                            if vals:
                                per[name,model].extend(vals);sums[r['name'],name,model].extend(vals)
            tr.set_actions(pl,cmd);m._ST['prev']=pl;hist=[pl]+hist[:2]
        metrics={}
        for (sc,k),v in per.items():metrics.setdefault(sc,{})[k]=metric(v)
        events.append(dict(**r,metrics=metrics));print(r['name'],r['mid'],flush=True)
    out={}
    for (co,sc,k),v in sums.items():out.setdefault(co,{}).setdefault(sc,{})[k]=metric(v)
    result=dict(sources=events,pooled=out,worker_only=True,parent_sha=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
           selected_before_external='Plain three-step auxiliary residual was frozen based on the 42/13 reproduction; the ten-step goal alternative was declined. No external labels used in either fit or model selection.',
           exclusion='All 55 original layouts excluded, including symmetric layouts and exchanged bases; both screenshot matches and qualification layout fingerprints excluded.',
           cpu_inference_ms_p50_p95=np.quantile(tim,[.5,.95]).round(3).tolist())
    (ROOT/'analysis/v7/master-spatial-external.json').write_text(json.dumps(result,indent=2)+'\n');print(out)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('model',type=Path);a=p.parse_args();main(a.data,a.model)
