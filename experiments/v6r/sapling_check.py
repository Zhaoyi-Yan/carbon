"""Check the generated leaf scorer against its fitted formula on one-step simulated boards."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter
import numpy as np


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('data','candidate','base','model','output'):p.add_argument('--'+arg,required=True,type=Path)
    args=p.parse_args();m=load(args.candidate,'sapling_agent');b=load(args.base,'grove_agent')
    report=json.loads(args.model.read_text())
    trees=np.asarray(report['coefficients'])
    subset=report['maps']['validation'][:6]
    timing=[];oldtime=[];changed=0;errors=[]
    for row in subset:
        g=json.load(gzip.open(args.data/'replays'/(row['match']+'-1.json.gz')))
        f=g['steps'];seat=row['observer'];track=m.BeliefTracker(m._Bel,seat)
        for t,fr in enumerate(f[:-1]):
            o=fr[0]['observation'];pl=o['players'];est,_=track.step(pl,t,fr[seat]['observation']['carbon'])
            actions=[s['action'] or {} for s in f[t+1]];track.set_actions(pl,actions[seat])
            if t%15:continue
            root=dict(step=t,player=seat,players=pl,trees=o['trees'],carbon=est)
            board=m._turn(root,g['configuration'],actions)
            assert board['_sap_previous'] is pl
            s=perf_counter();base=b._r6_asset_score(board,seat);oldtime.append((perf_counter()-s)*1000)
            s=perf_counter();actual=m._r6_asset_score(board,seat);timing.append((perf_counter()-s)*1000)
            meta=board['_st'].trees_meta
            now=board['players'];ctx=m._sapling_context(now,pl);correction=0.
            for side in (0,1):
                _,orig=b._r6_assets(now,meta,side,t+1,298)
                for uid,(_,age) in now[side][3].items():
                    if age>6 or t+1>=299:continue
                    old=float(orig[uid]@b._R6_TREE)
                    x=m._sapling_features(board,meta,side,uid,old,ctx,298)
                    ix=0 if age==1 else 1 if age<=3 else 2
                    gain=float(np.clip(old+np.dot(x,trees[ix]),-90,300)-np.clip(old,-90,300))
                    correction+=(1 if side==seat else -1)*gain/3000.
            errors.append(abs(base+correction-actual));changed+=abs(actual-base)>1e-6
    assert max(errors)<1e-10
    out=dict(base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
             simulated_leaves=len(errors),changed=int(changed),max_formula_error=float(max(errors)),
             per_leaf_ms_median_base=float(np.median(oldtime)),per_leaf_ms_median_candidate=float(np.median(timing)),
             per_leaf_delta_ms_percentiles=np.percentile(np.array(timing)-np.array(oldtime),[50,95,99]).tolist())
    args.output.write_text(json.dumps(out,indent=1)+'\n')
    print(out)
if __name__=='__main__':main()
