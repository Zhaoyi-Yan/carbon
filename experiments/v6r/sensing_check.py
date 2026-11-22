"""Score two sequential legal-view belief trackers. Full carbon is used only by the scorer."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter
import numpy as np


def score_one(model, row, directory, limit):
    with gzip.open(directory/row['file']) as f:
        g=json.load(f)
    title=row['id']
    seat=hashlib.sha256(title.encode()).digest()[3] % 2
    control=model.BeliefTracker(model._Bel,seat)
    candidate=model.BeliefTracker(model._Bel,seat)
    last=None
    metrics=defaultdict(lambda:np.zeros(6))
    cfg={k:g['configuration'][k] for k in ('cellAbsorptionRate','collectorAbsorptionRate','maxCellCarbon')}
    elapsed=[]
    for step,fr in enumerate(g['steps'][:limit+1]):
        now=fr[0]['observation']['players']
        visible_carbon=fr[seat]['observation']['carbon']
        obs=dict(players=now,trees=fr[0]['observation']['trees'],carbon=visible_carbon,player=seat,step=step)
        base,base_known=control.step(now,step,visible_carbon)
        est,known=candidate.step(now,step,visible_carbon)
        start=perf_counter()
        est,known=model._sense_tree_carbon(candidate.bel,obs,cfg,last,est,known)
        elapsed.append((perf_counter()-start)*1000)
        if step:
            visible=np.zeros(len(est),dtype=bool)
            for pos,_,role in now[seat][2].values():
                visible[[pos] if role=='COLLECTOR' else model.NB[pos]]=True
            truth=np.array(fr[0]['observation']['full_carbon'])
            a,b=np.abs(np.array(base)-truth),np.abs(np.array(est)-truth)
            fog=~visible
            unknown=np.logical_not(base_known)
            groups={'fog':fog,'base_unknown_fog':fog & unknown,
                    'newly_resolved':fog & unknown & np.array(known)}
            phase='1-75' if step<=75 else '76-150'
            for name, mask in groups.items():
                av,bv=a[mask],b[mask]
                metrics[phase,name]+= [mask.sum(),av.sum(),bv.sum(),sum(bv>.1),sum(bv>1),sum(bv>5)]
        if step+1 < len(g['steps']):
            action=g['steps'][step+1][seat]['action'] or {}
            control.set_actions(now,action)
            candidate.set_actions(now,action)
        last=now
    return metrics,elapsed,seat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--maps',default=36,type=int)
    p.add_argument('--offset',default=0,type=int)
    p.add_argument('--limit',default=150,type=int)
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('sensed_candidate',args.candidate)
    model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
    rows=json.loads((args.data/'index.json').read_text())
    assert len({r['id'] for r in rows})==len(rows)
    # Hash order is fixed before evaluating any fog errors or game outcomes.
    sample=sorted(rows,key=lambda r:hashlib.sha256(('tree-sensing-v1:'+r['id']).encode()).digest())[args.offset:args.offset+args.maps]
    agg=defaultdict(lambda:np.zeros(6));clock=[];games=[]
    for row in sample:
        stats,elapsed,seat=score_one(model,row,args.data,args.limit)
        maps={}
        for key,value in stats.items():
            agg[key]+=value
            if key[0]=='1-75' and key[1]=='fog':
                maps['early_fog_mae_gain']=round((value[1]-value[2])/value[0],5)
        clock.extend(elapsed)
        games.append(dict(match=row['id'],seat=seat,**maps))
    summary={}
    for (phase,key),(n,sa,sb,bad01,bad1,bad5) in agg.items():
        summary.setdefault(phase,{})[key]=dict(cells=int(n),baseline_mae=round(sa/n,6),
                                sensing_mae=round(sb/n,6),relative_reduction_pct=round((sa-sb)/sa*100,3),
                                sensing_errors_over_pt1=int(bad01),sensing_errors_over_1=int(bad1),
                                sensing_errors_over_5=int(bad5))
    report=dict(available_maps=len(rows),sampled_maps=len(sample),sample_offset=args.offset,max_step=args.limit,
                candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),sample=games,
                per_step_overhead_ms_p50_p95_p99=[round(x,3) for x in np.percentile(clock,[50,95,99])],
                summary=summary)
    args.output.write_text(json.dumps(report,indent=1)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sample','candidate_sha256')},indent=1))


if __name__=='__main__':
    main()
