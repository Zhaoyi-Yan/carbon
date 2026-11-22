"""Trace live style choices on held-out recorded games using decisions available at each turn."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import numpy as np
from backbone import configure_engine, limit_math_threads, load_strategy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('candidate','split','output'):p.add_argument(k,type=Path)
    p.add_argument('snapshots',nargs='+',type=Path)
    args=p.parse_args()
    configure_engine();limit_math_threads();m=load_strategy('candidate_for_trace',args.candidate.resolve())
    names=sorted(json.loads(args.split.read_text())['independent_maps'])[:4]
    traces=[]
    for name in names:
        paths=[d/'replays'/(name+'.json.gz') for d in args.snapshots]
        path=next(f for f in paths if f.exists())
        frames=json.loads(gzip.decompress(path.read_bytes()))['steps']
        for side in (0,1):
            tracker=m.BeliefTracker(m._Bel,side);prev=None;selections=[]
            for t,(before,after) in enumerate(zip(frames,frames[1:])):
                shared=before[0]['observation'];pl=shared['players']
                obs={k:shared[k] for k in ('players','step','trees')}
                obs['player']=side
                est,known=tracker.step(pl,t,before[side]['observation']['carbon'])
                x=m._quant(m.encode(pl,t,side,est,known,prev))
                m._v5_root(obs,est,x,prev)
                selections.append(1+m._ST['v5_history']['style'])
                tracker.set_actions(pl,after[side]['action'] or {});prev=pl
            traces.append(dict(match=name,seat=side,teacher_rank_counts=dict(collections.Counter(selections)),
                               switches=sum(a!=b for a,b in zip(selections,selections[1:]))))
    args.output.write_text(json.dumps(traces,indent=2)+'\n')
    print([(r['seat'],r['switches'],len(r['teacher_rank_counts'])) for r in traces],flush=True)


if __name__=='__main__':
    main()
