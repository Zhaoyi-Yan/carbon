"""Extract a full team's demonstrated actions using v7's deployed observation encoder."""
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/v7')]
from benchmark import configure_engine,limit_math_threads,load_strategy
from starvation_replay import read_replay,distance,LAND
from collect import write

V7='4894fe44c601b7360d37de2eed928802c73884f0929a5897604d174e333e02e6'


def init():
    global policy
    configure_engine()
    limit_math_threads()
    path=ROOT/'yan-v7.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==V7
    policy=load_strategy('v7_replay_encoder',path,path.read_text().rsplit('\n_net()',1)[0].encode())


def extract(task):
    key,entry,ids,dest=task
    r=entry['row']
    paths=Path(entry['path'])
    cache=dest/(key+'.npz')
    if not cache.exists():
        assert hashlib.sha256(paths.read_bytes()).hexdigest()==entry['sha256']
        env=read_replay(paths)
        cfg,frames=env['configuration'],env['steps']
        end=len(frames)-1
        assert end==r['steps'] and all(f['status']=='DONE' for f in frames[-1])
        first=frames[0][0]['observation']
        layout=hashlib.sha256(json.dumps([first['full_carbon'],[sorted(p[1].values()) for p in first['players']]],separators=(',',':')).encode()).hexdigest()
        cash=[f[0]['observation']['players'][0][0]-f[0]['observation']['players'][1][0] for f in frames]
        rows={k:[] for k in ('X','A','S','Y','T','M')}
        for side,slot in enumerate(r['order']):
            teacher=r['strategies'][slot]['id']
            if teacher not in ids:continue
            assert abs(frames[-1][side]['reward']-r['scores'][slot])<.015
            tracker=policy.BeliefTracker(policy._Bel,side)
            prev=None
            for t in range(end):
                board=frames[t][0]['observation']
                ps=board['players']
                obs=dict(players=ps,trees=board.get('trees',{}),step=t,player=side,carbon=frames[t][side]['observation']['carbon'])
                act=frames[t+1][side]['action'] or {}
                est,known=tracker.step(ps,t,obs['carbon'])
                est,known=policy._sense_tree_carbon(tracker.bel,obs,cfg,prev,est,known)
                x=np.rint(policy.encode(ps,t,side,est,known,prev)*100).clip(0,255).astype('u1')
                a=np.stack(policy.labels(ps,side,act))
                own=ps[side]
                if own[0]<cfg['recCollectorCost'] or len(own[2])>=cfg['workerLimit']:a[1].fill(-1)
                # Audit-only flags; never pass them to the network.
                mask=np.zeros(225,np.uint8)
                for uid,(pos,c,k) in own[2].items():
                    q=LAND[pos][act.get(uid,'STAY')]
                    if k=='COLLECTOR' and c>=75 and min(distance(q,h) for h in own[1].values())<min(distance(pos,h) for h in own[1].values()):
                        mask[pos]|=1
                newtrees=frames[t+1][0]['observation']['players'][side][3]
                for uid,(pos,_) in newtrees.items():
                    if uid not in own[3]:mask[pos]|=2
                if own[0]<cfg['recCollectorCost']:mask|=4
                for name,value in dict(X=x,A=a,S=policy._scalars(ps,t,side),T=ids[teacher],M=mask,
                    Y=(1-2*side)*sum(w*cash[min(t+k,end)] for w,k in ((.28,20),(.52,52),(.20,100)))/3000).items():
                    rows[name].append(value)
                tracker.set_actions(ps,act)
                prev=ps
        tmp=cache.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**{k:np.asarray(v,dtype={'T':'u1','S':'f','Y':'f'}.get(k)) for k,v in rows.items()},layout=layout)
        tmp.replace(cache)
    with np.load(cache,allow_pickle=False) as z:
        teachers=sorted(map(int,np.unique(z['T'])))
        return dict(game=key,layout=str(z['layout']),teachers=teachers,finished=r['finished'],rows=len(z['T']),
                    order=r['order'],seats=[r['strategies'][s]['id'] for s in r['order']],scores=r['scores'],
                    replay_sha256=entry['sha256'],source=entry['source'])


def partition(rows):
    used=set()
    # Prefer games with two selected seats when multiple games use the same layout.
    maps=[]
    for r in sorted(rows,key=lambda r:(-len(r['teachers']),-r['finished'],r['game'])):
        if r['layout'] not in used:
            used.add(r['layout']);maps.append(r)
    for fold,n in [('test',4),('validation',4)]:
        counts=Counter()
        for r in sorted(maps,key=lambda r:(-r['finished'],r['game'])):
            if 'fold' not in r and any(counts[t]<n for t in r['teachers']):
                r['fold']=fold
                counts.update(r['teachers'])
    for r in maps:r.setdefault('fold','train')
    for fold in ('train','validation','test'):
        assert {t for r in maps if r['fold']==fold for t in r['teachers']}==set(range(10)),fold
    return maps


if __name__=='__main__':
    out=Path(sys.argv[1]).resolve()
    teachers=json.loads((out/'teachers.json').read_text());ids={x['id']:i for i,x in enumerate(teachers)}
    sources=json.loads((out/'sources-local.json').read_text())
    dest=out/'encoded';dest.mkdir(exist_ok=True)
    with ProcessPoolExecutor(3,initializer=init) as pool:
        rows=list(pool.map(extract,[(k,v,ids,dest) for k,v in sorted(sources.items())]))
    write(out/'split.json',partition(rows))
