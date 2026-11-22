"""Use real future returns to weight expert decisions that improved a comparable position."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from extract import TEACHERS
from features import _v5_observe, _v5_context


def initialize(weights):
    global ECONOMIC
    ECONOMIC=np.load(weights)['economic']


def credit_one(task):
    game, path, target, cache=task
    frames=json.loads(gzip.decompress(path.read_bytes()))['steps']
    first=frames[0][0]['observation']
    layout=[first['full_carbon'],[sorted(p[1].values()) for p in first['players']]]
    phase=int(hashlib.sha256(json.dumps(layout,separators=(',',':')).encode()).hexdigest()[:8],16)%5
    info=[]
    for t,frame in enumerate(frames):
        obs=frame[0]['observation'];pl=obs['players']
        ec,b=_v5_observe(pl,obs.get('trees',{}),t,298)
        features,_=_v5_context(ec,b,pl,t,298,0)
        diff=(pl[0][0]-pl[1][0])/1000.
        future=diff+features@ECONOMIC if t<299 else diff
        info.append((diff,future))
    rows=[];teams=[]
    for side,key in enumerate(game['order']):
        tid=TEACHERS.get(game['strategies'][key]['id'])
        if tid is None:continue
        sign=1-2*side
        for t in range(phase,299,5):
            pl=frames[t][0]['observation']['players'][side]
            count=len(pl[2])+(len(pl[1]) if pl[0]>=30 and len(pl[2])<10 else 0)
            signal=sum(w*(info[min(t+span,299)][1]-info[t][1]) for w,span in ((.45,12),(.35,32),(.20,72)))
            rows.extend([sign*signal]*count);teams.extend([tid]*count)
    with np.load(cache) as old:
        assert np.array_equal(teams,old['T']),path
    np.save(target,np.array(rows,dtype=np.float32))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('out','cache','weights'):p.add_argument(k,type=Path)
    p.add_argument('snapshots',type=Path,nargs='+')
    p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    games={}
    for root in args.snapshots:
        for g in json.loads((root/'index.json').read_text()):
            path=root/g['file'];name=path.name.split('.')[0];cache=args.cache/(name+'.npz');target=args.out/(name+'.npy')
            if path.exists() and cache.exists() and not target.exists():
                games.setdefault(name,(g,path,target,cache))
    with ProcessPoolExecutor(args.jobs,initializer=initialize,initargs=(args.weights,)) as pool:
        list(pool.map(credit_one,games.values(),chunksize=2))
    print('credited',len(games),'games',flush=True)


if __name__=='__main__':
    main()
