"""Build partial-observation training examples for both seats of CarbonEdge games."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/v5')]
from benchmark import configure_engine, limit_math_threads, load_strategy
from unit_credit import outcome_weights

ELITE={'dd1f117a64c676225f0c6333b1cdd1e7':21,'226891089ba7f54b198c8f8dc771dfc2':22}
OLD={p['id']:i for i,p in enumerate(json.loads((ROOT/'experiments/v5/design.json').read_text())['ranking'])}


def init():
    global model
    configure_engine();limit_math_threads()
    path=ROOT/'yan-v5.py'
    # Encoding needs the observation tracker, but no policy inference.
    code=path.read_text().rsplit('\n_net()',1)[0]
    model=load_strategy('economic_features',path,code.encode())


def extract(task):
    row,path,out=task
    if out.is_file():return
    replay=json.loads(gzip.decompress(path.read_bytes()))
    f=replay['steps'];end=len(f)-1
    if end!=row['steps']:raise ValueError(row['id'])
    first=f[0][0]['observation']
    fp=hashlib.sha256(json.dumps([first['full_carbon'],[sorted(p[1].values()) for p in first['players']]],separators=(',',':')).encode()).hexdigest()
    x=[];sc=[];labels=[];target=[];styles=[];weights=[]
    cash=[z[0]['observation']['players'][0][0]-z[0]['observation']['players'][1][0] for z in f]
    offset=int(fp[:5],16)%4
    for side,key in enumerate(row['order']):
        teacher=row['strategies'][key]['id']
        style=ELITE.get(teacher,OLD.get(teacher,20))
        tr=model.BeliefTracker(model._Bel,side);prev=None
        for t in range(end):
            state=f[t][0]['observation'];pl=state['players'];action=f[t+1][side]['action'] or {}
            est,known=tr.step(pl,t,f[t][side]['observation']['carbon'])
            if t%4==offset:
                x.append(np.rint(model.encode(pl,t,side,est,known,prev)*100).clip(0,255).astype('u1'))
                sc.append(model._scalars(pl,t,side))
                a=np.stack(model.labels(pl,side,action))
                if pl[side][0]<30 or len(pl[side][2])>=10:a[1].fill(-1)
                labels.append(a);styles.append(style)
                flow=(.28*cash[min(end,t+20)]+.52*cash[min(end,t+52)]+.20*cash[min(end,t+100)])
                target.append((1-2*side)*flow/3000.)
                weights.append(outcome_weights(f,t,side))
            tr.set_actions(pl,action);prev=pl
    tmp=out.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,X=x,S=sc,A=labels,Y=np.asarray(target,'f'),T=np.asarray(styles,'i1'),W=weights,layout=fp)
    tmp.replace(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path);p.add_argument('snapshots',type=Path,nargs='+');p.add_argument('--jobs',type=int,default=3)
    args=p.parse_args();args.cache.mkdir(exist_ok=True,parents=True)
    games={}
    for root in args.snapshots:
        index=next((root/k for k in ('index.json','snapshot-index.json') if (root/k).is_file()),None)
        if index is None:continue
        for row in json.loads(index.read_text()):
            path=root/row['file'];name=path.name.split('.')[0]
            if any(s['id'] in ELITE for s in row['strategies'].values()) and path.is_file():
                games[name]=(row,path,args.cache/(name+'.npz'))
    tasks=[g for g in games.values() if not g[2].is_file()]
    print('source maps',len(games),'to encode',len(tasks),flush=True)
    with ProcessPoolExecutor(args.jobs,initializer=init) as pool:
        for i,_ in enumerate(pool.map(extract,tasks,chunksize=2),1):
            if i%40==0:print(i,flush=True)
    (args.cache/'index.json').write_text(json.dumps({n:[r['finished'],r['strategies']] for n,(r,_,_) in games.items()})+'\n')

if __name__=='__main__':main()
