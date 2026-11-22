"""Fit a spatial cash forecast with per-match centering and multiple future horizons."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import numpy as np
from features import _V5_DIST, _v5_observe, _v5_context
from backbone import load
from territory import _v5_territory
from fit import ridge
from extract import TEACHERS


def init():
    global MODEL
    MODEL = load()


def table(args):
    name, path, game = args
    replay = json.loads(gzip.decompress(path.read_bytes()))
    frames = replay['steps']
    features, outcomes = [], []
    for side, key in enumerate(game['order']):
        if game['strategies'][key]['id'] not in TEACHERS:
            continue
        belief = MODEL.BeliefTracker(MODEL._Bel,side)
        for t, (before,after) in enumerate(zip(frames,frames[1:])):
            obs = before[0]['observation']
            pl = obs['players']
            est,_ = belief.step(pl,t,before[side]['observation']['carbon'])
            if t%8==int(game['id'][:6],16)%8:
                ec,sty = _v5_observe(pl,obs.get('trees',{}),t,298)
                ec,_ = _v5_context(ec,sty,pl,t,298,side)
                land = _v5_territory(pl,est,t,298)*(1-2*side)
                features.append(np.r_[ec,land])
                future = [frames[min(299,t+span)][0]['observation']['players'] for span in (24,58,112)]
                diff = pl[side][0]-pl[1-side][0]
                outcomes.append([((q[side][0]-q[1-side][0])-diff)/1000. for q in future])
            belief.set_actions(pl,after[side]['action'] or {})
    return name, np.array(features), np.array(outcomes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('split','prior','weights','report'):p.add_argument(k,type=Path)
    p.add_argument('snapshots',type=Path,nargs='+')
    p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args()
    split=json.loads(args.split.read_text())
    tr=set(split['training_matches']);va=set(split['validation_matches'])
    games={}
    for root in args.snapshots:
        for g in json.loads((root/'index.json').read_text()):
            path=root/g['file'];name=path.name.split('.')[0]
            if name in tr|va and path.exists():games.setdefault(name,(name,path,g))
    assert len(games)==len(tr|va)
    with ProcessPoolExecutor(args.jobs,initializer=init) as pool:
        rows=list(pool.map(table,games.values(),chunksize=2))
    def matrices(group):
        selected=[(x,y) for n,x,y in rows if n in group]
        return (np.concatenate([x for x,y in selected]), np.concatenate([y for x,y in selected]),
                np.concatenate([x-x.mean(0) for x,y in selected]), np.concatenate([y-y.mean(0) for x,y in selected]))
    X,Y,DX,DY=matrices(tr)
    V,Z,_,_=matrices(va)
    scale=np.maximum(X.std(0),.12)
    design=np.concatenate([X,.75*DX])/scale
    labels=np.concatenate([Y,.75*DY])
    co=ridge(design,labels,220.)/scale[:,None]
    prior=dict(np.load(args.prior))
    score=lambda e:(np.abs(e).mean(0)*1000.).round(2).tolist()
    metrics=dict(future_turns=[24,58,112],unchanged=score(Z),economic48=score(Z-(V[:,:len(prior['economic'])]@prior['economic'])[:,None]),territory=score(Z-V@co),states=[len(X),len(V)])
    prior['territory']=co @ np.array([.18,.60,.22])
    np.savez_compressed(args.weights,**prior)
    args.report.write_text(json.dumps(metrics,indent=2)+'\n')
    print(metrics,flush=True)


if __name__=='__main__':
    main()
