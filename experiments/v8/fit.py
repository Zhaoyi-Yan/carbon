"""Fit income during uninterrupted tree ownership using public games; hold out whole maps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import option

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/v7')]
from starvation_replay import read_replay, LAND, distance


def make_examples(path):
    game=read_replay(path)
    frames=game['steps']
    cfg=game['configuration']
    active={}
    rows=[]
    for t in range(len(frames)-1):
        prev=frames[t][0]['observation']
        post=frames[t+1][0]['observation']
        for side,current in enumerate(post['players']):
            old=prev['players'][side][3]
            for tid,(q,age) in current[3].items():
                if tid not in old:
                    target=(tid in prev['players'][1-side][3])
                    prior_age=prev['players'][1-side][3][tid][1] if target else 0
                    stock=frames[t][side]['observation']['carbon']
                    x=option._v8_tree_features(prev,cfg,stock,side,q,prior_age,0)
                    rows.append([int(target),x,0.])
                    active[tid]=rows[-1]
                active[tid][2]+=post['trees'][tid][1]
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replays',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    option._V5_DIST=[[distance(i,j) for j in range(225)] for i in range(225)]
    option.NB=[list(m.values())[1:] for m in LAND]
    option.NB8=[{(p//15+y)%15*15+(p%15+x)%15 for x in (-1,0,1) for y in (-1,0,1) if x or y} for p in range(225)]
    games=sorted(args.replays.glob('*.html.gz'),key=lambda f:hashlib.sha256(f.name.encode()).hexdigest())
    assert len(games)==24
    groups=[sum((make_examples(f) for f in subset),[]) for subset in [games[:16], games[16:]]]
    out={'training':[f.name for f in games[:16]],'validation':[f.name for f in games[16:]],'coefficients':[],'validation_metrics':[]}
    for role in (0,1):
        arrays=[(np.array([x for r,x,y in group if r==role]),np.array([y for r,x,y in group if r==role])) for group in groups]
        (x,y),(test,target)=arrays
        coef=np.linalg.solve(x.T@x+6.*np.eye(x.shape[1]),x.T@y)
        predict=np.maximum(0.,test@coef)
        out['coefficients'].append(coef.round(7).tolist())
        out['validation_metrics'].append(dict(kind='capture' if role else 'plant',n_train=len(y),n_test=len(target),mae=round(float(np.abs(predict-target).mean()),2),constant_mae=round(float(np.abs(np.median(y)-target).mean()),2)))
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(out['validation_metrics'])


if __name__=='__main__':
    main()
