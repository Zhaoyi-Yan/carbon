"""Fit a conditional continuation of a collector's previous move on independent public games."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import convoy as c
from receipt_data import ACT, LAND, TARGET_LAYOUT, layout

ROOT=Path(__file__).resolve().parents[2]
def geometry():
    c._CONVOY_L=np.asarray(LAND)
    p=np.arange(225);dy=abs(p[:,None]//15-p//15);dx=abs(p[:,None]%15-p%15)
    c._CONVOY_D=np.minimum(15-dy,dy)+np.minimum(15-dx,dx)
def read(row):
    with gzip.open(ROOT/row['source']/'replays'/row['file']) as f:g=json.load(f)
    assert layout(g)==row['layout'] and row['layout']!=TARGET_LAYOUT
    return g
def splits(cache):
    rows=json.loads((cache/'index.json').read_text())['games']
    rows=sorted((r for r in rows if r['end']>=125 and r['layout']!=TARGET_LAYOUT),
                key=lambda r:hashlib.sha256(('convoy-public-v1'+r['layout']).encode()).hexdigest())
    return rows[:64],rows[64:88]
def observed(steps,t,side,uid,pos):
    result=[]
    for j in range(1,4):
        nextframe=steps[t+j]
        action=ACT[(nextframe[side]['action'] or {}).get(uid) or 'NONE']
        pos=int(c._CONVOY_L[pos,action])
        unit=nextframe[0]['observation']['players'][side][2].get(uid)
        if unit is None:break
        assert unit[0]==pos
        result.append((pos,action))
    return result
def examples(rows):
    xx=[]; yy=[]
    for row in rows:
        steps=read(row)['steps']
        for t in range(11,len(steps)-3,17):
            pl=steps[t][0]['observation']['players']
            for side in (0,1):
                for uid,(pos,load,k) in pl[side][2].items():
                    if k!='COLLECTOR':continue
                    route=observed(steps,t,side,uid,pos)
                    if len(route)<2:continue
                    ctx=c._convoy_context(pl,side,pos,load)
                    p,a=zip(*route[:-1])
                    xx.append(c._convoy_inputs(np.asarray(p),np.asarray(a),ctx))
                    yy.extend(v for q,v in route[1:])
    return np.concatenate(xx).astype(np.float64),np.asarray(yy)
def train(x,y):
    def loss(w):
        z=x@w;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        n=len(y);cost=-np.log(p[np.arange(n),y]).mean()+.002*(w@w)
        p[np.arange(n),y]-=1
        return cost,np.einsum('na,nak->k',p,x)/n+.004*w
    sol=minimize(loss,np.zeros(x.shape[-1]),jac=True,method='L-BFGS-B',options={'maxiter':95})
    assert sol.success
    return sol.x
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args(); geometry();trainrows,validation=splits(a.cache);x,y=examples(trainrows);w=train(x,y)
    out={'weights':w.round(6).tolist(),'examples':len(y),'training_layouts':[r['layout'] for r in trainrows],
         'validation_layouts':[r['layout'] for r in validation],'target_excluded':TARGET_LAYOUT,
         'model':'Conditional second/third movement given root-visible players, current path cell and prior direction. Survivor conditional.'}
    a.output.write_text(json.dumps(out,separators=(',',':'))+'\n');print(len(y),len(trainrows),len(validation))
