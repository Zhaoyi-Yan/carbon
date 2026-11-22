"""Collect alternative opponent actions strictly on prior training maps for identification."""
import argparse
import json
import numpy as np
from pathlib import Path
import distill as d
from distill_data import load_manifest,load,collect,configure_engine

def main(a):
    configure_engine(); rows,fold,_=load_manifest(a.public)
    model=load(a.parent)
    d.NPL=model.NPL;d.N=model.N;d.SIZE=model.SIZE
    d._V5_ROTATIONS=model._V5_ROTATIONS;d._net=model._net;d._ROLLOUT_LANDING=model._ROLLOUT_LANDING
    d._DS_LAND=np.asarray(model._ROLLOUT_LANDING)
    done=[];seen=set();a.cache.mkdir(exist_ok=True)
    for r in rows:
        if fold[r['group']]!='train' or r['group'] in seen:continue
        if sum(k['source']==r['source'] for k in done)>=7:continue
        seen.add(r['group']);name=f"{r['source']}-{r['id']}.npz";file=a.cache/name
        if not file.exists():
            values=collect(model,r,True);tmp=file.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,**values)
            tmp.replace(file)
        done.append(dict(file=name,group=r['group'],source=r['source'],id=r['id']));print(len(done),r['source'],flush=True)
    (a.cache/'index.json').write_text(json.dumps(done,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('public',type=Path);p.add_argument('parent',type=Path);p.add_argument('cache',type=Path)
    main(p.parse_args())
