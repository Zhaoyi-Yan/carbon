"""Record replay hashes and audit subset independence even under rotations, reflections and exchanged seats."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from receipt_check import real_target
from receipt_data import layout


def semantic_layout(game):
    obs=game['steps'][0][0]['observation']
    field=np.asarray(obs['full_carbon'],dtype='<f4').reshape(15,15)
    homes=[]
    for p in obs['players']:
        board=np.zeros((15,15),dtype='u1');board.ravel()[list(p[1].values())]=1
        homes.append(board)
    out=[]
    for flip in (False,True):
        for k in range(4):
            def trans(x):return np.ascontiguousarray(np.rot90(x[:,::-1] if flip else x,k)).tobytes()
            out.append(hashlib.sha256(trans(field)+b''.join(sorted(trans(b) for b in homes))).hexdigest())
    return min(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('target',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();rows=json.loads((a.cache/'index.json').read_text())['games']
    dest=semantic_layout(real_target(a.target));seen={};indices={}
    records=[]
    for r in rows:
        root=Path(r['source']);idx=root/'index.json'
        if root not in indices:
            meta=json.loads(idx.read_text()) if idx.exists() else []
            indices[root]={Path(m.get('file','')).name:m for m in meta if isinstance(m,dict)}
        f=root/'replays'/r['file'];raw=f.read_bytes();g=json.loads(gzip.decompress(raw))
        fp=semantic_layout(g);split='validation' if int(r['layout'][:8],16)%5==0 else 'training'
        if fp==dest or (fp in seen and seen[fp]!=split):raise ValueError(('layout overlap',r['file'],split,seen.get(fp)))
        assert r['layout']==layout(g)
        seen[fp]=split
        m=indices[root].get(f.name,{})
        records.append(dict(file=f.name,sha256=hashlib.sha256(raw).hexdigest(),seed=r['seed'],layout=r['layout'],
                            canonical=fp,split=split,archive_revision=m.get('revision')))
    with a.out.open('x') as f:
        for r in records:f.write(json.dumps(r,separators=(',',':'))+'\n')
    print(len(records),len(seen),dest,flush=True)

if __name__=='__main__':main()
