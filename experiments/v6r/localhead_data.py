"""Cache native-frame elite worker supervision from the fixed maps and the frozen CPU network."""
import argparse,base64,hashlib,importlib.util,io,json,re
from pathlib import Path
import numpy as np
import localhead as l

ROOT=Path(__file__).resolve().parents[2]


def load(path):
    spec=importlib.util.spec_from_file_location('elite_localhead',path)
    m=importlib.util.module_from_spec(spec)
    # The frozen production module otherwise requests two CPU threads during import warm-up.
    source=path.read_text().replace("_torch.set_num_threads(2)","_torch.set_num_threads(1)")
    exec(compile(source,str(path),"exec"),vars(m))
    m._torch.set_num_threads(1)
    return m


def collect(m,folder,names,rot,filename):
    l._LOCAL_PLANES=l.PLANES
    land=np.array(m._ROLLOUT_LANDING)
    params=m._POLICY.p;weight=params['hw.w'].reshape(5,-1).T
    stash={k:[] for k in ('a','b','c','z','y','role','game','style')}
    for g,name in enumerate(names):
        with np.load(folder/(name+'.npz'),allow_pickle=False) as z:
            select=np.isin(z['T'],[21,22])
            arr=z['X'][select];styles=z['T'][select].astype(np.int64);labels=z['A'][select,0]
        cells,inverse,acts,channels=m._V5_ROTATIONS[rot]
        for j in range(0,len(arr),12):
            x=arr[j:j+12].reshape(-1,54,225)[:,channels][:,:,inverse].reshape(-1,54,15,15).astype(np.float32)*.01
            ts=styles[j:j+12]
            hidden=m._NET(x,ts,hidden=True)
            rows,pos=np.nonzero(labels[j:j+12]>=0)
            native=cells[pos]
            a,b,c=l._localhead_features(x,hidden,rows,native,land)
            z0=a@weight+params['hw.b']
            role=(x.reshape(-1,54,225)[rows,4,native]>.5)
            values=(a,b,c,z0,labels[j:j+12][rows,pos],role,np.full(len(rows),g),ts[rows])
            for k,v in zip(stash,values):
                stash[k].append(v.astype(np.float16) if k in ('a','b','c','z') else v.astype(np.uint8))
        if g%30==29:print(filename.name,g+1,flush=True)
    with filename.open('xb') as f:np.savez(f,**{k:np.concatenate(vs) for k,vs in stash.items()},acts=acts)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('cache',type=Path)
    p.add_argument('out',type=Path);p.add_argument('--validation-rotations',action='store_true')
    a=p.parse_args();a.out.mkdir(exist_ok=True);m=load(a.parent)
    s=json.loads((ROOT/'experiments/v6r/elite-validation.json').read_text())
    for fold,names in [('train',s['training']),('test',s['validation'])]:
        for r in (range(4) if fold=='test' and a.validation_rotations else (0,)):
            out=a.out/f'{fold}{r}.npz'
            if not out.exists():collect(m,a.cache,names,r,out)
    print('done',flush=True)


if __name__=='__main__':main()
