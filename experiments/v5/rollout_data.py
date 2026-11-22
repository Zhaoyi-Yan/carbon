"""Cache observable board tensors and on-policy teacher logits for fast rollout distillation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import numpy as np
from backbone import load
from extract import TEACHERS


def init():
    global MODEL, NET
    MODEL=load();NET=MODEL._net()


def prepare(task):
    g,path,out=task
    if out.exists():return
    frames=json.loads(gzip.decompress(path.read_bytes()))['steps']
    all_x=[];all_styles=[];labels=[];teachers=[];values=[];scalars=[];future=[]
    for side,slot in enumerate(g['order']):
        teacher=TEACHERS.get(g['strategies'][slot]['id'])
        if teacher is None:continue
        tr=MODEL.BeliefTracker(MODEL._Bel,side);prev=None
        phase=int(g['id'][:5],16)%6
        for t,(before,after) in enumerate(zip(frames,frames[1:])):
            pl=before[0]['observation']['players'];action=after[side]['action'] or {}
            est,known=tr.step(pl,t,before[side]['observation']['carbon'])
            if t%6==phase:
                x=MODEL.encode(pl,t,side,est,known,prev)
                all_x.append(np.uint8(np.rint(x*100.).clip(0,255)))
                all_styles.append(teacher)
                label=np.full((2,225),-1,np.int8)
                for uid,(pos,_,_) in pl[side][2].items():label[0,pos]=MODEL.WIDX.get(action.get(uid),0)
                if pl[side][0]>=30 and len(pl[side][2])<10:
                    for uid,pos in pl[side][1].items():label[1,pos]=MODEL.BIDX.get(action.get(uid),0)
                labels.append(label);scalars.append(MODEL._scalars(pl,t,side))
                score=[]
                for span in (32,80):
                    at=frames[min(299,t+span)][0]['observation']['players']
                    score.append((at[side][0]-at[1-side][0])/3000.)
                future.append(score)
            prev=pl;tr.set_actions(pl,action)
    xx=np.stack(all_x);ss=np.stack(scalars)
    for i in range(0,len(xx),16):
        w,b,v=NET(xx[i:i+16].astype(np.float32)*.01,MODEL._SID,ss[i:i+16])
        teachers.extend(np.concatenate([w,b],axis=2).astype(np.float16))
        values.extend(v)
    np.savez_compressed(out,X=xx,T=np.array(all_styles,np.int8),A=np.stack(labels),L=np.stack(teachers),
                        V=np.array(values,np.float32),S=ss,Y=np.array(future,np.float32))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path);p.add_argument('split',type=Path)
    p.add_argument('snapshots',type=Path,nargs='+');p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args();a.cache.mkdir(exist_ok=True,parents=True)
    v=json.loads(a.split.read_text());names=set(v['training_matches']+v['validation_matches'])
    games={}
    for root in a.snapshots:
        for g in json.loads((root/'index.json').read_text()):
            path=root/g['file'];name=path.name.split('.')[0]
            if path.exists() and name in names:games.setdefault(name,(g,path,a.cache/(name+'.npz')))
    with ProcessPoolExecutor(a.jobs,initializer=init) as pool:
        for i,_ in enumerate(pool.map(prepare,games.values(),chunksize=2),1):
            if i%80==0:print(i,flush=True)


if __name__=='__main__':
    main()
