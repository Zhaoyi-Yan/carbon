"""Dense legal Master observations, per-unit action history and auxiliary three-turn future displacement."""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from distill_data import ROOT,load
from master_data import SAMPLES
import master_spatial as sp

def samples(m,row,out):
    with gzip.open(row['file']) as f:g=json.load(f)
    frames=g['steps'];s=row['side'];tr=m.BeliefTracker(m._Bel,s);hist=[]
    items=defaultdict(list);pending=[];training=row['fold']=='train';gid=int(row['group'][:8],16)
    def flush():
        if not pending:return
        batch=np.stack(pending).astype(np.float32);latent=m._net()(batch,np.full(len(batch),22,dtype=np.int64),hidden=True)
        items['h'].extend(latent.astype(np.float16));pending.clear()
    for t,f in enumerate(frames[:-1]):
        obs={**f[0]['observation'],**f[s]['observation']};obs.pop('full_carbon',None);obs['player']=s
        pl=obs['players'];own=pl[s];enemy=pl[1-s];prev=hist[0] if hist else None
        est,known=tr.step(pl,t,obs['carbon']);est,known=m._sense_tree_carbon(tr.bel,obs,g['configuration'],prev,est,known)
        cmd=frames[t+1][s]['action'] or {}
        used=t in SAMPLES or (training and (t+gid)%2==0)
        if used:
            x=m._quant(m.encode(pl,t,s,est,known,prev))
            rotations=((t+gid)%4,) if training else (0,1,2,3)
            for j in rotations:
                cells,inv,acts,chs=m._V5_ROTATIONS[j];xx=x.reshape(m.NPL,225)[chs][:,inv].reshape(m.NPL,15,15)
                trace=sp._mx_trace(pl,s,hist,cells,acts)
                a=np.full((2,225),-1,np.int8);targets=np.zeros((3,225),np.float16);mask=np.zeros(225,np.uint8)
                sites=list(own[1].values());foes=list(enemy[1].values())
                for uid,(q,load_,kind) in own[2].items():
                    rc=cells[q];a[0,rc]=acts[m.WIDX.get(cmd.get(uid),0)];collector=kind=='COLLECTOR'
                    nearhome=min(m.FEAT_DIST[q,k] for k in sites)<=4
                    away=min(m.FEAT_DIST[q,k] for k in foes)<=4
                    prey=any(k=='COLLECTOR' and v>=load_+35 and m.FEAT_DIST[q,p]<=3 for p,v,k in enemy[2].values())
                    mask[rc]=(1*collector+2*(collector and load_>=65 and nearhome)+4*(collector and load_<=15 and away)+8*(collector and load_<=15 and (away or prey)))
                    if t+3<len(frames):
                        later=frames[t+3][0]['observation']['players'][s][2].get(uid)
                        if later is not None:
                            dest=cells[later[0]];dy=(dest//15-rc//15+7)%15-7;dx=(dest%15-rc%15+7)%15-7
                            targets[:,rc]=[dy/3.,dx/3.,1.]
                if own[0]>=30 and len(own[2])<10:
                    for u,q in own[1].items():a[1,cells[q]]=m.BACT.index(cmd[u]) if u in cmd else 0
                for k,v in dict(x=np.rint(xx*100).astype(np.uint8),trace=np.packbits(trace.astype(np.uint8),axis=-1),y=a,goal=targets,mask=mask,t=t,rot=j).items():items[k].append(v)
                pending.append(xx)
                if len(pending)>=8:flush()
        tr.set_actions(pl,cmd);hist=[pl]+hist[:2]
    flush();np.savez(out,**{k:np.asarray(v) for k,v in items.items()})

def main(data):
    rows=json.loads((data/'master-worker/index.json').read_text());dest=data/'master-spatial';dest.mkdir(exist_ok=True)
    m=load(ROOT/'experiments/v7/checkpoints/harbormaster.py');sp._ROLLOUT_LANDING=m._ROLLOUT_LANDING
    for i,r in enumerate(sorted(rows,key=lambda r:(r['fold']!='train',r['group']))):
        path=dest/r['cache']
        if not path.exists():samples(m,r,path)
        print(i+1,len(rows),r['teacher'],r['fold'],flush=True)
    (dest/'index.json').write_text(json.dumps(rows)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
