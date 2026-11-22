"""Cache real Master commands and root features from the acting team's observed board."""
import argparse,gzip,hashlib,json,sys,types
from collections import Counter
from pathlib import Path
import numpy as np
import master
from distill_data import ROOT,fp
from receipt_data import layout,targets
from benchmark import new_game,configure_engine

NAMES=('Master7sD','Master7sB2','Master7sE')
SAMPLES=set(range(0,66,2))|set(range(70,290,8))
def manifest(roots):
    banned=set(json.loads((ROOT/'analysis/v7/distill-validation.json').read_text())['excluded_layouts'])
    seen=set();rows=[]
    for d in roots:
        for r in json.loads((d/'index.json').read_text()):
            for s,k in enumerate(r['order']):
                suffix=r['strategies'][k]['name'].split('-')[-1]
                key=(r['id'],r['game'],s);file=d/r['file']
                if suffix not in NAMES or key in seen or not file.exists():continue
                seen.add(key)
                if r['id'] in ('03100a95e8647c51a4d642fe08d856dd','4063a34afedb6d4767a9118e1dee9e5d'):continue
                with gzip.open(file) as f:g=json.load(f)
                assert g['rewards']==[r['scores'][v] for v in r['order']]
                sig=layout(g)
                if sig in banned:continue
                rows.append(dict(file=str(file),game=r['game'],match=r['id'],side=s,teacher=suffix,group=sig,seed=g['configuration']['randomSeed']))
    groups={r['group'] for r in rows};test=set()
    for name in NAMES:
        pool=sorted({r['group'] for r in rows if r['teacher']==name},key=lambda q:hashlib.sha256(('master-cash-v1'+q).encode()).hexdigest())
        test.update(pool[:max(3,len(pool)//4)])
    for r in rows:r['fold']='test' if r['group'] in test else 'train'
    return rows

def policy(path):
    source=path.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)')
    source=source.replace(master.ROT_ANCHOR,"    _MS_CAPTURE['h']=latent[::2]\n"+master.ROT_ANCHOR)
    source=source.replace(master.ROOT_ANCHOR,master.ROOT_ANCHOR+"\n    _MS_CAPTURE['own']=own")
    m=types.ModuleType('master_collector');m._MS_CAPTURE={};exec(compile(source,str(path),'exec'),m.__dict__);m._torch.set_num_threads(1)
    master.NPL=m.NPL;master.N=m.N;master._MS_LAND=np.asarray(m._ROLLOUT_LANDING);master._ROLLOUT_LANDING=m._ROLLOUT_LANDING;master._V5_ROTATIONS=m._V5_ROTATIONS
    return m
def deep_comparison(m,x):
    batch=[];rots=(m._V5_ROTATIONS[0],m._V5_ROTATIONS[2])
    for _,inverse,_,channels in rots:
        a=x.reshape(m.NPL,225)[channels][:,inverse].reshape(m.NPL,15,15);batch.extend((a,a))
    w,b,_=m._net()(np.stack(batch).astype(np.float32),np.array([1,2,1,2]))
    ws=np.zeros((2,225,5),np.float32);bs=np.zeros((2,225,3),np.float32)
    for j,(cells,_,acts,_) in enumerate(rots):
        ws+=.5*w[2*j:2*j+2][:,cells][:,:,acts];bs+=.5*b[2*j:2*j+2][:,cells]
    return ws,bs
def extract(m,r):
    with gzip.open(r['file']) as f:g=json.load(f)
    s=r['side'];frames=g['steps'];m._ST.clear();tr=m.BeliefTracker(m._Bel,s);previous=None
    data={k:[] for k in ('wx','bx','wz','bz','wy','by','wr','wt','bt','blocked','wpoor','bpoor','homeward','cash','prehire')}
    receipts,events=targets(g);econ={}
    for limit in (10,20,35,50,80):
        if limit>=len(frames):continue
        own=frames[limit][0]['observation']['players'][s]
        econ[str(limit)]=[round(float(receipts[:limit,s].sum()),2),round(own[0],2),sum(k=='COLLECTOR' for p,v,k in own[2].values()),len(own[3])]
    first_deposit=next((i+1 for i,v in enumerate(receipts[:,s]) if v>0),None)
    first_tree=next((i for i,f in enumerate(frames) if f[0]['observation']['players'][s][3]),None)
    for t,frame in enumerate(frames[:-1]):
        obs={k:v for k,v in frame[0]['observation'].items() if k!='full_carbon'}
        obs.update(frame[s]['observation']);obs.pop('full_carbon',None);obs['player']=s
        pl=obs['players'];own=pl[s];foe=pl[1-s];cmd=frames[t+1][s]['action'] or {}
        est,known=tr.step(pl,t,obs['carbon'])
        est,known=m._sense_tree_carbon(tr.bel,obs,g['configuration'],previous,est,known)
        if t in SAMPLES:
            x=m._quant(m.encode(pl,t,s,est,known,previous));mine=m._v5_root(obs,est,x,previous);mem=m._MS_CAPTURE
            uid,bid,role,wz,bz,wx,bx=master._ms_rows(pl,s,t,x,previous,mem['own'],mine,mem['h'])
            # Dedicated style-1/2 (two rotations each) form an offline strong baseline.
            w,b=deep_comparison(m,x)
            if uid:
                pos=[own[2][u][0] for u in uid];ys=[m.WIDX.get(cmd.get(u),0) for u in uid]
                danger=[own[2][u][2]=='COLLECTOR' and own[2][u][1]>=65 and any(ek=='COLLECTOR' and ev<=own[2][u][1] and m.FEAT_DIST[p,q]<=2 for q,ev,ek in foe[2].values()) for u,p in zip(uid,pos)]
                vals=(wx,np.vstack((wz,w[:,pos])).transpose(1,0,2),ys,role,[t]*len(uid),danger,[own[0]<145]*len(uid),[float(wx[j,a,5]) for j,a in enumerate(ys)])
                for k,v in zip(('wx','wz','wy','wr','wt','blocked','wpoor','homeward'),vals):data[k].append(np.array(v))
            # An unaffordable base cannot execute any choice; score only when an order is legal.
            if own[0]>=30 and len(uid)<10:
                bp=[own[1][u] for u in bid];ys=[m.BACT.index(cmd[u]) if u in cmd else 0 for u in bid]
                vals=(bx,np.vstack((bz,b[:,bp])).transpose(1,0,2),ys,[t]*len(bid),[own[0]<145]*len(bid),[own[0]]*len(bid),[len(uid)]*len(bid))
                for k,v in zip(('bx','bz','by','bt','bpoor','cash','prehire'),vals):data[k].append(np.array(v))
        tr.set_actions(pl,cmd);previous=pl;m._ST['prev']=pl
    out={k:np.concatenate(v) for k,v in data.items()};out['wx']=out['wx'].astype(np.float16);out['bx']=out['bx'].astype(np.float16)
    return out,dict(steps=econ,first_deposit=first_deposit,first_tree=first_tree,opening=sorted((frames[1][s]['action'] or {}).values()))
def main(a):
    rows=manifest(a.roots);a.out.mkdir(exist_ok=True,parents=True);m=policy(a.parent)
    for j,r in enumerate(rows):
        f=a.out/f"{r['match']}-{r['game']}-{r['side']}.npz";e=f.with_suffix('.json');r['cache']=f.name
        if not f.exists() or not e.exists():
            v,ec=extract(m,r);tmp=f.with_suffix('.tmp')
            with tmp.open('wb') as ff:np.savez_compressed(ff,**v)
            tmp.replace(f);e.write_text(json.dumps(ec))
        r['economy']=json.loads(e.read_text());print(j+1,len(rows),r['teacher'],r['fold'],flush=True)
    (a.out/'index.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('out',type=Path);p.add_argument('roots',nargs='+',type=Path)
    main(p.parse_args())
