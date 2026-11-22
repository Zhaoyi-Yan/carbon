"""Collect future gross arrivals using replay commands; keep the entire target map out of training."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import receipts as impl

TARGET_MATCH='03100a95e8647c51a4d642fe08d856dd'
TARGET_LAYOUT='6121ec0ff5ccac020570cc3ac18f9dbe4ce60ddab3f00075aa1dd3afa8f15d49'
TARGET_SEED=205864169
ROOT=Path(__file__).resolve().parents[2]
LAND=tuple(tuple(((p//15+dy)%15)*15+((p%15+dx)%15) for dy,dx in ((0,0),(-1,0),(0,1),(1,0),(0,-1))) for p in range(225))
ACT={k:i for i,k in enumerate(('NONE','UP','RIGHT','DOWN','LEFT'))}
impl._V7_RECEIPT_DIST=tuple(tuple(min(abs(a//15-b//15),15-abs(a//15-b//15))+min(abs(a%15-b%15),15-abs(a%15-b%15)) for b in range(225)) for a in range(225))


def layout(game):
    ob=game['steps'][0][0]['observation']
    return hashlib.sha256(json.dumps([ob['full_carbon'],[sorted(p[1].values()) for p in ob['players']]],separators=(',',':')).encode()).hexdigest()


def targets(game):
    cfg=game['configuration']
    if not (cfg['moveCost']==cfg['smeltCost']==0 and cfg['size']==15):raise ValueError('Nonstandard transfer rules')
    steps=game['steps'];values=[];events=[]
    for prev,next_ in zip(steps,steps[1:]):
        pl=prev[0]['observation']['players']
        sites={pos:s for s,p in enumerate(pl) for pos in p[1].values()}
        transfers={};cash=[0.,0.]
        for s,p in enumerate(pl):
            action=next_[s]['action'] or {}
            for uid,(pos,cargo,kind) in p[2].items():
                if cargo<=0:continue
                end=LAND[pos][ACT[action.get(uid) or 'NONE']]
                if end in sites:
                    bank=sites[end];cash[bank]+=cargo;transfers[uid]=bank,cargo
        values.append(cash);events.append(transfers)
    return np.asarray(values),events


def collect(game,idx,stride):
    steps=game['steps'];last=game['configuration']['episodeSteps']-2
    amounts,events=targets(game)
    total=np.vstack((np.zeros((1,2)),amounts.cumsum(axis=0)))
    boards=[];workers=[]
    for t in range(0,len(steps)-1,stride):
        pl=steps[t][0]['observation']['players'];end=len(steps)-1
        x,loads,dist,seat=impl._v7_receipt_rows(pl,t,last)
        ids=[uid for p in pl for uid,(pos,v,k) in p[2].items() if k=='COLLECTOR']
        for s in (0,1):
            h=np.minimum(np.arange(1,11),end-t)
            boards.append((idx,t,s,total[t+h,s]-total[t,s],max(0,last+1-t)))
        for i,(uid,s) in enumerate(zip(ids,seat)):
            e=np.zeros(10,dtype=np.float32)
            for j in range(min(10,end-t)):
                dest,money=events[t+j].get(uid,(-1,0.))
                if dest==s:e[j]=money
            workers.append((idx,t,s,x[i],loads[i],dist[i],e))
    return boards,workers


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('cache',type=Path);a.add_argument('dirs',type=Path,nargs='+')
    a.add_argument('--stride',type=int,default=3);args=a.parse_args();args.cache.mkdir(exist_ok=True)
    games={};meta=[];black=[];seen=set()
    for root in args.dirs:
        for f in sorted((root/'replays').glob('*.json.gz')):
            if f.name in seen:continue
            seen.add(f.name)
            if len(games)%25==0:print('maps so far',len(games),flush=True)
            with gzip.open(f) as z:game=json.load(z)
            fp=layout(game);seed=game['configuration']['randomSeed']
            if TARGET_MATCH in f.name or fp==TARGET_LAYOUT or seed==TARGET_SEED:black.append(f.name);continue
            if fp in games or len(game['steps'])<45:continue
            idx=len(games);games[fp]=idx
            output=args.cache/(fp+'.npz')
            if not output.exists():
                b,w=collect(game,idx,args.stride)
                bd={k:np.asarray([r[i] for r in b]) for i,k in enumerate(('group','t','s','y','left'))}
                wd={k:np.asarray([r[i] for r in w]) for i,k in enumerate(('wg','wt','ws','x','load','d','ey'))}
                with output.open('xb') as tmp:np.savez_compressed(tmp,**bd,**wd)
            meta.append({'layout':fp,'seed':seed,'file':f.name,'source':str(root),'end':len(game['steps'])-1})
    (args.cache/'index.json').write_text(json.dumps({'games':meta,'excluded_target':black,'target_layout':TARGET_LAYOUT,'stride':args.stride}))
    print('done',len(meta),black,flush=True)

if __name__=='__main__':main()
