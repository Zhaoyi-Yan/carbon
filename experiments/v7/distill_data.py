"""Replay logged final search commands from the observer's fog, with grouped layouts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np

import distill as d
from distill_fetch import EXCLUDE
from receipt_data import layout
from branch_replay import step as official_step
from convoy_fit import ROOT
sys.path.append(str(ROOT/'experiments/v6r'))
from localhead_data import load
from benchmark import configure_engine,new_game,observe

WINDOWS=((2,20),(61,83),(151,173),(240,262))
ACTIVE={t for a,b in WINDOWS for t in range(a,b)}
V6='e5c328aa5d18ced597791e7b839088f1a37d18e4e8dc83dd3d54b13d66e877b0'
def fp(obs):
    return hashlib.sha256(json.dumps([obs['full_carbon'],[sorted(p[1].values()) for p in obs['players']]],separators=(',',':')).encode()).hexdigest()
def local(r,side):
    state,env=new_game(r['seed'],300)
    for t,(actions,cash) in enumerate(r['actions_and_post_cash']):
        src=observe(state,side,t)
        official_step(state,env,actions)
        assert [pl[0] for pl in state[0].observation.players]==cash
        yield t,src,list(state[0].observation.players),actions,env.configuration
def public(path,side):
    with gzip.open(path) as f:g=json.load(f)
    for t,(old,nxt) in enumerate(zip(g['steps'],g['steps'][1:])):
        src={k:v for k,v in old[0]['observation'].items() if k!='full_carbon'}
        src.update(old[side]['observation']);src.pop('full_carbon',None);src['player']=side
        yield t,src,nxt[0]['observation']['players'],[p['action'] or {} for p in nxt],g['configuration']
def load_manifest(pub):
    panel=json.loads((ROOT/'experiments/v7/panels.json').read_text())
    seeds=set(panel['development']+[s for fold in ('qualifying','confirmation') for sub in panel[fold] for s in sub])
    forbidden={fp(new_game(s,300)[0][0].observation) for s in seeds}
    # Canonical layout fingerprints from both actual screenshot replays.
    forbidden.update(('6121ec0ff5ccac020570cc3ac18f9dbe4ce60ddab3f00075aa1dd3afa8f15d49',
                      '9dceac18f15f53a192488efc09f80c946b21430ba5420dfc0c70f95d3312ef39'))
    rows=[];seen=set()
    for f in sorted((ROOT/'analysis').glob('v6r-*-05-estuary.jsonl')):
        for line in f.open():
            r=json.loads(line);seat=next(i for i,p in enumerate(r['players']) if p['sha256']==V6)
            assert r['status']=='completed' and r['players'][1-seat]['name']=='yan-v5'
            sig=fp(new_game(r['seed'],300)[0][0].observation)
            assert r['seed'] not in seeds and sig not in forbidden
            rows.append(dict(source='local',group=sig,seat=seat,payload=r,seed=r['seed'],id=f"{r['seed']}-{seat}"))
    for r in json.loads((pub/'index.json').read_text()):
        assert r['match'] not in EXCLUDE
        f=pub/r['file']
        with gzip.open(f) as stream:g=json.load(stream)
        sig=layout(g)
        if sig in forbidden or r['seed'] in seeds:continue
        rows.append(dict(source='public',group=sig,seat=r['seat'],payload=f,seed=r['seed'],id=r['match']))
    fold={}
    for source in ('local','public'):
        ids=sorted({r['group'] for r in rows if r['source']==source},key=lambda x:hashlib.sha256(('distill-search-1'+x).encode()).hexdigest())
        hold=max(4,len(ids)//3)
        for i,g in enumerate(ids):fold[g]='test' if i<hold else 'train'
    return rows,fold,forbidden

def collect(m,row,other):
    observer=row['seat'] if other else 1-row['seat']
    frames=local(row['payload'],observer) if row['source']=='local' else public(row['payload'],observer)
    m._ST.clear();tracker=m.BeliefTracker(m._Bel,observer);previous=None
    stash={k:[] for k in ('x','z','y','role','t','alive','uid','pos','nextpos')}
    original=m._shadow_update
    def capture(obs,x,v,state,*rest):
        original(obs,x,v,state,*rest)
        mem['view']=v
    m._shadow_update=capture
    try:
        for t,o,after,actions,cfg in frames:
            if t>max(ACTIVE):continue
            pl=o['players'];side=1-observer
            est,known=tracker.step(pl,t,o['carbon'])
            est,known=m._sense_tree_carbon(tracker.bel,o,cfg,previous,est,known)
            if t in ACTIVE:
                mem={};x=m._quant(m.encode(pl,t,observer,est,known,previous))
                m._v5_root(o,est,x,previous)
                prior=m._ST['v5_history']['opponent'][0]
                if pl[side][2]:
                    ids,r,z,features=d._ds_features(mem['view'],pl,side,previous,prior,d._ds_experts(mem['view']))
                    n=len(ids)
                    ys=[m.WIDX.get(actions[side].get(uid),0) for uid in ids]
                    pos=[pl[side][2][u][0] for u in ids]
                    dest=[after[side][2][u][0] if u in after[side][2] else -1 for u in ids]
                    vals=(features,z,ys,r,np.full(n,t),[q>=0 for q in dest],[int(u.split('-')[-1]) for u in ids],pos,dest)
                    for key,v in zip(stash,vals):stash[key].append(np.asarray(v))
            tracker.set_actions(pl,actions[observer]);previous=pl;m._ST['prev']=pl
    finally:m._shadow_update=original
    return {k:np.concatenate(v) for k,v in stash.items()}

def main(a):
    configure_engine()
    rows,fold,ban=load_manifest(a.public)
    a.out.mkdir(exist_ok=True,parents=True)
    m=load(a.parent)
    d.NPL=m.NPL;d.N=m.N;d.SIZE=m.SIZE;d._V5_ROTATIONS=m._V5_ROTATIONS;d._net=m._net
    d._ROLLOUT_LANDING=m._ROLLOUT_LANDING;d._DS_LAND=np.asarray(m._ROLLOUT_LANDING)
    records=[]
    for j,row in enumerate(rows):
        grp=fold[row['group']]
        for neg in ((False,True) if grp=='test' else (False,)):
            fn=f"{row['source']}-{row['id']}-{'other' if neg else 'v6'}.npz"
            dest=a.out/fn
            if not dest.exists():
                arr=collect(m,row,neg)
                pending=dest.with_suffix('.tmp')
                with pending.open('wb') as f:np.savez_compressed(f,**arr)
                pending.replace(dest)
            records.append({**{k:v for k,v in row.items() if k!='payload'},'negative':neg,'fold':grp,'file':fn})
        print(j+1,len(rows),row['source'],grp,flush=True)
    (a.out/'index.json').write_text(json.dumps({'records':records,'excluded_layouts':sorted(ban),'windows':WINDOWS},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('public',type=Path);p.add_argument('parent',type=Path);p.add_argument('out',type=Path)
    main(p.parse_args())
