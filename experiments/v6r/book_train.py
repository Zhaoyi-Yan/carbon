"""Learn from 20 observed turns of elite play; evaluate on four disjoint sets of whole opening maps."""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import numpy as np
from book import ROOT,_book_context,_book_row,_book_votes


def load_policy():
    path=ROOT/'yan-v6.py';s=importlib.util.spec_from_file_location('book_frozen',path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m._torch.set_num_threads(1)
    return m


def records(m,data,gid):
    fp=ROOT/'analysis/v6r-artemis-games.jsonl'
    rows=[json.loads(s) for s in fp.read_text().splitlines()]
    rows=[r for r in rows if r['name'] in ('Apollo','Artemis')]
    seats=[]
    for r in rows:
        file=data/'replays'/f"{r['match']}-{r['game']}.json.gz"
        raw=file.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['file_sha256']:
            raise ValueError(file)
        frames=json.loads(gzip.decompress(raw))['steps']
        h={};side=r['seat'];tracker=m.BeliefTracker(m._Bel,side)
        m._ST.clear();prev=None; result=[]
        for t,frame in enumerate(frames[:min(21,len(frames)-1)]):
            shared=frame[0]['observation'];players=shared['players']
            obs=dict(step=t,player=side,players=players,trees=shared['trees'],carbon=frame[side]['observation']['carbon'])
            _book_context(obs,h,m.NB)
            ids={}
            for role in ('PLANTER','COLLECTOR'):
                row=_book_row(obs,h,role,m.FEAT_DIST,m._ROLLOUT_LANDING)
                if row is not None:ids[role]=row
            est,known=tracker.step(players,t,obs['carbon'])
            x=m._quant(m.encode(players,t,side,est,known,prev))
            logits,_=m._v5_root(obs,est,x,prev)
            act=frames[t+1][side]['action'] or {}
            for role,row in ids.items():
                uid,pos,canonical,features,special=row
                label=m.WIDX[act.get(uid) or 'NONE']
                rel=t-h['movers'][role][2]
                result.append(dict(role=role,y=label,canonical=canonical.tolist(),cy=int(np.where(canonical==label)[0][0]),
                             features=features,t=t,age=rel,high=(h['movers'][role][1]==max(players[side][1].values())),
                             enemy=h['enemy'],special=special,
                             p0=m._probs(logits[pos],m._TEMP).tolist(),p1=m._probs(logits[pos],1.).tolist()))
            h['prev']={uid:p for uid,(p,v,k) in players[side][2].items()}
            prev=players;tracker.set_actions(players,act)
        for rec in result:
            rec.update(g=gid[r['layout']],s=int(r['name']=='Artemis'),match=r['match'])
        seats.append(result)
        if len(seats)%25==0:print('seats',len(seats),flush=True)
    return [x for v in seats for x in v]


def make_arrays(recs):
    tables={}
    for role in ['PLANTER','COLLECTOR']:
        rows=[r for r in recs if r['role']==role]
        tables[role]=(np.stack([r['features'] for r in rows]).astype(np.float16).astype(np.float32),
                      np.asarray([r['cy'] for r in rows],int),np.asarray([r['g'] for r in rows],int),
                      np.asarray([r['s'] for r in rows],int))
    return tables


def evaluate(records):
    from collections import defaultdict
    totals=defaultdict(lambda:defaultdict(list));sub=[]
    for fold in range(4):
        train=[r for r in records if r['g']%4!=fold]
        test=[r for r in records if r['g']%4==fold]
        tables=make_arrays(train)
        for r in test:
            role=r['role'];can=r['canonical'];y=r['y']
            # Position and opening-category majority baseline, with symmetric directions.
            xx,yy,gg,ss=tables[role]; f=r['features']
            key=(r['age'] if r['age']<3 else -1,r['enemy'],r['high'])
            matches=[z['cy'] for z in train if z['role']==role and
                     (z['age'] if z['age']<3 else -1,z['enemy'],z['high'])==key and z['s']==r['s']]
            if len(matches)<4:matches=yy
            votes=(.5+np.bincount(matches,minlength=5))/(len(matches)+2.5)
            majority=np.empty(5);majority[can]=votes
            q=_book_votes(tables,(None,None,can,f,r['special']),role)
            p0,p1=np.array(r['p0']),np.array(r['p1'])
            preds={'opening_majority':majority,'book':q,'frozen_T04':p0,'frozen_T1':p1,
                   'mix_35':.65*p1+.35*q}
            groups=['all',role,('Artemis' if r['s'] else 'Apollo')+'_'+role]
            if r['s'] and r['special']:
                groups.append('Artemis_double_pre_tree')
                if r['age']==0:
                    groups.append('Artemis_double_first_step')
                    sub.append(dict(match=r['match'],fold=fold,actual=r['cy'],q=q[can].round(4).tolist(),base=p1[can].round(4).tolist()))
            for g in groups:
                for name,p in preds.items():
                    order=np.argsort(-p)
                    totals[g][name].append([float(order[0]==y),float(y in order[:2]),-np.log(max(1e-12,p[y]))])
    stats={g:{n:dict(n=len(a),accuracy=round(float(np.mean(a,axis=0)[0]),4),
                     top2=round(float(np.mean(a,axis=0)[1]),4),nll=round(float(np.mean(a,axis=0)[2]),4))
              for n,a in models.items()} for g,models in totals.items()}
    return stats,sub


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('table',type=Path)
    p.add_argument('audit',type=Path);p.add_argument('--cache',type=Path)
    a=p.parse_args()
    rows=[json.loads(s) for s in (ROOT/'analysis/v6r-artemis-games.jsonl').read_text().splitlines()]
    maps=sorted({r['layout'] for r in rows if r['name'] in ('Artemis','Apollo')})
    gids={k:i for i,k in enumerate(maps)}
    if a.cache and a.cache.exists():
        import pickle
        with a.cache.open('rb') as f: recs=pickle.load(f)
    else:
        recs=records(load_policy(),a.data,gids)
        if a.cache:
            import pickle
            with a.cache.open('xb') as f:pickle.dump(recs,f)
    tables=make_arrays(recs)
    pack={r+'_'+field:v.astype(ty) for r,cols in tables.items() for field,ty,v in zip('xygs',('f2','u1','i2','u1'),cols)}
    buf=io.BytesIO();np.savez_compressed(buf,**pack)
    with a.table.open('x') as f:
        json.dump({'layouts':len(gids),'seats':153,'packed_npz':base64.b64encode(buf.getvalue()).decode()},f)
    stats,details=evaluate(recs)
    with a.audit.open('x') as f:json.dump(dict(layouts=len(gids),folds=4,map_assignment=gids,metrics=stats,special_first=details),f,indent=2)


if __name__=='__main__':main()
