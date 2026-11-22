"""Retrieve opening worker moves from compatible, visible elite replay positions."""
import argparse
import base64
import hashlib
import inspect
import io
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def _book_context(obs, history, adjacent):
    players,side,t=obs['players'],obs['player'],obs['step']
    own,rival=players[side],players[1-side]
    if not history or t==0:
        history.clear()
        history.update(enemy=0,enemy_planter=-1,movers={},seen=np.zeros(225),known=np.zeros(225),prev={},at=-1)
    if t==history['at']:return
    if t<21:
        for pos,load,role in own[2].values():
            visible=[pos] if role=='COLLECTOR' else adjacent[pos]
            for q in visible:
                history['seen'][q]=obs['carbon'][q]
                history['known'][q]=1.
        if not history['enemy'] and rival[2]:
            units=list(rival[2].values())
            kinds=sum(w[2]=='PLANTER' for w in units)
            history['enemy']=1 if len(units)==2 and kinds==0 else (2 if len(units)==2 and kinds==1 else 3)
            history['enemy_planter']=next((p for p,v,k in units if k=='PLANTER'),-1)
        for uid,(p,load,role) in own[2].items():
            if role not in history['movers'] and p in own[1].values():
                history['movers'][role]=(uid,p,t)
    history['at']=t


def _book_row(obs,history,role,geometry,landing):
    if role not in history['movers']:return
    side,t=obs['player'],obs['step']; own,rival=obs['players'][side],obs['players'][1-side]
    uid,birth,born=history['movers'][role]
    if uid not in own[2]:return
    pos,load,_=own[2][uid]
    fy=1 if birth//15>=7 else -1
    fx=1 if birth%15>=7 else -1
    actions=np.array([0,1 if fy==1 else 3,2 if fx==1 else 4,3 if fy==1 else 1,4 if fx==1 else 2])
    cells=np.array(landing[pos])[actions]
    prev=history['prev'].get(uid,pos)
    last=next((i for i,a in enumerate(actions) if landing[prev][a]==pos),0)
    planter=role=='PLANTER'
    enemytype=history['enemy']
    high=birth==max(own[1].values())
    first_tree=bool(own[3])
    special=planter and enemytype==1 and high and born==2 and not first_tree
    age=t-born
    def offset(cell):
        return [fy*((cell//15-birth//15+7)%15-7),fx*((cell%15-birth%15+7)%15-7)]
    coord=np.array(offset(pos))
    v=[*(coord*.58),min(age,16)*.16,min(t,20)*.10,min(load,150)/70.,float(first_tree)*1.8,
       min(len(own[3]),3)*.45,min(len(own[2]),6)*.22,min(len(rival[2]),6)*.19,
       float(high)*.55,min(born,5)*.28,min(own[0],150.)/120.,float(special)*1.4]
    v.extend(.7*float(enemytype==i) for i in (0,1,2,3))
    v.extend(.65*float(last==i) for i in range(5))
    v.extend((np.array(offset(history['enemy_planter']))*.13).tolist() if history['enemy_planter']>=0 else [0.,0.])
    groups=[[p for k,(p,l,r) in own[2].items() if k!=uid and r=='COLLECTOR'],
            [p for k,(p,l,r) in own[2].items() if k!=uid and r=='PLANTER'],
            [p for p,l,r in rival[2].values() if r=='COLLECTOR'],[p for p,l,r in rival[2].values() if r=='PLANTER'],
            [p for p,a in own[3].values()],[p for p,a in rival[3].values()]]
    for loc in groups:
        d=np.minimum(6.,geometry[cells][:,loc].min(1)) if loc else np.full(5,6.)
        v.extend(.17*d)
    seen,known=history['seen'],history['known']
    # Known-zero and unknown cells stay distinct. No oracle map or opponent-private carbon.
    for dy,dx in ((0,0),(-1,0),(0,1),(1,0),(0,-1),(-2,0),(0,2),(2,0),(0,-2),(-1,-1),(-1,1),(1,1),(1,-1)):
        q=(pos//15+dy*fy)%15*15+(pos%15+dx*fx)%15
        v.extend((min(seen[q],100.)*.021,known[q]*.36))
    return uid,pos,actions,np.asarray(v,np.float32),bool(special)


def _book_votes(book,row,role):
    uid,pos,actions,vec,special=row
    # Rows are ranked across distinct archived games. One neighboring game gets one vote.
    xx,yy,groups,styles=book[role]
    d=np.square(xx-vec).sum(1)
    if special:
        d+=np.where(styles==1,0.,5.)
    k=min(140,len(d))
    nearest=np.argpartition(d,k-1)[:k]
    chosen=[];used=set()
    for j in nearest[np.argsort(d[nearest],kind='stable')]:
        game=int(groups[j])
        if game not in used:
            used.add(game);chosen.append(j)
        if len(chosen)==12:break
    ix=np.array(chosen)
    weights=np.exp(-(d[ix]-d[ix].min())/3.)
    posterior=(.15+np.bincount(yy[ix],weights=weights,minlength=5))/(weights.sum()+.75)
    physical=np.zeros(5)
    physical[actions]=posterior
    return physical


def _book_propose(obs,workers,bases):
    t=obs['step']
    if t>20:return
    h=_ST.setdefault('opening_book',{})
    _book_context(obs,h,NB)
    player=obs['players'][obs['player']]
    proposal=workers.copy(); used=False
    for role in ('PLANTER','COLLECTOR'):
        row=_book_row(obs,h,role,FEAT_DIST,_ROLLOUT_LANDING)
        if row is None:continue
        uid,pos,*_=row
        p=_book_votes(_OPENING_BOOK,row,role)
        if p.max()<.43:continue
        model=_probs(workers[pos],1.)
        mixed=.65*model+.35*p
        proposal[pos]=_TEMP*np.log(np.maximum(mixed,1e-8))
        used=True
    h['prev']={uid:p for uid,(p,c,k) in player[2].items()}
    if used:_ST['v5_history']['alternatives'].insert(0,(proposal,bases))


def _book_hire(obs, config):
    # The evidence concerns exactly this publicly visible reply on turn one.
    if obs['step']!=1:
        return
    me=obs['player']; own,rival=obs['players'][me],obs['players'][1-me]
    if own[2] or len(own[1])!=2 or len(rival[2])!=2:
        return
    if any(role!='COLLECTOR' for pos,cargo,role in rival[2].values()):
        return
    required=(config.get('recCollectorCost',30)+2*config.get('recCollectorIncreaseCost',0)
              +config.get('recPlanterCost',30))
    if own[0]<required or config.get('workerLimit',10)<2:
        return
    low,high=sorted(own[1],key=own[1].get)
    return {low:'RECCOLLECTOR',high:'RECPLANTER'}


def build(source,data=None):
    text=source.decode()
    anchor='def _act(obs, configuration=None, t0=None):'
    candidates='    cands = [act0]; keys = {tuple(sorted(act0.items()))}'
    if text.count(anchor)!=1 or text.count(candidates)!=1:
        raise ValueError('Expected a compatible yan-v6 root search')
    text=text.replace(anchor,inspect.getsource(_book_hire)+'\n\n'+anchor)
    text=text.replace(candidates,candidates+"""
    archived = _book_hire(obs,configuration)
    if archived is not None and archived != act0:
        cands.append(archived)
        keys.add(tuple(sorted(archived.items())))""")
    if data is not None:
        invoke='        lw,lb = _v6_response(obs,est,configuration,lw,lb)'
        if text.count(invoke)!=1:
            raise ValueError('Expected compatible root policy for optional worker book')
        packed=base64.b64encode(data).decode()
        helpers='\n\n'.join(inspect.getsource(f) for f in (_book_context,_book_row,_book_votes,_book_propose))
        load=("_OPENING_BOOK = {}\nwith np.load(io.BytesIO(base64.b64decode('"+packed+"')),allow_pickle=False) as _book_src:\n"
              "    for _kind in ('PLANTER','COLLECTOR'):\n"
              "        _OPENING_BOOK[_kind] = tuple(_book_src[_kind+'_'+_f].astype(_ty) for _f,_ty in "
              "(('x',np.float32),('y',int),('g',int),('s',int)))\n")
        text=text.replace(anchor,load+'\n'+helpers+'\n\n'+anchor)
        text=text.replace(invoke,invoke+'\n        _book_propose(obs,lw,lb)')
    compile(text,'<opening-book>','exec')
    return text


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--workers',type=Path,help='Experimental full worker retrieval table')
    p.add_argument('--parent',type=Path,default=ROOT/'yan-v6.py')
    a=p.parse_args()
    data=base64.b64decode(json.loads(a.workers.read_text())['packed_npz']) if a.workers else None
    text=build(a.parent.read_bytes(),data)
    with a.output.open('x') as f:f.write(text)
    print(hashlib.sha256(text.encode()).hexdigest())
