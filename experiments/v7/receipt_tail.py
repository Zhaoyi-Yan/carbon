"""Estimate large ten-turn team deposits using affordable mining followed by a route to a depot."""
import inspect
import json
from pathlib import Path
import numpy as np


def _v7_tail_features(players,stock,previous,flow,step,last):
    side_features=[]
    dist=_V7_RECEIPT_DIST
    hard_horizon=max(0,min(10,last+1-step))
    trees={pos for p in players for pos,age in p[3].values()}
    depots={pos for p in players for pos in p[1].values()}
    blocked=trees|depots
    for side,own in enumerate(players):
        homes=list(own[1].values())
        enemies=[(q,v) for q,v,k in players[1-side][2].values() if k=='COLLECTOR']
        old=previous[side][2] if previous is not None else {}
        loads=[];heavy=returning=low=loot=0.;mine=[];rich_low=rich_heavy=bank_then=stationary=0.
        for uid,(p,c,role) in own[2].items():
            if role!='COLLECTOR' or not homes:continue
            prior=old.get(uid,(p,c,role));d=min(dist[p][h] for h in homes)
            if d>hard_horizon:continue
            load=min(c,900.);loads.append(load)
            toward=min(dist[prior[0]][h] for h in homes)>d
            weight=1./(1.+.1*d)
            if c>=90:heavy+=load*weight;returning+=load*weight*int(toward)
            else:low+=1
            nearby_haul=max((v-c for q,v in enemies if dist[p][q]<=2 and v>c+35),default=0.)
            loot+=min(nearby_haul,250.)*weight
            best=0.;on_spot=0.
            for q in _V7_TAIL_L[p]:
                if q in blocked:continue
                home=min(dist[q][h] for h in homes)
                spare=hard_horizon-home-int(q!=p)
                if spare<1:continue
                # A stationary collector can take 30%, then 21%, then 14.7%, before returning.
                harvest=float(stock[q])*(1.-.7**min(5,spare))
                safety=.58 if any(dist[q][a]<=1 and b<=c for a,b in enemies) else 1.
                opportunity=harvest*safety/(1.+.12*home+.28*int(q!=p))
                best=max(best,opportunity)
                if q==p:on_spot=opportunity
            mine.append(best)
            if c<85:rich_low+=best
            else:rich_heavy+=best
            if prior[0]==p and c>prior[1]+2:stationary+=on_spot
            if toward and c>=90:bank_then+=best
        loads.sort(reverse=True);mine.sort(reverse=True)
        a,b=map(float,flow[side]);f=b/300.
        # First five fields are a calibration on the frozen cash model; next nine use public fleet; final six use legal soil.
        side_features.append((1.,a/200.,f,min(f,3.)**2,max(0.,f-1.3),
                              len(loads)/7.,(sum(loads)/500.),sum(loads[:2])/400.,heavy/400.,returning/300.,low/6.,loot/350.,
                              step/300.,hard_horizon/10.,sum(mine)/200.,sum(mine[:2])/130.,rich_low/170.,rich_heavy/160.,
                              stationary/100.,bank_then/100.))
    return np.asarray(side_features)


def _v7_tail_forward(x,flow,model):
    z=(x-model['mean'])/model['scale']
    logits=z@model['logits'].T+model['bias']
    probs=1./(1.+np.exp(-np.clip(logits,-30.,30.)))
    probs[:,1]=np.minimum(probs[:,0],probs[:,1])
    amount=np.maximum(0.,flow[:,1]+z@model['cash']+model['cash_b'])
    excess=np.maximum(0.,z@model['excess']+model['excess_b'])
    return np.column_stack((probs,amount,excess))


def _v7_receipt_tail(board):
    if LAST_TURN+1-board['step']<10:return None
    if '_v7_tail' not in board:
        flow=(_v7_flows(board) if '_V7_MOTION_COEF' not in globals() else
              _v7_receipts(board['players'],board['step'],LAST_TURN))
        prior=board.get('_sap_previous',_ST.get('prev'))
        x=_v7_tail_features(board['players'],board['carbon'],prior,flow,board['step'],LAST_TURN)
        board['_v7_tail']=_v7_tail_forward(x,flow,_V7_TAIL_MODEL)
    return board['_v7_tail']


def build(code,model=None):
    marker='def _v7_flows(board):'
    if code.count(marker)!=1:raise ValueError('Expected v7 receipt forecast')
    param=json.loads(Path(model or Path(__file__).with_name('receipt-tail-model.json')).read_text())
    setup="_V7_TAIL_MODEL={k:np.asarray(v) for k,v in "+repr(param)+".items()}\n_V7_TAIL_L=tuple(tuple(map(int,row)) for row in _V5_REACH)\n"
    helpers='\n\n'.join(inspect.getsource(f) for f in (_v7_tail_features,_v7_tail_forward,_v7_receipt_tail))
    return code.replace(marker,setup+'\n'+helpers+'\n\n'+marker)
