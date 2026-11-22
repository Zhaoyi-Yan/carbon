"""A linear correction to team receipts from the two preceding public worker states."""
import inspect
import json
from pathlib import Path
import numpy as np


def _v7_motion_features(players, prev, older, base, step, last):
    remaining=max(0,last+1-step)
    rows=[]
    for s,pl in enumerate(players):
        oldworkers=prev[s][2] if prev is not None else {}
        earlier=older[s][2] if older is not None else {}
        homes=tuple(pl[1].values())
        side=[]
        for i,requested in enumerate((5,10)):
            h=min(remaining,requested)
            features=[1.,min(float(base[s,i]),1200.)/100.,min(1.,remaining/10.),min(1.,pl[0]/100.)]+[0.]*12
            for uid,(cell,load,role) in pl[2].items():
                old=oldworkers.get(uid)
                if role!='COLLECTOR' or old is None or not homes or load<2:continue
                near=min(_V7_RECEIPT_DIST[cell][b] for b in homes)
                if near>h:continue
                former,old_load,_=old
                d0=min(_V7_RECEIPT_DIST[former][b] for b in homes)
                gain=d0-near
                prior=earlier.get(uid)
                prior_gain=min(_V7_RECEIPT_DIST[prior[0]][b] for b in homes)-d0 if prior is not None else 0
                still=cell==former
                change=load-old_load
                # A prompt receipt can contain only the cargo already aboard plus future collection.
                weight=min(load,650.)/100. / (1.+.22*near)
                observations=(gain,prior_gain,float(gain==1 and prior_gain==1),float(gain==-1 and prior_gain==-1),
                              float(still),float(still and prior is not None and prior[0]==former),
                              float(still and change>2.),float(not still and change>2.),float(change<-2.),
                              gain*float(near<=2),gain*float(load>=100),min(1.,max(-1.,change/max(25.,load))))
                for j,v in enumerate(observations):
                    features[4+j]+=weight*v
            side.append(features)
        rows.append(side)
    return np.asarray(rows)


def _v7_motion_apply(base,x,coefficients,left):
    raw=np.maximum(0.,base+np.sum(x*coefficients,axis=-1))
    raw[:,1]=raw[:,0] if left<=5 else np.maximum(raw[:,0],raw[:,1])
    return raw


def build(code, model=None):
    model=model or Path(__file__).with_name('receipt-motion-model.json')
    weights=json.loads(Path(model).read_text())['weights']
    marker='def _v7_flows(board):'
    old="        board['_v7_flow']=_v7_receipts(board['players'],board['step'],LAST_TURN)"
    sim="'_sap_previous':o['players']}"
    remember="    st['prev'] = players"
    for term in (marker,old,sim,remember):
        if code.count(term)!=1:raise ValueError('Missing or ambiguous v7 receipt anchor: '+term)
    helpers='\n\n'.join(inspect.getsource(f) for f in (_v7_motion_features,_v7_motion_apply))
    code=code.replace(marker,'_V7_MOTION_COEF=np.asarray('+repr(weights)+')\n\n'+helpers+'\n\n'+marker)
    code=code.replace(sim,"'_sap_previous':o['players'], '_v7_motion_older':o.get('_sap_previous',_ST.get('prev'))}")
    code=code.replace(remember,"    st['receipt_previous2'] = st.get('prev')\n"+remember)
    code=code.replace(old,"""        base=_v7_receipts(board['players'],board['step'],LAST_TURN)
        left=LAST_TURN+1-board['step']
        prev=board.get('_sap_previous',_ST.get('prev'))
        older=board.get('_v7_motion_older',_ST.get('receipt_previous2'))
        if left>0:
            x=_v7_motion_features(board['players'],prev,older,base,board['step'],LAST_TURN)
            base=_v7_motion_apply(base,x,_V7_MOTION_COEF,left)
        board['_v7_flow']=base""")
    return code
