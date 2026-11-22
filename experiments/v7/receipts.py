"""Inject a CPU estimate of absolute deposits at each side's bases in the next five and ten decisions."""
import base64
import inspect
import io
from pathlib import Path
import numpy as np

MODEL=Path(__file__).with_name('receipt-model.npz')


def _v7_receipt_rows(players,step,last):
    # The forecast uses public fleet, treasury, and depot geometry, including shortest-path interception.
    collectors=[[(p,v) for p,v,k in pl[2].values() if k=='COLLECTOR'] for pl in players]
    bases=[tuple(pl[1].values()) for pl in players]
    features=[];loads=[];home_dist=[];seats=[]
    length=max(1,last+1)
    left=max(0,last+1-step)
    for side,pl in enumerate(players):
        own,enemy=collectors[side],collectors[1-side]
        homes=bases[side]
        if not homes:continue
        enemyhomes=bases[1-side]
        trees=tuple(p for p,age in pl[3].values())
        for p,load in own:
            d=_V7_RECEIPT_DIST[p]
            hb=min(homes,key=d.__getitem__);near=d[hb];other=max(d[b] for b in homes)
            risk=path=block=hunt=16
            hunt_load=0.
            for q,v in enemy:
                gap=d[q]
                if v<=load:
                    risk=min(risk,gap)
                    block=min(block,_V7_RECEIPT_DIST[hb][q])
                    path=min(path,gap+_V7_RECEIPT_DIST[q][hb]-near)
                else:
                    hunt=min(hunt,gap)
                    if gap<=2:hunt_load=max(hunt_load,v-load)
            friends=sum(q!=p and d[q]<=2 for q,_ in own)
            nearby_trees=sum(d[q]<=2 for q in trees)
            enemyhome=min((d[q] for q in enemyhomes),default=15)
            phase=step/length
            features.append((near/7.,other/10.,min(load,400.)/100.,np.log1p(max(0.,load))/5.,
                             min(risk,8)/5.,min(path,8)/4.,min(block,8)/5.,min(hunt,8)/5.,
                             min(hunt_load,200.)/100.,min(enemyhome,9)/7.,friends/3.,
                             min(nearby_trees,5)/3.,len(own)/8.,len(enemy)/8.,len(pl[3])/12.,
                             min(pl[0],250.)/100.,phase,max(0.,1.-left/40.),max(0.,1.-left/12.),
                             float(near==0), float(near==1),float(risk<=2 and path<=2),
                             float(block<=2 and other-near>=3), min(near,other-near)/7.))
            loads.append(load);home_dist.append(near);seats.append(side)
    return (np.asarray(features,dtype=np.float32).reshape(-1,24),np.asarray(loads,dtype=np.float32),
            np.asarray(home_dist,dtype=np.int16),np.asarray(seats,dtype=np.int8))


def _v7_receipts(players,step,last):
    remaining=max(0,last+1-step)
    value=np.zeros((2,2),dtype=np.float32)
    if not remaining:return value
    x,load,dist,seat=_v7_receipt_rows(players,step,last)
    z=_V7_RECEIPT_MODEL
    horizons=min(5,remaining),min(10,remaining)
    for s,pl in enumerate(players):
        n=sum(kind=='COLLECTOR' for pos,cargo,kind in pl[2].values())
        band=0 if n==0 else (1 if n<=2 else (2 if n<=5 else 3))
        phase=0 if step<20 else (1 if step<100 else 2)
        background=z['background'][band,phase]
        value[s]=background[horizons[0]-1],background[horizons[1]-1]
    if not len(x):return value
    hidden=np.tanh(x@z['w0']+z['b0'])
    out=hidden@z['w1']+z['b1']
    reach=_V7_RECEIPT_STEPS[None,:]>=np.maximum(1,dist[:,None])
    logits=out[:,:11].copy();logits[:,:10]=np.where(reach,logits[:,:10],-100.)
    weights=np.exp(logits-logits.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
    incremental=load[:,None]*weights[:,:10]+8.*np.logaddexp(0.,out[:,11:])*reach
    horizons=min(5,remaining),min(10,remaining)
    for s in (0,1):
        landing=incremental[seat==s].sum(axis=0).cumsum()
        value[s]+=landing[horizons[0]-1],landing[horizons[1]-1]
    return value


def build(code):
    # Can be inserted before existing module initialization regardless of parent search variant.
    if 'def _v7_receipts(' in code:raise ValueError('Receipt forecast already present')
    marker='def _act(obs, configuration=None, t0=None):'
    if code.count(marker)!=1:raise ValueError('Expected a compatible strategy entry')
    payload=base64.b64encode(MODEL.read_bytes()).decode()
    init="""_V7_RECEIPT_DIST=tuple(tuple(min(abs(a//15-b//15),15-abs(a//15-b//15))+min(abs(a%15-b%15),15-abs(a%15-b%15)) for b in range(225)) for a in range(225))
_V7_RECEIPT_STEPS=np.arange(1,11)
with np.load(io.BytesIO(base64.b64decode('PAYLOAD')),allow_pickle=False) as _v7_rz:
    _V7_RECEIPT_MODEL=dict(_v7_rz)
""".replace('PAYLOAD',payload)
    return code.replace(marker,init+'\n'+inspect.getsource(_v7_receipt_rows)+'\n'+inspect.getsource(_v7_receipts)+'\n'+marker)
