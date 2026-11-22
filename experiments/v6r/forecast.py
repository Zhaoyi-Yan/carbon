"""Public-board features and an antisymmetric two-horizon cash-flow forecast for search leaves."""
import numpy as np
from functools import lru_cache

_FC_DIST = tuple(tuple(min(abs(a//15-b//15),15-abs(a//15-b//15))+min(abs(a%15-b%15),15-abs(a%15-b%15)) for b in range(225)) for a in range(225))
_FC_HISTORY = {}


@lru_cache(maxsize=256)
def _fc_home(bases):
    return tuple(min((_FC_DIST[p][b] for b in bases),default=15) for p in range(225))


def _fc_summary(players, meta):
    return tuple((p[0],sum(v for _,v,role in p[2].values() if role=="COLLECTOR"),
                  sum(meta.get(k,(0,0))[1] for k in p[3]),len(p[2]),len(p[3])) for p in players)


def _fc_track(obs):
    step=obs["step"]
    if step==0 or (step-1 not in _FC_HISTORY and step not in _FC_HISTORY):
        _FC_HISTORY.clear()
    _FC_HISTORY[step]=_fc_summary(obs["players"],obs.get("trees",{}))
    _FC_HISTORY.pop(step-36,None)


def _fc_inputs(players,meta,t,last,history):
    snapshot=_fc_summary(players,meta)
    # A simulated leaf can use public history before the real root and its own simulated balance.
    lag8=history.get(max(0,t-8),snapshot)
    lag24=history.get(max(0,t-24),snapshot)
    turns=max(0,last+1-t)
    collectors=[[(p,v) for p,v,k in s[2].values() if k=="COLLECTOR"] for s in players]
    planters=[[p for p,_,k in s[2].values() if k=="PLANTER"] for s in players]
    bases=[_fc_home(tuple(s[1].values())) for s in players]
    per=[]
    for side,player in enumerate(players):
        bank,cargo,rate,nw,nt=snapshot[side]
        cs=collectors[side];enemy=collectors[1-side];home=bases[side];op_home=bases[1-side]
        near3=near6=reachable=covered=endangered=far=bank_dist=empty_pressure=0.
        for pos,v in cs:
            d=home[pos];dist=_FC_DIST[pos]
            killer=min((dist[q] for q,w in enemy if w<=v),default=15)
            near3+=v*(d<=3);near6+=v*(d<=6);reachable+=v*(d<=turns)
            covered+=v*(killer>=3);endangered+=v*(killer<=1)
            far+=v*(d>=7);bank_dist+=d;empty_pressure+=(v<20 and op_home[pos]<=4)
        flow20=flow50=young=old=exposed=good=defended=0.
        for uid,(pos,age) in player[3].items():
            pay=meta.get(uid,(None,0.))[1];life=max(0,50-age);dist=_FC_DIST[pos]
            rival=min((dist[q] for q in planters[1-side]),default=15)
            guards=min((dist[q] for q,v in cs if v<35),default=15)
            flow20+=pay*min(20,turns,life);flow50+=pay*min(50,turns,life)
            young+=age<=8;old+=age>=38;good+=pay>=3
            exposed+=pay*(rival<=3);defended+=pay*(guards<rival)
        d8=lag8[side];d24=lag24[side]
        per.append([min(bank,150)/100,min(bank,500)/300,min(bank,3000)/1800,
            len(cs)/6,len(planters[side])/4,nt/7,cargo/400,rate/25,near3/300,near6/400,reachable/400,covered/400,
            endangered/200,far/200,bank_dist/30,empty_pressure/4,flow20/400,flow50/600,young/4,old/4,
            exposed/15,defended/20,good/4,
            (bank-d8[0])/250,(bank-d24[0])/500,(cargo-d8[1])/250,(cargo-d24[1])/400,
            (rate-d8[2])/20,(rate-d24[2])/20,(nw-d8[3])/3,(nt-d8[4])/3,(nt-d24[4])/5])
    context=[t/max(1,last),min(turns,20)/20,min(turns,50)/50,min(turns,100)/100]
    return np.asarray([per[0]+per[1]+context,per[1]+per[0]+context],dtype=np.float32)


def _fc_predict(board,last,history):
    t=board["step"]
    if t>=last+1:
        return np.zeros(2,dtype=np.float32)
    st=board.get("_st")
    meta=st.trees_meta if st is not None else board.get("trees",{})
    x=_fc_inputs(board["players"],meta,t,last,history)
    mean,scale,ridge,w1,b1,w2,b2,w3,b3=_FC_WEIGHTS
    phase=x[0,-4];late=1.-x[0,-2]
    x=np.clip((x-mean)*scale,-6.,6.)
    d=x[0]-x[1]
    base=np.concatenate((d,d*phase,d*late))@ridge
    h=np.maximum(x@w1+b1,0)
    h=np.maximum(h@w2+b2,0)
    out=h@w3+b3
    return (base+out[0]-out[1])*250.


def _fc_heads(board,side,last,history,legacy):
    """Return actual incremental cash predictions in the viewer's orientation."""
    t=board["step"]
    if t>=last+1:
        return np.zeros(2,dtype=np.float32)
    sign=1-2*side
    cash=(board["players"][0][0]-board["players"][1][0])*sign
    public=_fc_predict(board,last,history)*sign
    land=legacy*3000-cash
    basis=np.asarray([1.,t/max(1,last+1),max(0.,1.-(last+1-t)/60.)])
    inputs=np.asarray([public[0],public[1],land])
    fitted=(basis[:,None]*inputs[None,:]).ravel()@_FC_BLEND
    old=land*basis@_FC_TERR
    return fitted*_FC_ALPHA+(1.-_FC_ALPHA)*old


def _fc_leaf(board,side,last,legacy):
    heads=_fc_heads(board,side,last,_FC_HISTORY,legacy)
    cash=board["players"][side][0]-board["players"][1-side][0]
    short=(cash+heads[0]*_FC_USE20+heads[1]*(1.-_FC_USE20))/3000.
    return legacy+_FC_MIX*(short-legacy)

def _fc_public_short(board,side,last):
    """Fast standalone predictor, without the territory evaluator."""
    margin=board["players"][0][0]-board["players"][1][0]
    return (margin+_fc_predict(board,last,_FC_HISTORY)[0])*(1-2*side)/3000.
