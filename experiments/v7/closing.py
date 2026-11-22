"""Settle the final simultaneous move against multiple legal opponent replies."""
import inspect
import numpy as np


def _v7_settle(obs,cfg,estimate,base,enemy_logits,t0=None):
    if obs['step']!=LAST_TURN or not _RX:return base
    if t0 is not None and _time.perf_counter()-t0>.78:return base
    side=obs['player'];foe=1-side;players=obs['players'];own=players[side];rival=players[foe]
    root=dict(obs,carbon=[float(c) for c in estimate])
    # No unit survival premium remains after this turn. Different carriers may
    # deliver on the same home square, even if the referee removes them on impact.
    def cashout(pl,act):
        act=dict(act);depots=set(pl[1].values())
        for bid in pl[1]:act.pop(bid,None)
        for uid,(p,c,k) in pl[2].items():
            if k=='COLLECTOR' and c>1:
                path=next((a for a,q in enumerate(_V7_L[p]) if q in depots),None)
                if path is not None:
                    if path:act[uid]=WACT[path]
                    else:act.pop(uid,None)
        return act
    seed=17011+side
    w,b=enemy_logits
    replies=[]
    for idx,temp in enumerate((1.,.65,1.35)):
        act=_sample(players,foe,w/temp,b/temp,np.random.RandomState(seed+idx*17),obs['step'])
        if idx:act=cashout(rival,act)
        if act not in replies:replies.append(act)
    scores={}
    def worth(a):
        key=tuple(sorted(a.items()))
        if key not in scores:
            values=[]
            for reply in replies:
                board=_turn(root,cfg,[reply,a] if side else [a,reply])
                cash=board['players'];delta=cash[side][0]-cash[foe][0]
                sign=board.get('_terminal_sign')
                if sign is not None and sign!=0:delta=(1-2*side)*sign*18000.
                values.append(delta+(75 if delta>0 else -75 if delta<0 else 0))
            scores[key]=.85*sum(values)/len(values)+.15*min(values)
        return scores[key]
    best=base;score=worth(best)
    other=cashout(own,best);raw=worth(other)
    if raw>score:best,score=other,raw
    roster=sorted(own[2],key=lambda u:-own[2][u][1])
    for uid in roster:
        chosen=best
        for a in range(5):
            if t0 is not None and _time.perf_counter()-t0>.84:return chosen
            test=dict(best)
            if a:test[uid]=WACT[a]
            else:test.pop(uid,None)
            v=worth(test)
            if v>score+.008:chosen,score=test,v
        best=chosen
    return best


def build(code):
    marker='def _act(obs, configuration=None, t0=None):'
    check="        st['liquidity_action']=dict(act)"
    assert code.count(marker)==code.count(check)==1
    code=code.replace(marker,inspect.getsource(_v7_settle)+'\n\n'+marker)
    return code.replace(check,"        act=_v7_settle(obs,configuration,est,act,st['v5_history']['opponent'],t0)\n"+check)
