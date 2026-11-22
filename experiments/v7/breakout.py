"""Finance one new hire under blockade with a single courier and a route to either depot."""
import inspect
from pathlib import Path

import numpy as np


def _v7_watch_cash(obs, previous, move, history):
    """Count turns since a carried load entered our depot using executed own commands."""
    side=obs['player'];step=obs['step'];own=obs['players'][side]
    prior=history.get('liquidity_step')
    if previous is None or not step or prior!=step-1:
        history['last_delivery']=step
    elif move:
        homes=set(own[1].values())
        for uid,(pos,cargo,role) in previous[side][2].items():
            if role=='COLLECTOR' and cargo>=15 and _V7_L[pos,WIDX.get(move.get(uid),0)] in homes:
                history['last_delivery']=step
                break
    history['liquidity_step']=step
    return step-history['last_delivery']


def _v7_cash_route(obs, config, base, logits, foe_logits, drought):
    side=obs['player'];us,them=obs['players'][side],obs['players'][1-side]
    price=config.get('recCollectorCost',30)
    homes=list(us[1].values());rival_homes=list(them[1].values())
    if not homes or obs['step']<20 or us[0]>=2.2*price or len(us[2])>8:return base
    desperate=us[0]<price and (not us[3] or drought>=22)
    if not desperate and (drought<8 or us[0]+2*price>them[0] and len(us[2])+2>=len(them[2])):return base
    fleet=[(u,p,c) for u,(p,c,k) in us[2].items() if k=='COLLECTOR' and c>=price]
    if not fleet:return base
    enemy=[(p,c) for p,c,k in them[2].values() if k=='COLLECTOR']
    near=_V7_D[homes].min(0)
    # Generate an alternate full route; some first moves can increase depot distance.
    # Losing a collision on our depot still credits the recovered cargo to us.
    hires={p for u,p in us[1].items() if u in base}
    original={u:WIDX.get(base.get(u),0) for u in us[2]}
    ends={u:int(_V7_L[p,original[u]]) for u,(p,c,k) in us[2].items()}
    taxes=np.zeros(225)
    for p,age in them[3].values():
        if age<48:
            dy,dx=divmod(p,15)
            for i in (-1,0,1):
                for j in (-1,0,1):taxes[((dy+i)%15)*15+(dx+j)%15]+=.055
    best=None
    for uid,pos,load in fleet:
        if not near[pos] or near[pos]>min(11,LAST_TURN+1-obs['step']):continue
        harmful=[q for q,v in enemy if v<=load]
        danger=np.zeros(225);future=np.zeros(225)
        if harmful:
            dist=_V7_D[harmful]
            # Persistent depot guards make one approach more expensive than the other.
            crowd=(dist<=1).sum(0);neighbors=(dist==2).sum(0)
            future=np.minimum(.86,.22*crowd+.075*neighbors+.23*(dist==0).any(0))
            safe=np.ones(225)
            for q in harmful:
                prob=_probs(foe_logits[q],.78)
                safe[_V7_L[q]]*=1.-(.9*prob+.02)
            danger=1.-safe
        future[homes]=0.;danger[homes]=0.
        future[rival_homes]=1.;danger[rival_homes]=1.
        # Discount arrival for delay, interception and carbon drained by enemy trees.
        survive=(1.-future*.80)*.924*np.exp(-taxes)
        val=np.zeros(225);val[homes]=1.
        for _ in range(min(14,LAST_TURN+1-obs['step'])-1):
            val=(val*survive)[_V7_L[:,1:]].max(1);val[homes]=1.
        nodes=_V7_L[pos]
        scores=val[nodes]*(1.-danger[nodes])*.94*np.exp(-taxes[nodes])
        scores[np.isin(nodes,homes)]=1.
        held={v for u,v in ends.items() if u!=uid and v not in homes}
        scores[np.isin(nodes,list(held))]=0.
        scores[0]*=.76
        # Close alternatives use the trained movement prior. The opportunity to restore
        # a minimum recruit reserve saturates, so riskier larger loads need not go first.
        priors=_probs(logits[pos],.85);rank=scores+.026*priors
        a=int(rank.argmax());q=int(nodes[a])
        if a==original[uid] or scores[a]<(.12 if desperate else .22):continue
        if scores[a]-scores[original[uid]]<(.018 if desperate else .060):continue
        value=(min(load,2.8*price)+.14*load)*scores[a]
        if best is None or value>best[0]:best=(value,uid,a,q)
    if best is None:return base
    _,uid,a,q=best
    action=dict(base)
    if a:action[uid]=WACT[a]
    else:action.pop(uid,None)
    # Preserve an inbound load when the incumbent also recruited on that depot.
    if q in hires:
        for bid,p in us[1].items():
            if p==q:action.pop(bid,None)
    return action


def build(code):
    marker='def _act(obs, configuration=None, t0=None):'
    assert code.count(marker)==1
    functions='\n\n'.join(inspect.getsource(f) for f in (_v7_watch_cash,_v7_cash_route))
    code=code.replace(marker,functions+'\n\n'+marker)
    before="    act = _r6_opening(obs, configuration, lb, act)"
    assert code.count(before)==1
    return code.replace(before,'''    if configuration is not None and _SEARCH_OK:
        prior_move=st.get('liquidity_action',{})
        drought=_v7_watch_cash(obs,st['prev'],prior_move,st)
        enemy=st['v5_history']['opponent'][0]
        act=_v7_cash_route(obs,configuration,act,lw,enemy,drought)
        st['liquidity_action']=dict(act)
'''+before)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    with a.output.open('x') as f:f.write(build(a.parent.read_text()))
