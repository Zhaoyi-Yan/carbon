"""Two-turn emergency deposits with a certified safe intermediate square."""
import inspect
from pathlib import Path


def _v7_short_deposit(obs, config, base, drought):
    s=obs['player'];us,enemy=obs['players'][s],obs['players'][1-s]
    price=config.get('recCollectorCost',30)
    pending=_ST.pop('v7_next_depot',None)
    if pending is not None:
        turn,uid,home=pending
        if turn==obs['step'] and uid in us[2] and home in us[1].values():
            p,c,k=us[2][uid]
            if k=='COLLECTOR' and c>0 and home in _V7_L[p,1:]:
                result=dict(base);result[uid]=WACT[list(_V7_L[p]).index(home)]
                for bid,tile in us[1].items():
                    if tile==home:result.pop(bid,None)
                return result
    if obs['step']<18 or us[0]>=2*price or drought<7 or not us[1]:return base
    if us[0]>=price and len(us[2])+1>=len(enemy[2]):return base
    homes=set(us[1].values());foe_bases=set(enemy[1].values())
    cmds={u:WIDX.get(base.get(u),0) for u in us[2]}
    ends={u:int(_V7_L[p,cmds[u]]) for u,(p,c,k) in us[2].items()}
    recruits={p for u,p in us[1].items() if u in base}
    offers=[]
    for uid,(p,c,k) in us[2].items():
        if k!='COLLECTOR' or c<price or p in homes:continue
        harmful=[q for q,v,r in enemy[2].values() if r=='COLLECTOR' and v<=c]
        occupied={v for u,v in ends.items() if u!=uid}|recruits
        for a,q in enumerate(_V7_L[p,1:],1):
            if q in homes:
                offers.append((c+price,1,int(a==cmds[uid]),uid,a,int(q)))
            elif LAST_TURN-obs['step']>=1 and q not in occupied|foe_bases and any(_V7_D[q,b]==1 for b in homes):
                if all(_V7_D[q,z]>=2 for z in harmful):
                    offers.append((.80*c,0,int(a==cmds[uid]),uid,a,int(q)))
    if not offers:return base
    _,immediate,_,uid,a,q=max(offers)
    if not immediate:
        home=min(b for b in homes if _V7_D[q,b]==1)
        _ST['v7_next_depot']=(obs['step']+1,uid,home)
    result=dict(base);result[uid]=WACT[a]
    for bid,p in us[1].items():
        if q==p:result.pop(bid,None)
    return result


def build(code):
    marker='def _act(obs, configuration=None, t0=None):'
    anchor='        act=_v7_keep_float(obs,configuration,act,drought)'
    assert code.count(marker)==code.count(anchor)==1
    code=code.replace(marker,inspect.getsource(_v7_short_deposit)+'\n\n'+marker)
    return code.replace(anchor,anchor+'\n        act=_v7_short_deposit(obs,configuration,act,drought)')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    with a.output.open('x') as f:f.write(build(a.parent.read_text()))
