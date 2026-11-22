"""Local action credit from the next public board: transfers, collisions, seizes and deposits."""
import numpy as np


def outcome_weights(frames,t,side):
    before=frames[t][0]['observation']['players']
    after=frames[t+1][0]['observation']['players']
    my,enemy=before[side],before[1-side];my_next,enemy_next=after[side],after[1-side]
    bases=set(my[1].values())
    enemy_trees={p for p,age in enemy[3].values()}
    new_trees={p for k,(p,age) in my_next[3].items() if k not in my[3]}
    missing={p for uid,(p,c,role) in enemy[2].items() if uid not in enemy_next[2]}
    weight=np.ones((2,225),np.float32)
    def distance(p,q):
        dr=abs(p//15-q//15);dc=abs(p%15-q%15)
        return min(dr,15-dr)+min(dc,15-dc)
    for wid,(pos,cargo,role) in my[2].items():
        value=1.
        unit=my_next[2].get(wid)
        danger=any(distance(pos,q)<=2 and r=='COLLECTOR' for q,v,r in enemy[2].values())
        if danger:value+=.16
        if unit:
            q,load,_=unit
            if role=='COLLECTOR':
                if load>cargo+40 and any(distance(q,p)<=1 for p in missing):value+=.9
                if q in bases and cargo>=35:value+=min(.65,cargo/300.)
            elif q in new_trees:value+=.60 if q in enemy_trees else .15
        elif cargo>25 and not any(distance(pos,b)<=1 for b in bases):
            value*=max(.32,.75-cargo/650.)
        elif role=='PLANTER' and not any(distance(pos,q)<=1 for q in missing):
            value*=.65
        weight[0,pos]=value
    return weight.astype(np.float16)
