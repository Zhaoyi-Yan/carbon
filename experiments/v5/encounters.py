"""Adjust move priors using the predicted direction of a specific nearby opponent."""
def _v5_encounters(players,side,step,own,enemy):
    my_workers,my_bases=own
    opponent=1-side
    us=players[side]
    rival=players[opponent]
    if not us[2] or not rival[2]:return own
    left=LAST_TURN+1-step
    mine=set(us[1].values());theirs=set(rival[1].values())
    near=_v5_home(tuple(mine))
    others=_v5_home(tuple(theirs))
    at={}
    for uid,(start,cargo,kind) in rival[2].items():
        q=enemy[0][start]/.9
        weights=np.exp(q-q.max())
        weights=.90*weights/weights.sum()+.10/5.
        for p,prob in zip(_ROLLOUT_LANDING[start],weights):
            at.setdefault(p,[]).append((cargo,kind,float(prob)))
    margin=(us[0]-rival[0])/1000.
    caution=1.+.26*np.tanh(margin)*max(0.,1.-left/85.)
    new=my_workers.copy()
    for uid,(start,cargo,kind) in us[2].items():
        changes=np.zeros(5)
        for a,p in enumerate(_ROLLOUT_LANDING[start]):
            probability=at.get(p,())
            if not probability:continue
            own_goods=cargo*max(0.,1.-.045*near[p])*(near[p]<=left)
            worker=22.+34.*min(1.,left/55.)
            for enemy_load,role,prob in probability:
                enemy_goods=enemy_load*max(0.,1.-.045*others[p])*(others[p]<=left)
                if kind=='PLANTER':
                    worth=-worker if role=='COLLECTOR' else 0.
                elif role=='PLANTER':
                    worth=.72*worker
                elif cargo<enemy_load:
                    worth=worker+enemy_goods+(enemy_load if p not in theirs else -enemy_load)*.72
                elif cargo>enemy_load:
                    worth=-(worker+own_goods+(cargo if p not in mine else -cargo)*.72)*caution
                else:
                    worth=.40*(enemy_goods-own_goods)
                # The policy already encodes reachable enemies. Price the extra risk
                # of this opponent's likely destination relative to a neutral move.
                changes[a]+=(prob-.18)*np.clip(worth/105.,-4.,4.)
        new[start]+=.95*np.clip(changes-changes.mean(),-2.,2.)
    return new,my_bases
