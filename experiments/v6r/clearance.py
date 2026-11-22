"""Offer the best joint conflict-free worker moves under local contact and delivery values."""
import argparse
import inspect
from pathlib import Path


def _r6_clearance(obs, original, locked, logits, enemy_logits, stock):
    side = obs['player']
    own, rival = obs['players'][side], obs['players'][1-side]
    if len(own[2]) < 3:
        return None
    landing = _V5_REACH
    homes = list(own[1].values())
    enemy_homes = set(rival[1].values())
    dist = np.asarray(_v5_home(tuple(homes)))
    remaining = LAST_TURN + 1 - obs['step']
    rival_contacts = [[] for _ in range(N)]
    for pos, cargo, kind in rival[2].values():
        weight = _probs(enemy_logits[pos], .8)
        for q, probability in zip(landing[pos], weight):
            rival_contacts[q].append((cargo, kind, .95*float(probability)+.01))
    tree_tax = np.zeros(N)
    for pos, age in rival[3].values():
        if age<50:
            tree_tax[pos] += .10
            tree_tax[np.asarray([shift(pos,i,j,15) for i,j in N8])] += .10
    held = {p for uid,p in own[1].items() if uid in original}
    for uid in locked:
        if uid in own[2]:
            held.add(landing[own[2][uid][0],WIDX.get(original.get(uid),0)])
    roster = [(uid,unit) for uid,unit in own[2].items() if uid not in locked]
    if not roster:
        return None
    rows = []
    for i,(uid,(p,load,role)) in enumerate(roster):
        collector = role=='COLLECTOR'
        row = {N+i:(-1.e5,0)}
        for a,q in enumerate(landing[p]):
            if q in held:
                continue
            # Independent local risks are used to generate a joint proposal.
            # The main lookahead still scores it with the game simulator.
            contact = 0.
            for other,kind,prob in rival_contacts[q]:
                if kind=='COLLECTOR' and (not collector or other<=load):
                    contact -= prob*(50.+1.35*load)
                elif collector:
                    contact += prob*(30.+min(other,350.)*.90)
                else:
                    contact -= prob*15.
            econ = 0.
            if collector:
                if q in enemy_homes:
                    econ -= 55.+2.*load
                if load and q in homes:
                    econ += (1. if remaining<14 else .41)*load
                if q not in homes:
                    econ -= 1.65*load*tree_tax[q]
                if dist[q]>=remaining and q not in homes:
                    econ -= .75*load
                if not a and q not in enemy_homes and q not in homes and dist[q]<remaining:
                    econ += .16*stock[q]
            raw = float(logits[p,a]-logits[p].max())
            row[int(q)] = (raw + (contact+econ)/48., a)
        rows.append(row)
    out = original.copy()
    for (uid,_),row,dst in zip(roster,rows,assign_moves(rows)):
        if dst>=N:
            return None
        move_idx = row[dst][1]
        if move_idx:
            out[uid]=WACT[move_idx]
        else:
            out.pop(uid,None)
    return out if out!=original else None


def build(code):
    start = 'def _search(obs,'
    where = '    for T, cap in _S_GEN:'
    if code.count(start)!=1 or code.count(where)!=1:
        raise ValueError('expected a single root planner')
    code=code.replace(start,inspect.getsource(_r6_clearance)+'\n\n'+start)
    code=code.replace(where,"""    if most>=4 and len(cands)<most:
        market=_r6_clearance(obs,act0,fixed or {},lw0,_ST['v5_history']['opponent'][0],est)
        if market is not None:
            marker=tuple(sorted(market.items()))
            if marker not in keys:cands.append(market);keys.add(marker)
"""+where)
    compile(code,'<joint-clearance>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    with args.output.open('x') as f:f.write(build(args.parent.read_text()))
