"""One root-search alternative for a collector's stay/return decision.

Only the occupied cell's carbon is used. The referee reveals it to that collector.
Model output classes: stay to collect, get closer to an own home, other movement.
"""
import numpy as np

_HV_PRESERVE_BANK = False
_HV_COMMANDS = (None, "UP", "RIGHT", "DOWN", "LEFT")
_HV_MOVES = tuple(tuple(((p//15+dy)%15)*15+(p%15+dx)%15 for dy, dx in ((0,0),(-1,0),(0,1),(1,0),(0,-1))) for p in range(225))
_HV_DIST = tuple(tuple(min(abs(p//15-q//15),15-abs(p//15-q//15))+min(abs(p%15-q%15),15-abs(p%15-q%15)) for q in range(225)) for p in range(225))


def _hv_rows(obs, last):
    players, side, t = obs["players"], obs["player"], obs["step"]
    own, foe = players[side], players[1-side]
    homes, enemy_bases = list(own[1].values()), set(foe[1].values())
    if not homes or not enemy_bases:
        return []
    tree_cells = {u[0] for p in players for u in p[3].values()}
    enemy_tree = [u[0] for u in foe[3].values()]
    carriers = [(k,p,v) for k,(p,v,role) in own[2].items() if role == "COLLECTOR"]
    enemies = [(p,v) for p,v,role in foe[2].values() if role == "COLLECTOR"]
    wallet = own[0]
    rows = []
    for uid, pos, cargo in carriers:
        carbon = float(obs["carbon"][pos])
        if pos in tree_cells or pos in homes or pos in enemy_bases or carbon < 3:
            continue
        dist = _HV_DIST[pos]
        home = min(dist[h] for h in homes)
        left = last - t + 1
        danger = min((dist[q] for q,v in enemies if v <= cargo), default=15)
        close_trees = sum(min(abs(pos//15-q//15),15-abs(pos//15-q//15))<=1
                          and min(abs(pos%15-q%15),15-abs(pos%15-q%15))<=1 for q in enemy_tree)
        ally = min((dist[q] for k,q,_ in carriers if k!=uid),default=15)
        rival = min((dist[q] for q,_ in enemies),default=15)
        expected = .30*carbon
        x = (min(carbon,100)/100, min(cargo,400)/200, min(cargo,100)/100, min(home,14)/10,
             min(max(left-home,0),25)/25, min(left,60)/60, t/299, min(danger,7)/7,
             min(rival,7)/7, min(ally,7)/7, min(close_trees,4)/4, len(carriers)/10,
             len(enemies)/10, min(dist[q] for q in enemy_bases)/10,
             min(wallet,300)/300, min(wallet,60)/60,
             max(-1,min(1,(wallet-foe[0])/1200)), expected/(expected+cargo+1))
        rows.append((uid,pos,cargo,home,x))
    return rows


def _hv_predict(rows):
    x = np.asarray([r[-1] for r in rows], dtype=np.float32)
    h = np.maximum(x @ _HV_MODEL[0] + _HV_MODEL[1], 0)
    logits = h @ _HV_MODEL[2] + _HV_MODEL[3]
    exp = np.exp(logits - logits.max(axis=1,keepdims=True))
    return exp/exp.sum(axis=1,keepdims=True)


def _hv_proposal(obs, last, root, lw, locked=()):
    rows = _hv_rows(obs, last)
    if not rows:
        return None
    side = obs["player"]
    own, foe = obs["players"][side], obs["players"][1-side]
    homes = set(own[1].values())
    landing = {k:_HV_MOVES[p][_HV_COMMANDS.index(root.get(k))] for k,(p,_,_) in own[2].items()}
    enemies = [(p,v) for p,v,role in foe[2].values() if role=="COLLECTOR"]
    enemy_base = set(foe[1].values())
    best = None
    for row, prob in zip(rows, _hv_predict(rows)):
        uid,pos,cargo,home,_ = row
        if uid in locked:
            continue
        label = int(prob.argmax())
        if label == 2 or prob[label] < _HV_THRESH[label]:
            continue
        base_move = _HV_COMMANDS.index(root.get(uid))
        end = _HV_MOVES[pos][base_move]
        old = 0 if base_move==0 else (1 if min(_HV_DIST[end][h] for h in homes)<home else 2)
        gain = float(prob[label]-prob[old])
        if old==label or gain<.28 or (label==1 and cargo<20) or (label==0 and last-obs["step"]+1<=home+2):
            continue
        reserved = {q for k,q in landing.items() if k!=uid}
        reserved.update(p for k,p in own[1].items() if k in root)
        choices = [0] if label==0 else sorted(range(1,5), key=lambda a:-lw[pos][a])
        for action in choices:
            dest = _HV_MOVES[pos][action]
            if dest in reserved or dest in enemy_base or (label==1 and min(_HV_DIST[dest][h] for h in homes)>=home):
                continue
            if any(_HV_DIST[dest][q]<=1 and v<=cargo for q,v in enemies):
                continue
            if best is None or gain>best[0]:
                best = (gain,uid,action)
            break
    if best is None:
        return None
    _,uid,action = best
    pos,cargo,_ = own[2][uid]
    old_action = _HV_COMMANDS.index(root.get(uid))
    if (_HV_PRESERVE_BANK and cargo>0 and _HV_MOVES[pos][old_action] in homes
            and _HV_MOVES[pos][action] not in homes):
        return None
    if _HV_GAPS is not None:
        cap = _HV_GAPS[0 if action==0 else 1]
        if cap is None or float(lw[pos][old_action]-lw[pos][action])>cap:
            return None
    proposal = dict(root)
    if action:
        proposal[uid] = _HV_COMMANDS[action]
    else:
        proposal.pop(uid,None)
    return proposal
