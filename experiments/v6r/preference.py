"""One-turn positional features and optional planter proposal from a revealed-preference fit."""
import argparse
import inspect
import json
from pathlib import Path

FEATURES = ['alive','one_more_tree','cash_delta','safe_nearest','enemy_collector_adjacent','foe_planter_adjacent',
            'alone_from_own_planters','on_enemy_tree','enemy_tree_value_here','on_friendly_tree',
            'plantable_yield_here','best_adjacent_yield','reach_enemy_tree','reach_young_rich_tree',
            'territory','own_base','edge_of_own_forest']
FIXED = [2.5,.55,.10,.12,-.8,-.12,.04,1.1,.5,-.2,.40,.18,.18,.15,.1,-.25,.1]


def _pref_features(origin,board,side,wid):
    old = origin['players'][side]
    mine,rival = board['players'][side],board['players'][1-side]
    w = mine[2].get(wid)
    if w is None:
        return [0.]*17
    pos=w[0]
    carbon=board['carbon']
    own_t={q:age for q,age in mine[3].values()}
    rival_t={q:age for q,age in rival[3].values()}
    bases=set(mine[1].values()) | set(rival[1].values())
    tree_cells=own_t.keys() | rival_t.keys()
    def yield_at(q):
        return sum(carbon[h] for h in NB[q] if h not in tree_cells)/300.
    def site(q):
        return yield_at(q) if q not in tree_cells and q not in bases else 0.
    ec=min((DIST[pos][q] for q,_,role in rival[2].values() if role=='COLLECTOR'),default=12)
    ep=min((DIST[pos][q] for q,_,role in rival[2].values() if role=='PLANTER'),default=12)
    pals=min((DIST[pos][q] for k,(q,_,role) in mine[2].items() if k!=wid and role=='PLANTER'),default=8)
    old_tree_count=len(old[3])
    local_captures=[(max(0,1-dist/6),max(0,1.-age/50)*yield_at(q))
                    for q,age in rival_t.items() if (dist:=DIST[pos][q])<6]
    return [1.,len(mine[3])-old_tree_count,(mine[0]-old[0])/150.,min(ec,5)/5.,float(ec<=1),
            float(ep<=1),min(pals,5)/5.,float(pos in rival_t),
            (1-rival_t[pos]/50)*yield_at(pos) if pos in rival_t else 0.,float(pos in own_t),
            site(pos),max(site(q) for q in NB[pos]),max((v[0] for v in local_captures),default=0.),
            max((d*r for d,r in local_captures),default=0.),
            (min(DIST[pos][q] for q in mine[1].values())-min(DIST[pos][q] for q in rival[1].values()))/8.,
            float(pos in bases),min((DIST[pos][q] for q in own_t),default=6)/6.]


def _pref_propose(root,cfg,side,act0,locked,reply):
    if not 12<=root['step']<=190:
        return None
    players=root['players']; mine,rival=players[side],players[1-side]
    planters=[w for w,unit in mine[2].items() if unit[2]=='PLANTER' and w not in locked and act0.get(w) in WACT[1:]]
    if not planters:
        return None
    # Spend the additional search on one disputed or immediately plantable site.
    def priority(w):
        p=mine[2][w][0]
        return min((DIST[p][q] for q,_ in rival[3].values()),default=8)
    wid=min(planters,key=priority);pos=mine[2][wid][0]
    land={move(v[0],WIDX.get(act0.get(k),0)) for k,v in mine[2].items() if k!=wid}
    land.update(p for k,p in mine[1].items() if k in act0)
    land.update(rival[1].values())
    original=WIDX.get(act0.get(wid),0);scored=[]
    for action,q in enumerate(NB[pos],1):
        if action!=original and q in land:
            continue
        ours=dict(act0)
        if action:ours[wid]=WACT[action]
        else:ours.pop(wid,None)
        joint=[ours,reply] if side==0 else [reply,ours]
        b=_turn(root,cfg,joint)
        v=sum(x*w for x,w in zip(_pref_features(root,b,side,wid),_PREF_WEIGHTS))
        scored.append((v,action,ours))
    base=next(v for v,k,_ in scored if k==original)
    best=max(scored,key=lambda r:r[0])
    return best[2] if best[1]!=original and best[0]-base>=.85 else None


def export(base,model,out):
    text=base.read_text()
    point="    opp = {}\n"
    if text.count(point)!=1:raise ValueError('Original search reply anchor unavailable')
    learned=json.loads(model.read_text())['weights']
    prelude='_PREF_WEIGHTS = '+repr(learned)+'\n\n'+inspect.getsource(_pref_features)+'\n'+inspect.getsource(_pref_propose)+'\n'
    text=text.replace('def _search(obs,',prelude+'\ndef _search(obs,',1)
    more="""    if most>=4 and step%2==0:
        opp[0] = _sample(players, op, lwo, lbo, np.random.RandomState(seed0+7), step)
        proposal = _pref_propose(root,configuration,me,act0,fixed or {},opp[0])
        if proposal is not None and tuple(sorted(proposal.items())) not in keys:
            if len(cands)>=most:
                cands[-1]=proposal
            else:
                cands.append(proposal)
"""
    text=text.replace(point,point+more)
    compile(text,str(out),'exec')
    out.write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base','model','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();export(a.base,a.model,a.output)
