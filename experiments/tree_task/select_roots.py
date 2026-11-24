"""Preselect one affordable new-tree decision per public replay."""
import hashlib
import json
from starvation_replay import LAND, distance, read_replay
from plan import planting_cost


def order_key(*parts):
    return hashlib.sha256('/'.join(map(str,('tree-task-v1',*parts))).encode()).hexdigest()


def options(env, side, match):
    cfg,frames = env['configuration'],env['steps']
    categories={'invested':[],'left':[]}
    for turn in range(65,245):
        players=frames[turn][0]['observation']['players']
        own,enemy=players[side],players[1-side]
        quote=planting_cost(players,side,cfg)
        # Keep enough starting cash for the tree and both possible hires.
        if own[0]<quote+cfg['recCollectorCost']+cfg['recPlanterCost']:
            continue
        blocked={p for pl in players for p in pl[1].values()}
        blocked.update(p for pl in players for p,_ in pl[3].values())
        local_planters=[p for p,_,k in enemy[2].values() if k=='PLANTER']
        hostile={q for p,_,k in enemy[2].values() if k=='COLLECTOR' for q in LAND[p].values()}
        logged=frames[turn+1][side]['action'] or {}
        for uid,(p,_,role) in own[2].items():
            if role!='PLANTER' or p in blocked or p in hostile:
                continue
            if not any(distance(p,q)<=4 for q in local_planters):
                continue
            label='left' if logged.get(uid) else 'invested'
            categories[label].append(dict(turn=turn,unit=uid,cell=p,cost=quote,
                                          cash=own[0],recorded_command=logged.get(uid,'STAY')))
    for category,rows in categories.items():
        rows.sort(key=lambda row: order_key(match,category,row['turn'],row['unit']))
    return categories


def choose(data):
    trials=[]
    layouts=set()
    for match,seat,opponent in json.loads((data/'games.json').read_text()):
        path=data/(match+'-1.html.gz')
        env=read_replay(path)
        initial=env['steps'][0][0]['observation']
        layout=order_key(initial['full_carbon'],[sorted(p[1].values()) for p in initial['players']])
        assert layout not in layouts
        layouts.add(layout)
        trials.append((dict(match=match,seat=seat,opponent=opponent,layout=layout,
                            seed=env['configuration']['randomSeed'],
                            replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest()), options(env,seat,match)))
    cases=[]
    for seat,count in ((0,2),(1,2)):
        eligible=sorted((p for p in trials if p[0]['seat']==seat),
                        key=lambda p:order_key(p[0]['match'],'assignment'))
        # Both seats contribute, with two historically invested roots per seat.
        chosen={meta['match'] for meta,opts in [p for p in eligible if p[1]['invested']][:count]}
        assert len(chosen)==count
        for meta,opts in eligible:
            label='invested' if meta['match'] in chosen else 'left'
            if not opts[label]:
                raise ValueError(f"No {label} opportunity in {meta['match']}")
            cases.append(dict(**meta,category=label,**opts[label][0]))
    assert len(cases)==8
    return cases
