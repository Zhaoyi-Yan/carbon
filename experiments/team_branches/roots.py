"""Preselect tree and late-depot roots without using results after the decision."""
import hashlib
import json
from pathlib import Path

from select_roots import options, order_key
from starvation_replay import LAND, distance, read_replay


def metadata(path,env,match,side,opponent):
    ob=env['steps'][0][0]['observation']
    return dict(match=match,seat=side,opponent=opponent,seed=env['configuration']['randomSeed'],
                layout=order_key(ob['full_carbon'],[sorted(p[1].values()) for p in ob['players']]),
                path=str(path),
                replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def cargo(team):
    return sum(c for _,c,k in team[2].values() if k=='COLLECTOR')


def late_positions(env,side,match):
    out=[]
    for t in range(239,267):
        teams=env['steps'][t][0]['observation']['players']
        own,foe=teams[side],teams[1-side]
        gap=own[0]-foe[0]
        freight=cargo(foe)-cargo(own)
        reachable=sum(c for p,c,k in own[2].values() if k=='COLLECTOR' and
                      min(distance(p,q) for q in own[1].values())<=9)
        if 0<=gap<=650 and freight>=100 and reachable>=90:
            out.append(dict(turn=t,cash=round(gap,2),freight_gap=round(freight,2),
                            our_cargo=round(cargo(own),2),enemy_cargo=round(cargo(foe),2)))
    return sorted(out,key=lambda c:order_key(match,'team-late-v1',c['turn']))


def load_cases(old,recent):
    archive={}
    for root in (old,recent):
        for match,seat,opp in json.loads((root/'games.json').read_text()):
            path=root/(match+'-1.html.gz')
            env=read_replay(path)
            archive[match]=(metadata(path,env,match,seat,opp),env)
    prior=Path(__file__).resolve().parents[2]/'analysis'
    excluded={}
    for sub in ('tree-task','tree-payback'):
        protocol=json.loads((prior/sub/'protocol.json').read_text())
        for case in protocol.get('roots',protocol.get('cases',[])):
            excluded.setdefault(case['match'],[]).append(case['turn'])

    trees=[]
    used=set()
    for side in (0,1):
        for label in ('invested','left'):
            eligible=sorted((m for m,(meta,_) in archive.items() if meta['seat']==side and m not in used),
                            key=lambda m:order_key(m,'team-tree-v1',label))
            count=0
            for match in eligible:
                meta,env=archive[match]
                choices=[p for p in options(env,side,match)[label]
                         if all(abs(p['turn']-t)>8 for t in excluded.get(match,[]))]
                if choices:
                    trees.append(dict(**meta,**choices[0],cohort='trees',prior_choice=label))
                    used.add(match)
                    count+=1
                if count==2:
                    break
            assert count==2
    late=[]
    seed_root='4665cd15a5516650cd80eb5cda90cf43'
    for match,(meta,env) in archive.items():
        choices=late_positions(env,meta['seat'],match)
        if match==seed_root:
            root=next(c for c in choices if c['turn']==250)
            late.insert(0,dict(**meta,**root,cohort='late',reference='Word Master7sE turn 250'))
        elif choices:
            late.append(dict(**meta,**choices[0],cohort='late',reference='cash lead and cargo deficit'))
    late=[late[0]]+sorted(late[1:],key=lambda c:('Master7sE' not in c['opponent'],order_key(c['match'],'team-late-v1')))
    assert len(late)>=4
    return trees+late[:4]
