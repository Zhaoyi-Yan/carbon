"""Test a local planter-stay reply against independent replay roots."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

from audit_scores import (MATCHES, REVISIONS, ROOT, SALT, LAND, archive_state, configure_engine, decision,
                 instrument, model, observe, one_step, read_replay, replay_prefix, reset, restore, write)
from select_roots import order_key
from starvation_replay import distance


def planters(team):
    return [(u,p) for u,(p,_,kind) in team[2].items() if kind=='PLANTER']


def freeze(cache_dirs):
    prior=ROOT/'analysis'
    excluded={}
    for path in ('tree-task','tree-payback','team-branches'):
        protocol=json.loads((prior/path/'protocol.json').read_text())
        for r in protocol.get('roots',protocol.get('cases',[])):
            excluded.setdefault(r['match'],[]).append(r['turn'])
    maps={side:[] for side in (0,1)}
    for cache in cache_dirs:
        for match,side,opponent in json.loads((cache/'games.json').read_text()):
            if match in MATCHES:continue
            path=cache/(match+'-1.html.gz')
            e=read_replay(path)
            cfg,frames=e['configuration'],e['steps']
            root=frames[0][0]['observation']
            layout=order_key(root['full_carbon'],[sorted(p[1].values()) for p in root['players']])
            viable=[]
            for t in range(25,150):
                if any(abs(t-d)<=8 for d in excluded.get(match,())):continue
                q=frames[t][0]['observation']['players']
                if any(distance(p,z)==1 for _,p in planters(q[side]) for _,z in planters(q[1-side])):
                    viable.append(t)
            if viable:
                t=min(viable,key=lambda t:order_key('reply-holdout',match,t))
                case=dict(match=match,turn=t,seat=side,opponent=opponent,seed=cfg['randomSeed'],
                          layout=layout,replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                maps[side].append((case,path))
    selected=[]
    for seat,cases in maps.items():
        cases.sort(key=lambda entry:order_key('reply-map',entry[0]['match']))
        assert len(cases)>=3,seat
        selected.extend(cases[:3])
    assert len({c['layout'] for c,_ in selected})==len(selected)==6
    return selected


def alternate(obs, candidates, sampled):
    side=obs['player']
    own,enemy=obs['players'][side],obs['players'][1-side]
    own_planters=planters(own)
    choices=[]
    for uid,cell in planters(enemy):
        if uid not in sampled:
            continue
        occupied={LAND[p][sampled.get(other,'STAY')] for other,(p,_,_) in enemy[2].items() if other!=uid}
        if cell in occupied:
            continue
        affected=[any(LAND[p][acts.get(unit,'STAY')]==cell for unit,p in own_planters)
                  for acts in candidates]
        if 0<sum(affected)<len(affected):
            choices.append((sum(affected),uid))
    if not choices:
        return None
    _,uid=max(choices)
    return {k:v for k,v in sampled.items() if k!=uid}


def losses(event, obs):
    # Observe the identity of both teams' dead planters, including trades.
    return tuple(tuple(uid for uid in v['died'] if uid in {k for k,_ in planters(team)})
                 for v,team in zip(event['units'],obs['players'],strict=True))


def evaluate(case,path,source):
    game=read_replay(path)
    cfg=game['configuration']
    own,t=case['seat'],case['turn']
    a=model(source,'held_reply_side0',True)
    b=model(source,'held_reply_side1',True)
    mods=[a,b]
    state,mem,visible=replay_prefix(game,mods,t)
    snapshot=archive_state(state)
    obs=observe(state,own,t)
    p=mods[own]
    reset(mods,mem)
    p._AUDIT=dict(proposed=[],stages=[])
    decision(p,state,own,t,visible,SALT)
    log=p._AUDIT
    if not log.get('final_scores'):
        return dict(case=case,available=False,reason='no_finalists')
    roots=log['admitted']
    native=log['replies'][0]
    supportive=alternate(obs,roots,native)
    actual=game['steps'][t+1][1-own]['action'] or {}
    alternatives=dict(sampled=native,historical=actual)
    if supportive is not None:
        alternatives['local_stay']=supportive
    outcomes={}
    rankings={}
    for label,reply in alternatives.items():
        events=[]
        for act in roots:
            both=[None,None]
            both[own],both[1-own]=act,reply
            events.append(losses(one_step(snapshot,cfg,both),obs))
        outcomes[label]=events
        if label=='sampled' or reply==native:
            trace=log
        else:
            state,_=restore(snapshot,cfg)
            reset([p],[mem[own]])
            p._AUDIT=dict(proposed=[],stages=[],reply_override=reply)
            decision(p,state,own,t,visible,SALT)
            trace=p._AUDIT
            assert trace['admitted']==roots and trace['priors']==log['priors']
        rankings[label]=dict(selected=trace['selected'],scores=trace['final_scores'])
    return dict(case=case,available=supportive is not None,actions=roots,replies=alternatives,
                outcomes=outcomes,rankings=rankings)


def summarize(rows):
    stats=Counter(roots=len(rows))
    for r in rows:
        if 'outcomes' not in r:continue
        y=r['outcomes']['historical']
        native=r['outcomes']['sampled']
        stats['evaluated_actions']+=len(y)
        alt=r['outcomes'].get('local_stay')
        stats['alternate_available']+=alt is not None
        if alt is not None:
            ranking=r['rankings']
            a,b,c=[ranking[k]['selected'] for k in ('sampled','historical','local_stay')]
            stats['choice_changed']+=a!=c
            stats['match_logged_reply_score_choice']+=b==c and a!=b
        for i,actual in enumerate(y):
            if not any(actual):
                stats['extra_predicted_collision_on_logged_no_death']+=bool(
                    alt is not None and any(alt[i]) and not any(native[i]))
                continue
            stats['recorded_death_actions']+=1
            if native[i]==actual:
                stats['native_match']+=1
            elif alt is not None and alt[i]==actual:
                stats['additional_match']+=1
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old','recent','out'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    os.environ['CARBON_DEBUG']='1'
    configure_engine()
    a.out.mkdir(parents=True,exist_ok=False)
    tag,expected=REVISIONS['yan-v8']
    source=subprocess.check_output(['git','show',tag+':submission.py'],cwd=ROOT)
    assert hashlib.sha256(source).hexdigest()==expected
    traced=a.out/'traced.py'
    traced.write_text(instrument(source.decode(),expected))
    roots=freeze((a.old,a.recent))
    write(a.out/'protocol.json',dict(roots=[c for c,_ in roots],frozen=[tag,expected],
                                    budget='one reply per scoring pass, holdout, salt 8141'))
    rows=[evaluate(c,path,traced) for c,path in roots]
    with gzip.open(a.out/'trace.json.gz','wt') as f:
        json.dump(rows,f,default=lambda x:x.item())
    write(a.out/'summary.json',summarize(rows))
