"""Verify a shared-history handoff against each separate parent on legal, sequential observations."""
import argparse
import copy
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
from branch_replay import ROOT,HERE,OUT
from benchmark import configure_engine,load_strategy,validate_actions
from receipt_check import real_target


def disable_search(*args):
    return args[3]


def cell(function,name):
    return function.__closure__[function.__code__.co_freevars.index(name)]


def run(candidate,replays):
    os.environ['CARBON_DEBUG']='1'
    start=time.perf_counter()
    dual=load_strategy('handoff_dual',candidate)
    init=time.perf_counter()-start
    nested_agent=cell(dual._dual_reserve,'agent').cell_contents
    nested_act=cell(nested_agent,'_act').cell_contents
    for name in ('_search','_reply_choice'):
        cell(nested_act,name).cell_contents=disable_search
    dual._search=dual._reply_choice=disable_search
    classic=load_strategy('handoff_v6',ROOT/'yan-v6.py')
    pursuit=load_strategy('handoff_challenger',HERE/'checkpoints/harbormaster.py')
    for m in (classic,pursuit):
        m._search=m._reply_choice=disable_search
    counts=Counter();rows=[];runs=[]
    for path in replays:
        game=real_target(path);frames=game['steps']
        cfg={k:v for k,v in game['configuration'].items() if k!='randomSeed'}
        for s in (0,1):
            dual._ST={};dual._FC_HISTORY={};prior=False;switch=[];active=Counter()
            for t,frame in enumerate(frames[:-1]):
                obs=dict(frame[0]['observation']);obs.update(frame[s]['observation']);obs.pop('full_carbon',None)
                team=dual._v7_choose_team(obs,dual._ST);parent=classic if team=='reserve' else pursuit
                before=dual._ST.get('dual_team')
                check=t%13==0 or (before is not None and before!=team)
                if check:
                    parent._ST=copy.deepcopy(dual._ST);parent._FC_HISTORY=copy.deepcopy(dual._FC_HISTORY)
                actual=dual.agent(obs,cfg);validate_actions(actual,obs['players'][s]);assert dual._ST['dual_team']==team
                if before!=team:switch.append([t,team])
                active[team]+=1
                if check:
                    expected=parent.agent(obs,cfg);units=set(obs['players'][s][2])|set(obs['players'][s][1])
                    changes=[uid for uid in units if actual.get(uid)!=expected.get(uid)]
                    counts[team+'_states']+=1;counts[team+'_actions']+=len(units);counts[team+'_different_actions']+=len(changes)
                    if changes:
                        rows.append(dict(game=path.name,seat=s,turn=t,team=team,different=changes))
                recorded=frames[t+1][s]['action'] or {}
                dual._ST['tr'].set_actions(obs['players'],recorded);dual._ST['liquidity_action']=dict(recorded)
                if 'v7_next_depot' in dual._ST and dual._ST.get('v7_next_depot'):
                    uid=dual._ST['v7_next_depot'][1]
                    if actual.get(uid)!=recorded.get(uid):dual._ST.pop('v7_next_depot',None)
                if 'corridor_track' in dual._ST:
                    dual._ST['corridor_track']=[item for item in dual._ST['corridor_track'] if actual.get(item[1])==recorded.get(item[1])]
            runs.append(dict(game=path.name,seat=s,counts=dict(active),switches=switch))
    assert counts['reserve_states']>10 and counts['pursuit_states']>10
    return dict(candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),initialization_seconds=round(init,3),
                test='Both game seats; official moves teacher-forced between comparisons. At switches and every 13 turns independently sample each policy root from identical trackers and RNG. Search bypassed for parent action parity.',counts=dict(counts),different=rows,runs=runs)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('out',type=Path);p.add_argument('replay',nargs='+',type=Path)
    a=p.parse_args();configure_engine()
    data=run(a.candidate,a.replay);a.out.write_text(json.dumps(data,indent=2)+'\n');print(data['initialization_seconds'],data['counts'],flush=True)
