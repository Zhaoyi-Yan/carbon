"""Compare team actions at common roots, then trace where v8 omitted or rejected them."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tree_task'))
from run_task import (REVISIONS, archive_state, board_digest, build, configure_engine,
                      decision, digest, history, metrics, model, observe, read_replay,
                      replay_prefix, reset, restore, step, update_memory, validate_actions, write)
from plan import arrivals
from starvation_replay import LAND, cash_flows, distance
from roots import load_cases

SALT=8141


def traced_source(source):
    text=build(source,expected_sha256=REVISIONS['yan-v8'][1])
    anchor="        _AUDIT.update(final_scores={k:mean[k] for k in alive}, selected=best)"
    assert text.count(anchor)==1
    return text.replace(anchor,anchor+'''
        if 'force' in _AUDIT:
            assert _AUDIT['force'] in alive
            best=_AUDIT['force']''')


def tree_actions(obs, original, stock, worker):
    side=obs['player']
    own,enemy=obs['players'][side],obs['players'][1-side]
    pos=own[2][worker][0]
    occupied=arrivals(own,original,worker)
    threats={q for p,_,k in enemy[2].values() if k=='COLLECTOR' for q in LAND[p].values()}
    plant=dict(original)
    plant.pop(worker,None)
    feasible=pos not in occupied
    allowed=[(m,p) for m,p in LAND[pos].items() if m!='STAY' and p not in occupied and
             p not in enemy[1].values()]
    if allowed:
        chosen=min(allowed,key=lambda a:(a[1] in threats,original.get(worker)!=a[0],
                                         -sum(float(stock[q]) for q in LAND[a[1]].values()),a[0]))
        defer={**original,worker:chosen[0]}
    else:
        defer=original
    return {'plant':(plant if feasible else original,feasible),
            'defer':(defer,bool(allowed))}


def transport_actions(policy, obs, cfg, original, stock):
    own=obs['players'][obs['player']]
    homes=tuple(own[1].values())
    bank=policy._v7_plan(obs,cfg,stock,original,'bank')
    # Treat a bank proposal as eligible only if a loaded collector advances toward a depot.
    def progress(act):
        return sum(weight*(min(distance(p,b) for b in homes)-
                           min(distance(LAND[p][act.get(u,'STAY')],b) for b in homes))
                   for u,(p,weight,kind) in own[2].items() if kind=='COLLECTOR' and weight>=45)
    if progress(bank)<=progress(original):
        bank=original
        alternatives=[]
        for uid,(pos,load,kind) in own[2].items():
            if kind!='COLLECTOR' or load<60:
                continue
            occupied=arrivals(own,original,uid)
            foe=obs['players'][1-obs['player']]
            threats={q for p,c,k in foe[2].values() if k=='COLLECTOR' and c<=load for q in LAND[p].values()}
            here=min(distance(pos,b) for b in homes)
            for m,p in LAND[pos].items():
                if m=='STAY' or p in occupied or (p in threats and p not in homes):
                    continue
                if min(distance(p,b) for b in homes)>=here:
                    continue
                test={**original,uid:m}
                extra=progress(test)-progress(original)
                if extra>0:
                    alternatives.append((extra,m,uid,test))
        if alternatives:
            bank=max(alternatives,key=lambda x:x[:3])[3]
    intercept=policy._v7_plan(obs,cfg,stock,original,'intercept')
    return {'bank':(bank,bank!=original),'intercept':(intercept,intercept!=original)}


def locate(trace, action, worker=None):
    proposed=[src for src,a in trace['proposed'] if a==action]
    pool=trace.get('admitted',[])
    idx=next((i for i,a in enumerate(pool) if a==action),None)
    scores=trace.get('final_scores',{})
    stages=dict(trace['stages'])
    return dict(proposed=proposed,admitted=idx is not None,scored_depth=trace.get('scored_depth',{}).get(idx),
                finalist_score=scores.get(idx),winning_score=scores.get(trace.get('selected')),
                selected=idx is not None and trace.get('selected')==idx,
                matching_unit_command=(sum(a.get(worker)==action.get(worker) for a in pool)
                                       if worker is not None else None),
                stage_hits=[stage for stage,a in trace['stages'] if a==action],
                search_action_matches=stages.get('search')==action)


def branch(mods,snapshot,cfg,visible,case,memory,root_actions):
    state,env=restore(snapshot,cfg)
    reset(mods,memory)
    start,side=case['turn'],case['seat']
    totals=[Counter(),Counter()]
    owners={uid:i for i,p in enumerate(state[0].observation.players) for uid in p[3]}
    initial=[p[0] for p in state[0].observation.players]
    marks={}
    hash_trace=hashlib.sha256()
    for t in range(start,299):
        old=state[0].observation
        actions=root_actions if t==start else [decision(p,state,i,t,visible,SALT) for i,p in enumerate(mods)]
        for i,a in enumerate(actions):
            validate_actions(a,old.players[i])
        step(state,env,actions)
        obs=state[0].observation
        flows,*_=cash_flows([{'observation':old},{}],
            [{'observation':obs,'action':actions[0]},{'action':actions[1]}],cfg,owners)
        for bucket,flow in zip(totals,flows):
            bucket.update(flow)
        hash_trace.update(bytes.fromhex(digest([actions,board_digest(state)])))
        end=t+1
        if end in (start+1,start+10,start+20,299):
            cash=[p[0] for p in obs.players]
            raw=[cash[i]-initial[i] for i in (side,1-side)]
            marks[end]=dict(team_cash_change=raw,net_cash=round(raw[0]-raw[1],2),
                            ledger=metrics(totals,side),cargo=[round(sum(c for _,c,_ in obs.players[i][2].values()),2)
                                                              for i in (side,1-side)])
        if any(p.status!='ACTIVE' for p in state):
            break
    assert end==299 and all(p.status in ('ACTIVE','DONE') for p in state)
    return dict(end=end,marks=marks,scores=[p.reward for p in state],trace=hash_trace.hexdigest())


def evaluate(spec):
    case,out,aa=spec
    configure_engine()
    path=Path(case['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==case['replay_sha256']
    game=read_replay(path)
    cfg=game['configuration']
    own,t=case['seat'],case['turn']
    enemy=1-own
    mods=[model(out/'yan-v8.py',f'team_v8_{i}',True) for i in (0,1)]
    traced=model(out/'traced.py','team_v8_traced',True)
    state,memory,visible=replay_prefix(game,mods,t)
    snap=archive_state(state)
    reset(mods,memory)
    natural=[decision(m,state,i,t,visible,SALT) for i,m in enumerate(mods)]
    after=history(mods)
    stock=mods[own]._ST['tr'].bel.estimate()
    obs=observe(state,own,t)
    edits=(tree_actions(obs,natural[own],stock,case['unit']) if case['cohort']=='trees'
           else transport_actions(mods[own],obs,visible,natural[own],stock))
    state,_=restore(snap,cfg)
    reset([traced],[memory[own]])
    traced._AUDIT=dict(proposed=[],stages=[])
    assert decision(traced,state,own,t,visible,SALT)==natural[own]
    log=traced._AUDIT
    recorded=game['steps'][t+1][enemy]['action'] or {}
    responses={'v8':natural[enemy],'historical':recorded}
    arms={'original':dict(action=natural[own],ready=True,memory=after[own],
                         audit=locate(log,natural[own],case.get('unit')))}
    for name,(act,ready) in edits.items():
        reset([mods[own]],[after[own]])
        update_memory(mods[own],obs,natural[own],act)
        arms[name]=dict(action=act,ready=ready,memory=history([mods[own]])[0],
                        audit=locate(log,act,case.get('unit')))
    finals=log.get('final_scores',{})
    other=[i for i in finals if i!=log.get('selected')]
    if other:
        index=max(other,key=finals.get)
        reset([traced],[memory[own]])
        traced._AUDIT=dict(proposed=[],stages=[],force=index)
        acted=decision(traced,state,own,t,visible,SALT)
        assert traced._AUDIT['final_scores']==finals
        arms['runner_up']=dict(action=acted,ready=True,memory=history([traced])[0],
                               raw=log['admitted'][index],audit=locate(log,log['admitted'][index]),
                               altered_after_search=acted!=log['admitted'][index])
    if aa:
        a,b=[branch(mods,snap,cfg,visible,case,after,natural) for _ in range(2)]
        assert a==b
        return dict(verified=True,seed=case['seed'],turn=t,trace=a['trace'])
    result=[]
    for variant,reply in responses.items():
        reply_obs=observe(state,enemy,t)
        reset([mods[enemy]],[after[enemy]])
        update_memory(mods[enemy],reply_obs,natural[enemy],reply)
        enemy_mem=history([mods[enemy]])[0]
        outcomes={}
        for name,arm in arms.items():
            first=natural.copy()
            first[own],first[enemy]=arm['action'],reply
            hist=after.copy()
            hist[own],hist[enemy]=arm['memory'],enemy_mem
            # An unavailable or identical synthetic edit preserves the original history.
            if name!='original' and name!='runner_up' and first[own]==natural[own]:
                played=outcomes['original']['played']
            elif variant=='historical' and recorded==natural[enemy]:
                played=result[0]['arms'][name]['played']
            else:
                played=branch(mods,snap,cfg,visible,case,hist,first)
            outcomes[name]={**{k:v for k,v in arm.items() if k!='memory'},'played':played}
        result.append(dict(case={k:v for k,v in case.items() if k!='path'},response=variant,
                           enemy_first=reply,same_reply=recorded==natural[enemy],
                           board=digest(snap),arms=outcomes))
    return result


def main(old,recent,out):
    out.mkdir(parents=True)
    tag,expected=REVISIONS['yan-v8']
    src=subprocess.check_output(['git','show',tag+':submission.py'],cwd=ROOT)
    assert hashlib.sha256(src).hexdigest()==expected
    (out/'yan-v8.py').write_bytes(src)
    (out/'traced.py').write_text(traced_source(src.decode()))
    cases=load_cases(old,recent)
    write(out/'protocol.json',dict(frozen=REVISIONS['yan-v8'],salt=SALT,checkpoints=[10,20,'terminal'],
        budget='width 4, depth at most 2, one search reply, three tail steps, fixed synthetic clock',
        roots=[{k:v for k,v in c.items() if k!='path'} for c in cases]))
    with ProcessPoolExecutor(3) as pool:
        write(out/'aa.json',pool.submit(evaluate,(cases[-1],out,True)).result())
        with gzip.open(out/'branches.jsonl.gz','xt') as f:
            for batch in pool.map(evaluate,[(c,out,False) for c in cases]):
                for r in batch:
                    f.write(json.dumps(r,default=lambda x:x.item())+'\n')


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='0':
        os.environ['PYTHONHASHSEED']='0'
        os.execv(sys.executable,[sys.executable,*sys.argv])
    os.environ['CARBON_DEBUG']='1'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
        os.environ[k]='1'
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old','recent','out'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    main(args.old.resolve(),args.recent.resolve(),args.out.resolve())
