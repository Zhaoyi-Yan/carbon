"""Restore the four frozen v8 development roots and audit their exact finalist scores."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tree_task'))
from run_task import (REVISIONS, archive_state, board_digest, configure_engine, decision,
                      digest, model, observe, read_replay, replay_prefix, reset, restore,
                      step, validate_actions, write)
SALT = 8141
from instrument_score import instrument
from starvation_replay import LAND, cash_flows

MATCHES = ('b02fe3801ed1a033338d5cce15c2ee6d','422eb40662920ca572bb10d455e900ec',
           'ef1582ed1be039f3114cc258950cb04a','cf04f35e3658f47cf17dc32ae10dc1cc')


def close(a,b):
    assert math.isclose(a,b,rel_tol=0,abs_tol=1e-10),(a,b)


def reconstruct(log,t):
    final=log['final_scores']
    tail=log['tail']
    winning=log['selected']
    runner=max((i for i in final if i!=winning),key=final.get)
    values={(i,r):value for i,r,value in log['eval_values']}
    for i, mean_score in log['before_tail'].items():
        close(mean_score,sum(values[i,r] for r in range(log['rolls']))/log['rolls'] - log['priors'][i])
    for i in final:
        slot=tail['candidates'].index(i)
        pre=log['before_tail'][i]
        close(final[i], .52*pre+.48*(tail['values'][slot]+tail['shared']))
    for row in log['leaf']:
        v=row['raw_critic']
        if row['banking_weight'] is not None:
            w=row['banking_weight']
            v=(1-w)*v+w*row['banking_value']
        if row['fc'] is not None:
            v+=row['fc_weight']*(row['fc']-v)
            v+=row['netcash_weight']*(row['netcash']-v)
        close(v,row['value'])
    best=max(final,key=final.get)
    close(log['base_score'],final[0])
    assert best==log['max_before_eps']
    assert winning==(0 if final[best]-final[0]<=log['eps'] else best)
    def path(i):
        row,=[r for r in log['leaf'] if r['candidate']==i and r['reply']==0 and r['turn']==t+2]
        return dict(raw=row['raw_critic'], fc=row['fc'], netcash=row['netcash'], leaf=row['value'],
                    prior=log['priors'][i],pre_tail=log['before_tail'][i],
                    tail=tail['values'][tail['candidates'].index(i)],final=final[i])
    a,b=path(winning),path(runner)
    return winning,runner,{k:b[k]-a[k] for k in a if a[k] is not None and b[k] is not None}


def one_step(snapshot,cfg,actions):
    state,env=restore(snapshot,cfg)
    old=state[0].observation
    for i,a in enumerate(actions):
        validate_actions(a,old.players[i])
    owners={uid:i for i,p in enumerate(old.players) for uid in p[3]}
    step(state,env,actions)
    new=state[0].observation
    ledger,receipts,losses,tree_events=cash_flows([{'observation':old},{}],
        [{'observation':new,'action':actions[0]},{'action':actions[1]}],cfg,owners)
    units=[]
    for i in (0,1):
        survivors=new.players[i][2]
        items=old.players[i][2]
        units.append(dict(died=sorted(items.keys()-survivors.keys()),joined=sorted(survivors.keys()-items.keys()),
            cargo_delta={uid:round(survivors[uid][1]-items[uid][1],5)
                         for uid in items.keys()&survivors.keys() if items[uid][1]!=survivors[uid][1]},
            stationary_collectors={uid:round(survivors[uid][1]-items[uid][1],5) for uid in items.keys()&survivors.keys()
                if items[uid][2]=='COLLECTOR' and uid not in actions[i]}))
    return dict(board=board_digest(state),cash=[p[0] for p in new.players],ledger=ledger,
                receipts=receipts,lost_carbon=losses,tree_events=tree_events,units=units)


def trajectory(mods,snapshot,cfg,visible,prefix,turn):
    state,env=restore(snapshot,cfg)
    reset(mods,prefix)
    rows=[]
    for t in range(turn,turn+5):
        for m in mods:
            if hasattr(m,'_AUDIT'):
                m._AUDIT=dict(proposed=[],stages=[])
        actions=[decision(m,state,i,t,visible,SALT) for i,m in enumerate(mods)]
        step(state,env,actions)
        rows.append((actions,board_digest(state)))
    return rows


def memory_digest(mod):
    tracker=mod._ST['tr']
    belief=list(tracker.bel.estimate())
    return digest(dict(previous=mod._ST['prev'],destinations=tracker.last_dest,
                       belief=belief,
                       forecasts=sorted(mod._FC_HISTORY)))


def audit(case,cache,out,earlier):
    path=cache/(case['match']+'-1.html.gz')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==case['replay_sha256']
    game=read_replay(path)
    cfg=game['configuration']
    assert cfg['randomSeed']==case['seed']
    side,t=case['seat'],case['turn']
    mods=[model(out/'v8.py',f'score_plain_{i}',True) for i in (0,1)]
    traced=model(out/'traced.py','score_trace',True)
    assigned=mods.copy()
    assigned[side]=traced
    state,mem,visible=replay_prefix(game,mods,t)
    snapshot=archive_state(state)
    initial_digest=digest(snapshot)
    assert initial_digest==earlier['v8']['board']
    reset(mods,mem)
    memory=memory_digest(mods[side])
    plain=[decision(m,state,i,t,visible,SALT) for i,m in enumerate(mods)]
    state,_=restore(snapshot,cfg)
    reset(assigned,mem)
    traced._AUDIT=dict(proposed=[],stages=[])
    observed=observe(state,side,t)
    act=decision(traced,state,side,t,visible,SALT)
    assert act==plain[side]==earlier['v8']['arms']['original']['action']
    assert plain[1-side]==earlier['v8']['enemy_first']
    log=traced._AUDIT
    winner,runner,delta=reconstruct(log,t)
    assert log['admitted'][runner]==earlier['v8']['arms']['runner_up']['raw']
    for score,ref in ((log['final_scores'][winner],earlier['v8']['arms']['runner_up']['audit']['winning_score']),
                      (log['final_scores'][runner],earlier['v8']['arms']['runner_up']['audit']['finalist_score'])):
        close(score,ref)
    traced._AUDIT=dict(proposed=[],stages=[],force=runner)
    reset([traced],[mem[side]])
    state,_=restore(snapshot,cfg)
    forced=decision(traced,state,side,t,visible,SALT)
    assert forced==earlier['v8']['arms']['runner_up']['action']
    destinations={u:LAND[q][forced.get(u,'STAY')] for u,(q,_,_) in observed['players'][side][2].items()}
    assert traced._ST['tr'].last_dest==destinations
    assert traced._ST['liquidity_action']==forced
    assert traced._AUDIT['final_scores']==log['final_scores']
    traced._AUDIT=None

    # With the detailed recorder still active, compare complete joint actions and referee states.
    plain_steps=trajectory(mods,snapshot,cfg,visible,mem,t)
    traced_steps=trajectory(assigned,snapshot,cfg,visible,mem,t)
    assert plain_steps==traced_steps
    changed=[]
    own=observed['players'][side]
    for uid in sorted(act.keys()|forced.keys()):
        commands=[a.get(uid,'STAY') for a in (act,forced)]
        if commands[0]==commands[1]:
            continue
        pos,load,kind=own[2][uid]
        tiles=[LAND[pos][d] for d in commands]
        near=[(i,u,p,c,k) for i,team in enumerate(observed['players']) for u,(p,c,k) in team[2].items()
              if u!=uid and any(p in LAND[z].values() for z in tiles)]
        changed.append(dict(unit=uid,commands=commands,position=pos,cargo=load,kind=kind,
                            tiles=tiles,nearby=near))
    replies=dict(sampled=log['replies'][0],natural=plain[1-side],historical=game['steps'][t+1][1-side]['action'] or {})
    response_scores={}
    for label,reply in replies.items():
        if reply==replies['sampled']:
            trace=log
        else:
            state,_=restore(snapshot,cfg)
            reset([traced],[mem[side]])
            traced._AUDIT=dict(proposed=[],stages=[],reply_override=reply)
            decision(traced,state,side,t,visible,SALT)
            trace=traced._AUDIT
            traced._AUDIT=None
            assert trace['admitted']==log['admitted'] and trace['priors']==log['priors']
            assert trace['replies'][0]==reply
        res=dict(selected=trace['selected'],before_tail=trace['before_tail'],
                 finalists=trace['final_scores'],tail=trace.get('tail'),leaf=trace.get('leaf',[]))
        response_scores[label]=res
    choices={str(i):a for i,a in enumerate(log['admitted'])}
    choices.update(executed=act,forced_runner=forced)
    # The two Artemis deviations are separable for one transition.
    if len(changed)==2 and case['match']==MATCHES[1]:
        for r in changed:
            single=dict(act)
            single[r['unit']]=r['commands'][1]
            choices['only_'+r['unit']]=single
    transitions={}
    for name,a in choices.items():
        transitions[name]={}
        for label,reply in replies.items():
            actions=[None,None]
            actions[side],actions[1-side]=a,reply
            transitions[name][label]=one_step(snapshot,cfg,actions)
    observed_input={k:observed[k] for k in ('players','carbon','trees','step','player')}
    former={}
    for label,row in earlier.items():
        a=row['arms']['original']['played']['marks']['299']['net_cash']
        b=row['arms']['runner_up']['played']['marks']['299']['net_cash']
        former[label]=round(b-a,2)
    return dict(case=case,board=initial_digest,observation=observed_input,memory=memory,
                log=log,winner=winner,runner=runner,score_delta=delta,changed=changed,
                replies=replies,response_scores=response_scores,transitions=transitions,historical_terminal_deltas=former,
                consistency=dict(root=True,full_five_step_digest=digest(plain_steps),
                                 finalist_action_reproduced=True,score_reproduced=True,forced_memory_matches=True))


def main(cache,out):
    configure_engine()
    out.mkdir(parents=True,exist_ok=False)
    tag,expected=REVISIONS['yan-v8']
    raw=subprocess.check_output(['git','show',tag+':submission.py'],cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest()==expected
    (out/'v8.py').write_bytes(raw)
    (out/'traced.py').write_text(instrument(raw.decode(),expected))
    previous=ROOT/'analysis/team-branches'
    protocol=json.loads((previous/'protocol.json').read_text())
    roots={c['match']:c for c in protocol['roots'] if c['cohort']=='trees'}
    with gzip.open(previous/'branches.jsonl.gz','rt') as f:
        prior={k:{} for k in MATCHES}
        for line in f:
            r=json.loads(line)
            identifier=r['case']['match']
            if identifier in prior and r['case']['cohort']=='trees':
                prior[identifier][r['response']]=r
    rows=[audit(roots[k],cache,out,prior[k]) for k in MATCHES]
    import numpy
    import torch
    write(out/'protocol.json',dict(frozen=[tag,expected],root_salt=SALT,
         search=dict(width=4,depth=2,replies=1,tail=3,wall_clock=False),
         roots=[roots[k] for k in MATCHES],runtime=dict(python=platform.python_version(),
         numpy=numpy.__version__,torch=torch.__version__)))
    with gzip.open(out/'trace.json.gz','wt') as f:
        json.dump(rows,f,default=lambda x:x.item())
    write(out/'summary.json',[dict(case=r['case'],changed=r['changed'],score_delta=r['score_delta'],
                followup_margins=r['historical_terminal_deltas'],consistency=r['consistency']) for r in rows])


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='0':
        os.environ['PYTHONHASHSEED']='0'
        os.execv(sys.executable,[sys.executable,*sys.argv])
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
        os.environ[k]='1'
    os.environ['CARBON_DEBUG']='1'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    main(a.cache.resolve(),a.out.resolve())
