"""Focused recorded-route stress plus prospective whole-layout evaluation of paid hire relocation."""
import copy
import gzip
import hashlib
import json
import os
import numpy as np
import preempt_tree_probe as core
import preempt_depot
from preempt_tree import HERE,MODEL,_V7_PT_D as D
from branch_replay import write
from close_replay import archive_state,restore
from benchmark import new_game,configure_engine,limit_math_threads,observe,validate_actions
ROOT=HERE.parents[1];OUT=ROOT/'analysis/v7'


def open_loop(row,cfg,frames,t,side,coef):
    state,env=new_game(row['seed'],300)
    for i in range(t):core.step(state,env,row['actions_and_post_cash'][i][0])
    snap=archive_state(state);results=[]
    for treatment in (False,True):
        state,env=restore(snap,cfg);interest=None;marks={};edits=[];newcash=0.;tree_turns=0;uncmd=0
        for off in range(core.HORIZON):
            at=t+off;ob=observe(state,side,at);actions=[]
            for s,record in enumerate(frames[at+1]):
                pl=state[0].observation.players[s];allowed=set(pl[1])|set(pl[2]);a={k:v for k,v in record['action'].items() if k in allowed}
                hist=frames[at][0]['observation']['players'][s][2];uncmd+=len(pl[2].keys()-hist.keys())
                actions.append(a)
            b=actions[side]
            if off==0:
                alt,interest=preempt_depot.plan(ob,cfg,b,frames[at-1][0]['observation']['players'],coef)
                focus=interest['tree'] if interest else None
            elif off<=3 and interest:alt,interest=preempt_depot.follow(ob,cfg,b,interest)
            else:alt=b
            if treatment and alt!=b:edits.append([at,[k for k in sorted(alt.keys()|b.keys()) if alt.get(k)!=b.get(k)]]);actions[side]=alt
            for s,a in enumerate(actions):validate_actions(a,state[0].observation.players[s])
            core.step(state,env,actions);ps=state[0].observation.players;meta=state[0].observation.trees
            owner=(1 if focus in ps[side][3] else (-1 if focus in ps[1-side][3] else 0))
            newcash+=owner*meta.get(focus,(None,0))[1];tree_turns+=owner
            if not treatment:assert [p[0] for p in ps]==row['actions_and_post_cash'][at][1]
            if off+1 in (5,12,18):marks[str(off+1)]=dict(cash=[p[0] for p in ps],tree_owner_relative=owner,target_net_income=round(newcash,2),target_relative_turns=tree_turns)
        results.append(dict(marks=marks,edits=edits,unmatched_worker_turns=uncmd))
    return results


def choose_windows(cfg,frames,coef,key):
    windows=[]
    for t in range(25,len(frames)-core.HORIZON):
        o=frames[t][0]['observation'];prior=frames[t-1][0]['observation']['players']
        for s in (0,1):
            # Condition on the historical move to study decisions that actually funded
            # a recruit. No later state or post-action outcome is used to pick branches.
            act=frames[t+1][s]['action'] or {}
            if 'RECCOLLECTOR' not in act.values():continue
            _,intent=preempt_depot.plan(dict(o,step=t,player=s),cfg,act,prior,coef)
            if intent:windows.append((hashlib.sha256(f'preempt-depot-turn/{key}/{s}/{t}'.encode()).hexdigest(),t,s,intent))
    return sorted(windows)


def main():
    os.environ['CARBON_DEBUG']='1';limit_math_threads();configure_engine()
    assert hashlib.sha256(core.CP.read_bytes()).hexdigest()==core.SHA
    reg=json.loads(MODEL.read_text());coef=reg['coef']
    core.plan,core.follow=preempt_depot.plan,preempt_depot.follow
    m=core.model(core.CP,'depot_soundings',True);rival=core.model(ROOT/'experiments/v6r/checkpoints/estuary.py','depot_v6',False)
    rr=[json.loads(l) for l in (OUT/'05-qualifying.jsonl').open()]
    case=next(r for r in rr if r['seed']==590159492 and r['players'][0]['name']=='soundings');cfg,fs,lay=core.local_frames(case)
    local=dict(seed=590159492,turn=209,side=0,hindsight_selected_largest_home_grove_capture=True,
               historical_stress=open_loop(case,cfg,fs,209,0,coef),live=core.outcomes(cfg,fs,lay,209,0,'local_discovery',coef,m,rival))
    excluded=set(reg['training_maps'])|set(reg['test_maps'])
    rows=json.loads((core.DATA/'receipts-cache/index.json').read_text())['games']
    pool=[r for r in rows if r['layout'] not in excluded and r['end']>=240 and r['file'].split('-')[0] not in ('03100a95e8647c51a4d642fe08d856dd','4063a34afedb6d4767a9118e1dee9e5d','cf276b32e4fd38f43497e60f271a3854')]
    pool.sort(key=lambda r:hashlib.sha256(('preempt-hire-independent/'+r['layout']).encode()).hexdigest())
    attempts=[];skipped=[];active=0
    def save():
        n=len(attempts);ok=[x for x in attempts if x['edits']]
        summary=dict(layouts_attempted=n,active=len(ok),unchanged=n-len(ok),
          active_positive_cash18=sum(c['delta']['18']['cash']>0 for c in ok),active_negative_cash18=sum(c['delta']['18']['cash']<0 for c in ok),
          mean_active_deltas={h:{k:round(float(np.mean([c['delta'][h][k] for c in ok])),2) for k in ('cash','bank','tree_income','tree_cost','target_owned_turns','target_income')} for h in ('5','12','18')} if ok else {})
        write(OUT/'preempt-depot-branches.json',dict(local=local,independent=attempts,skipped_maps=skipped,summary=summary,
           candidate_sha256=hashlib.sha256((HERE/'preempt_depot.py').read_bytes()).hexdigest(),frozen_sha256=core.SHA,
           predeclared='Independent hash order, excludes 162 risk-study maps. One event per map. First 10 active decisions or 16 attempted, whichever comes first; also cap the prospective archival scan at 85 maps. Only recorded current recruits and pre-action observations select windows.'))
    save();print('focused',local['historical_stress'],local['live']['delta']['18']['cash'],flush=True)
    for r in pool[:85]:
        cfg,fs,k=core.parse_public(r);win=choose_windows(cfg,fs,coef,k)
        if not win:skipped.append(k);continue
        _,t,s,meta=win[0]
        outcome=core.outcomes(cfg,fs,k,t,s,'independent_depot',coef,m,rival);attempts.append(outcome)
        if outcome['edits']:active+=1
        print('depot',len(attempts),active,t,outcome['delta']['18']['cash'],flush=True);save()
        if active>=10 or len(attempts)>=16:break
    save()


if __name__=='__main__':main()
