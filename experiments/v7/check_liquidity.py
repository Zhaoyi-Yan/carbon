"""Exercise the packaged candidate's treasury and two-step delivery hooks in the official game."""
import argparse
import json
from pathlib import Path
from branch_replay import step
from starvation_replay import read_replay
from benchmark import configure_engine,limit_math_threads,load_strategy,new_game,observe
from close_replay import restore,archive_state,referee_events


def run(model,replay):
    from zerosum_env.envs.carbon import idgen
    d=read_replay(replay);cfg=d['configuration'];frames=d['steps']
    new_game(cfg['randomSeed'],300)
    init=dict(seats=[dict(observation=p['observation'],status=p['status'],reward=p['reward'],action=p['action'] or {}) for p in frames[0]],
              id_counters=dict(idgen._id_dict))
    state,env=restore(init,cfg);side=1;history={};previous=None;action=None
    release=None
    for t in range(91):
        assert state[0].observation.players==frames[t][0]['observation']['players']
        o=observe(state,side,t);drought=model._v7_watch_cash(o,previous,action,history)
        previous=o['players'];actions=[f['action'] or {} for f in frames[t+1]];action=actions[side]
        if t==86:
            revised=model._v7_keep_float(o,cfg,action,drought)
            bought=[k for k in action if action[k]=='RECPLANTER']
            assert bought and all(k not in revised for k in bought)
            release=dict(step=t,drought=drought,original_hire=[action[b] for b in bought],alternative=[revised.get(b) for b in bought])
        if t==90:saved=archive_state(state);break
        step(state,env,actions)
    rows=[]
    for use_candidate in (False,True):
        state,env=restore(saved,cfg);model._ST.pop('v7_next_depot',None)
        actions_out=[];cash_events=[]
        for t in (90,91):
            actions=[f['action'] or {} for f in frames[t+1]]
            if use_candidate:actions[side]=model._v7_short_deposit(observe(state,side,t),cfg,actions[side],drought+t-90)
            actions_out.append(actions[side].get('player-1-worker-11'))
            events=[]
            with referee_events(events):step(state,env,actions)
            cash_events.extend([t+1,*e] for e in events if e[1]==side and e[0] in ('bank','base_collision','plant','seize','tree_yield'))
        rows.append(dict(applied=use_candidate,courier_actions=actions_out,cash=state[0].observation.players[side][0],cash_events=cash_events))
    assert rows[0]['cash']==7.53 and rows[1]['courier_actions']==['DOWN','DOWN'] and rows[1]['cash']==120.7,rows
    return dict(verified_prefix_frames=91,own_seat=side,release=release,branches=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('replay',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();limit_math_threads();configure_engine()
    result=run(load_strategy('liquidity_release',a.candidate),a.replay)
    import hashlib
    result['candidate_sha256']=hashlib.sha256(a.candidate.read_bytes()).hexdigest()
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(result['release'],[(r['courier_actions'],r['cash']) for r in result['branches']])
