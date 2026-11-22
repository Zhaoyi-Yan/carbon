"""Run the published agent entry at three actual prefix states and follow the committed courier."""
import argparse
import hashlib
import json
from pathlib import Path
from branch_replay import ROOT,OUT,step
from benchmark import configure_engine,limit_math_threads,load_strategy,new_game,observe,validate_actions
from close_replay import restore,archive_state,referee_events
from starvation_replay import read_replay


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();limit_math_threads();configure_engine()
    from zerosum_env.envs.carbon import idgen
    src=a.candidate.read_bytes();g=read_replay(ROOT.parent/'v7-data/starvation-4063.html.gz');frames=g['steps'];cfg=g['configuration']
    public_cfg={k:v for k,v in cfg.items() if k!='randomSeed'};data=[]
    new_game(cfg['randomSeed'],300)
    init=dict(seats=[dict(observation=f['observation'],status=f['status'],reward=f['reward'],action=f['action'] or {}) for f in frames[0]],id_counters=dict(idgen._id_dict))
    for start in (48,89):
        state,env=restore(init,cfg);m=load_strategy('entry_at_'+str(start),a.candidate)
        search,reply=m._search,m._reply_choice
        m._search=lambda o,c,e,base,*extra:base
        m._reply_choice=lambda o,c,e,base,*extra:base
        for t in range(start):
            o=observe(state,1,t);snapshot=dict(idgen._id_dict);m.agent(o,public_cfg);ids=idgen._id_dict;ids.clear();ids.update(snapshot)
            cmds=[f['action'] or {} for f in frames[t+1]]
            m._ST['tr'].set_actions(o['players'],cmds[1]);m._ST['liquidity_action']=dict(cmds[1])
            m._ST.pop('corridor_track',None);m._ST.pop('v7_next_depot',None)
            step(state,env,cmds)
        m._search=search;m._reply_choice=reply
        assert state[0].observation.players==frames[start][0]['observation']['players']
        turns=[];deposits=0.
        for t in range(start,start+6):
            o=observe(state,1,t);snapshot=dict(idgen._id_dict);ours=m.agent(o,public_cfg);ids=idgen._id_dict;ids.clear();ids.update(snapshot)
            validate_actions(ours,o['players'][1])
            p0=o['players'][0];op={u:c for u,c in (frames[t+1][0]['action'] or {}).items() if u in p0[2] or u in p0[1]};ev=[]
            with referee_events(ev):step(state,env,[op,ours])
            deposits+=sum(v[2] for v in ev if v[1]==1 and v[0] in ('bank','base_collision'))
            turns.append(dict(turn=t,our_actions=ours,treasury=state[0].observation.players[1][0],gross=round(deposits,2)))
        data.append(dict(start=start,turns=turns));print(start,turns[-1]['treasury'],turns[-1]['gross'],flush=True)
    a.out.write_text(json.dumps(dict(sha256=hashlib.sha256(src).hexdigest(),method='Replay both recorded sides up to the fork with the candidate’s own trackers; after the fork run the unmodified agent while the opponent continues recorded commands for surviving IDs.',cases=data),indent=2)+'\n')
if __name__=='__main__':main()
