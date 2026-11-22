"""Replay all early courier opportunities against the original Master7sE instructions."""
import json
from collections import Counter
from pathlib import Path
from branch_replay import ROOT, OUT, configure_engine,step
from benchmark import load_strategy,limit_math_threads,new_game,observe
from close_replay import restore,archive_state,referee_events
from starvation_replay import read_replay
from depot_corridor import _v7_corridor


def main():
    limit_math_threads();configure_engine();m=load_strategy('courier_check',ROOT/'experiments/v7/checkpoints/soundings.py')
    _v7_corridor.__globals__.update({k:getattr(m,k) for k in ('_V7_L','_V7_D','WIDX','WACT','LAST_TURN')})
    from zerosum_env.envs.carbon import idgen
    path=ROOT.parent/'v7-data/starvation-4063.html.gz';g=read_replay(path);f=g['steps'];cfg=g['configuration']
    new_game(cfg['randomSeed'],300)
    init=dict(seats=[dict(observation=b['observation'],status=b['status'],reward=b['reward'],action=b['action'] or {}) for b in f[0]],id_counters=dict(idgen._id_dict))
    wanted=[48,49,89,249,268];state,env=restore(init,cfg);starts={};prev=None;prior={};hist={}
    for t in range(max(wanted)+1):
        assert state[0].observation.players==f[t][0]['observation']['players']
        if t in wanted:starts[t]=(archive_state(state),dict(hist),prev,prior)
        o=observe(state,1,t);m._v7_watch_cash(o,prev,prior,hist);act=[b['action'] or {} for b in f[t+1]]
        prev=o['players'];prior=act[1];step(state,env,act)
    results=[]
    for start,(ss,hh,prev0,pr) in starts.items():
        for variant in ('recorded','two_turn','four_turn'):
            state,env=restore(ss,cfg);h=dict(hh);prev=prev0;prior=pr;m._ST={};ledger=Counter();marks=[];changes=[]
            for t in range(start,start+12):
                o=observe(state,1,t);drought=m._v7_watch_cash(o,prev,prior,h);act=[dict(b['action'] or {}) for b in f[t+1]]
                # Re-use historical worker IDs for this short conditional intervention only.
                us=o['players'][1];before=dict(act[1])
                if variant!='recorded':act[1]=m._v7_short_deposit(o,cfg,act[1],drought)
                if variant=='four_turn':act[1],_=_v7_corridor(o,cfg,act[1],drought,m._ST)
                if act[1]!=before:changes.append([t,{k:[before.get(k),act[1].get(k)] for k in before.keys()|act[1].keys() if before.get(k)!=act[1].get(k)}])
                prev=o['players'];prior=act[1];events=[]
                with referee_events(events):step(state,env,act)
                for row in events:
                    if row[1]==1:ledger[row[0]]+=row[2]
                if t-start+1 in (4,6,12):
                    cash=state[0].observation.players[1][0]
                    marks.append(dict(horizon=t-start+1,cash=cash,deposits=round(ledger['bank']+ledger['base_collision'],2)))
            results.append(dict(start=start,variant=variant,changes=changes,marks=marks))
    (OUT/'corridor-branches.json').write_text(json.dumps(results,indent=2)+'\n')
    for r in results:print(r['start'],r['variant'],r['marks'][0],r['marks'][-1])
if __name__=='__main__':main()
