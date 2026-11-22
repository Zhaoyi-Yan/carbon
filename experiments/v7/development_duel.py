"""Compare conditional and always-on economic search on paired development games with an equal node budget."""
import argparse
import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from branch_replay import HERE, OUT, ROOT, board_digest, step
from benchmark import configure_engine, new_game, observe, validate_actions
from phase_switch import preserved_decision, single_thread
from preempt_tree_probe import model
from close_replay import referee_events


def game(seed, side, mine, rival):
    state, env = new_game(seed, 300)
    visible = {k:v for k,v in env.configuration.items() if k!='randomSeed'}
    position = board_digest(state)
    mods = [mine,rival] if side==0 else [rival,mine]
    cash = [Counter(),Counter()]
    triggers, marks = Counter(), {}
    for m in mods:
        m._ST, m._FC_HISTORY = {}, {}
    for turn in range(299):
        moves = []
        for seat, participant in enumerate(mods):
            obs = observe(state,seat,turn)
            act = preserved_decision(participant,obs,visible,seed)
            validate_actions(act,obs['players'][seat])
            moves.append(act)
        triggers[str(mine._ST.get('v7_depot_race'))] += 1
        events = []
        with referee_events(events):
            step(state,env,moves)
        for name,seat,delta,*other in events:
            cash[seat][name] += delta
        roster = state[0].observation.players
        end = turn+1
        done = end==299 or any(s.status!='ACTIVE' for s in state)
        if done or end in (50,100,150,190,215,250):
            marks[end] = dict(cash=[p[0] for p in roster], fleets=[len(p[2]) for p in roster], trees=[len(p[3]) for p in roster],
                              deposits=[round(x['bank']+x['base_collision'],2) for x in cash])
        if done:
            break
    scores = [s.reward for s in state]
    return dict(seed=seed,layout=position,seat=side,margin=round(scores[side]-scores[1-side],2),end=end,marks=marks,triggers=dict(triggers))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();single_thread();configure_engine()
    files = [a.first,a.second,ROOT/'yan-v6.py']
    modules = [model(path,'dev_duel_'+str(i),True) for i,path in enumerate(files)]
    result = dict(budget='Width 4, depth 2, one live reply, three tail plies. Per-step per-seat random streams shared across arms. Four previously designated development layouts. No tournament timing.',sha256=[hashlib.sha256(f.read_bytes()).hexdigest() for f in files],games=[])
    for seed in json.loads((HERE/'panels.json').read_text())['development']:
        for seat in (0,1):
            for arm, module in enumerate(modules[:2]):
                row=game(seed,seat,module,modules[2])
                row['arm']=arm;result['games'].append(row)
                print(seed,seat,arm,row['margin'],row['triggers'],flush=True)
            assert result['games'][-2]['layout']==result['games'][-1]['layout']
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    result['summary']={}
    for arm in (0,1):
        rows=[x for x in result['games'] if x['arm']==arm]
        result['summary'][arm]=dict(wins=sum(x['margin']>0 for x in rows),mean=round(statistics.mean(x['margin'] for x in rows),2),
                                    seat_wins=[sum(x['seat']==s and x['margin']>0 for x in rows) for s in (0,1)])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['summary'],flush=True)


if __name__=='__main__':
    main()
