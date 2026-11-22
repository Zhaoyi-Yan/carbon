"""Identify when the terminal loads first became substantial, using the saved referee actions."""
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import new_game,configure_engine,limit_math_threads

def terminal_loads(path,seat):
    configure_engine();limit_math_threads()
    from zerosum_env.envs.carbon import carbon
    archive=json.loads(gzip.decompress(path.read_bytes()))
    result=archive['result']
    log=archive['actions_and_post_cash']
    state,env=new_game(result['seed'],result['episode_steps'])
    history=[]
    for turn in range(len(log)+1):
        players=state[0].observation.players
        own=players[seat]
        history.append({k:(v[0],v[1]) for k,v in own[2].items() if v[2]=='COLLECTOR'})
        if turn==len(log):break
        pair,cash=log[turn]
        for s,a in zip(state,pair):s.action=a
        carbon.compute_next_board(state,env)
        assert [p[0] for p in state[0].observation.players]==cash
    def distance(p,q):
        y,x=divmod(p,15);yy,xx=divmod(q,15);dy=abs(y-yy);dx=abs(x-xx)
        return min(dy,15-dy)+min(dx,15-dx)
    records=[]
    for uid,(pos,carried) in history[-1].items():
        if carried<20:continue
        turn=len(log)
        while turn and history[turn-1].get(uid,(None,0))[1]>=.90*carried:turn-=1
        earlier,load=history[turn][uid]
        records.append(dict(worker=uid,final_cargo=carried,turn_reached_90_percent=turn,cargo_at_that_turn=load,
                            moves_left=len(log)-turn,minimum_home_distance=min(distance(earlier,b) for b in own[1].values())))
    return dict(seed=result['seed'],seat=seat,winner=result['winner'],scores=result['scores'],terminal_loads=records)

if __name__=='__main__':
    data=ROOT/'analysis/v6r-diagnostic-actions'
    results=[terminal_loads(path,int(path.name.split('seat')[-1][0])) for path in sorted(data.glob('*.gz'))]
    (ROOT/'analysis/v6r-stranded.json').write_text(json.dumps(results,indent=2)+'\n')
