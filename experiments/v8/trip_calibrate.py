"""Build five trip-length priors and diagnose the earlier cargo-only seat pairings."""
import gzip
import json
from pathlib import Path
from statistics import mean
import sys
from trip_history import _v8_observe_trips
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine, new_game


def main():
    configure_engine()
    from zerosum_env.envs.carbon import carbon
    root=ROOT/'analysis/v8'
    seen=[]
    audit=[]
    for family in ('payback','option'):
        with gzip.open(root/(family+'-games.jsonl.gz'),'rt') as games:
            for row in map(json.loads,games):
                if not any(p['name']=='cargo' for p in row['players']):
                    continue
                state,env=new_game(row['seed'],300)
                own=next(i for i,p in enumerate(row['players']) if p['name']=='cargo')
                memories=[{},{}]
                before=state[0].observation.players
                for s in (0,1):
                    _v8_observe_trips(memories[s], before[s], None, 0)
                for t,(actions,expected) in enumerate(row['actions_and_post_cash'],1):
                    for entry,action in zip(state,actions):
                        entry.action=action
                    carbon.compute_next_board(state,env)
                    after=state[0].observation.players
                    assert [p[0] for p in after]==expected
                    for s in (0,1):
                        _v8_observe_trips(memories[s],after[s],before[s],t)
                    before=after
                seen.extend(memories[1-own]['completed'])
                if family=='option':
                    cash=row['actions_and_post_cash'][124][1]
                    audit.append(dict(seed=row['seed'],seat=own,pre125=round(cash[own]-cash[1-own],2),
                                      margin=round(row['scores'][own]-row['scores'][1-own],2)))
    # A port-to-port cycle includes unloading, repositioning and time spent idle.
    examples=sorted((duration,credit) for t,origin,duration,credit in seen)
    bands=[examples[i*len(examples)//5:(i+1)*len(examples)//5] for i in range(5)]
    priors=[(round(mean(x[0] for x in band)),round(mean(x[1] for x in band),2)) for band in bands]
    (root/'trip-prior.json').write_text(json.dumps(dict(source='v7 seats from 16 prior cargo games',count=len(examples),prior=priors,previous_seats=audit),indent=2)+'\n')
    print(len(examples),priors)

if __name__=='__main__':
    main()
