"""Reconstruct tree location, ownership spells, and realized income after a seizure."""
import gzip
import json
import sys
from pathlib import Path

from evaluate import configure_engine, new_game
from starvation_replay import distance


def spells(row):
    from zerosum_env.envs.carbon import carbon
    state,env=new_game(row['seed'],row['episode_steps'])
    home=[list(p[1].values()) for p in state[0].observation.players]
    active={};records=[]
    for t,(actions,cash) in enumerate(row['actions_and_post_cash'],1):
        for s,a in zip(state,actions):s.action=a
        carbon.compute_next_board(state,env)
        obs=state[0].observation
        assert [p[0] for p in obs.players]==cash
        present={}
        for owner,player in enumerate(obs.players):
            for tid,(cell,age) in player[3].items():
                old=active.get(tid)
                if old is None or old['owner']!=owner:
                    if old:old['end_reason']='seized'
                    item=dict(tree=tid,cell=cell,owner=owner,start=t,age_at_start=age,
                              origin='capture' if old else 'plant',
                              home_distance=min(distance(cell,b) for b in home[owner]),yield_sum=0)
                    records.append(item)
                else:item=old
                present[tid]=item
                item['last']=t
                item['yield_sum']+=obs.trees[tid][1]
        for tid in active.keys()-present.keys():active[tid]['end_reason']='expired'
        active=present
    for item in active.values():item['end_reason']='game_end'
    for item in records:item['yield_sum']=round(item['yield_sum'],4)
    return records


def main(out):
    configure_engine()
    with gzip.open(out/'tree-spells.jsonl.gz','wt') as dest:
        for stage in ('screening','retry','confirmation','confirmation-after-retry'):
            path=out/(stage+'.jsonl.gz')
            if not path.exists():continue
            with gzip.open(path,'rt') as f:
                for row in map(json.loads,f):
                    if row['status']!='completed':continue
                    dest.write(json.dumps(dict(seed=row['seed'],stage=stage,
                                               players=[x['name'] for x in row['players']],
                                               spells=spells(row)))+'\n')


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
