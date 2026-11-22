"""Check replacement moves against the original referee while holding the recorded reply fixed."""
import argparse
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
import last_worker as guard
from benchmark import configure_engine

class NeutralLogits:
    def __getitem__(self,key):
        return 0.

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    configure_engine()
    from zerosum_env.envs.carbon.helpers import Board
    guard.WACT=['NONE','UP','RIGHT','DOWN','LEFT']
    guard.WIDX={k:i for i,k in enumerate(guard.WACT)}
    guard._ROLLOUT_LANDING=[tuple(((pos//15+r)%15)*15+(pos%15+c)%15 for r,c in ((0,0),(-1,0),(0,1),(1,0),(0,-1))) for pos in range(225)]
    changed=[];filtered=0;total=0
    for game in json.loads((args.snapshot/'index.json').read_text()):
        key=next((k for k,v in game['strategies'].items() if v['name']=='yan-v5'),None)
        if key is None:continue
        side=game['order'].index(key)
        data=json.loads(gzip.decompress((args.snapshot/game['file']).read_bytes()))
        cfg=data['configuration']
        for frame,after in zip(data['steps'],data['steps'][1:]):
            obs=dict(frame[0]['observation'],player=side)
            player=obs['players'][side];total+=1
            if player[3] or len(player[2])!=1 or player[0]>=min(cfg['recCollectorCost'],cfg['recPlanterCost']):continue
            filtered+=1
            actions=[x['action'] or {} for x in after]
            new=guard._v6_last_worker(obs,cfg,actions[side],NeutralLogits())
            if new==actions[side]:continue
            obs['carbon']=obs['full_carbon']
            observed=after[0]['observation']['players'][side]
            pair=actions.copy();pair[side]=new
            judged=Board(obs,cfg,pair).next().observation['players'][side]
            assert len(judged[2])==1
            changed.append({'game':game['id'],'step':obs['step'],'recorded':actions[side],'safe':new,
                            'recorded_workers_after':len(observed[2]),'safe_workers_after':len(judged[2])})
    args.output.write_text(json.dumps({'turns':total,'eligible':filtered,'changed':changed},indent=2)+'\n')
    print(total,filtered,len(changed),sum(not x['recorded_workers_after'] for x in changed))

if __name__=='__main__':main()
