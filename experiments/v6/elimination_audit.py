"""Check the official rewards and exhaust final worker moves with the rival's recorded reply."""
import argparse
import gzip
import itertools
import json
from pathlib import Path
import sys
from liquidity import _v6_eliminated

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    configure_engine()
    from zerosum_env.envs.carbon.helpers import Board
    games=json.loads((args.snapshot/'index.json').read_text())
    checked=0
    cases=[]
    for g in games:
        key=next((k for k,v in g['strategies'].items() if v['name']=='yan-v5'),None)
        if key is None:
            continue
        data=json.loads(gzip.decompress((args.snapshot/g['file']).read_bytes()))
        frames=data['steps']
        config=data['configuration']
        floor=min(config['recCollectorCost'],config['recPlanterCost'])
        side=g['order'].index(key)
        for frame in frames[:-1]:
            assert not any(_v6_eliminated(p,floor) for p in frame[0]['observation']['players'])
        final=frames[-1][0]['observation']
        flags=[_v6_eliminated(p,floor) for p in final['players']]
        assert any(flags)==(g['steps']<config['episodeSteps']-1)
        checked+=1
        if not any(flags):
            continue
        previous=dict(frames[-2][0]['observation'])
        for i,flag in enumerate(flags):
            reward=final['step']-config['episodeSteps']-1 if flag else previous['players'][i][0]
            assert reward==g['scores'][g['order'][i]]
        previous['carbon']=previous['full_carbon']
        player=previous['players'][side]
        assert player[0]<floor and len(player[2])<=2
        units=list(player[2])
        actions=[dict(f['action'] or {}) for f in frames[-1]]
        baseline=Board(previous,config,actions).next().observation
        assert baseline['players']==final['players']
        rows=[]
        for moves in itertools.product((None,'UP','RIGHT','DOWN','LEFT'), repeat=len(units)):
            change=actions.copy()
            change[side]={k:a for k,a in zip(units,moves) if a is not None}
            after=Board(previous,config,change).next().observation
            mine=after['players'][side]
            rows.append({'moves':moves, 'workers':len(mine[2]), 'cash':mine[0],
                         'survives':not _v6_eliminated(mine,floor)})
        cases.append({'match':g['id'],'seat':side,'end':final['step'],'units':units,
                      'cash_before':player[0],'recorded':actions[side],'alternatives':rows})
    args.output.write_text(json.dumps({'checked':checked,'cases':cases},indent=2)+'\n')
    for c in cases:
        print(c['end'],sum(r['survives'] for r in c['alternatives']),len(c['alternatives']))

if __name__=='__main__':
    main()
