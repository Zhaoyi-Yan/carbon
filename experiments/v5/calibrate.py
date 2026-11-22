"""Calibrate context specialization and opponent matchups on training games only."""
import argparse
import json
from pathlib import Path
import numpy as np
from extract import TEACHERS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for v in ('split','cache','action','territory','output'):p.add_argument(v,type=Path)
    p.add_argument('snapshots',type=Path,nargs='+')
    a=p.parse_args();model=dict(np.load(a.action))
    train=set(json.loads(a.split.read_text())['training_matches'])
    C=np.concatenate([np.load(a.cache/(n+'.npz'))['C'] for n in sorted(train)])
    model['route_center']=model['route'] @ C.mean(0)
    games={}
    for root in a.snapshots:
        for g in json.loads((root/'index.json').read_text()):
            name=Path(g['file']).name.split('.')[0]
            if name in train:games[name]=g
    rank=json.loads((Path(__file__).parent/'design.json').read_text())['ranking']
    scores=np.zeros((20,20));counts=np.zeros((20,20))
    for g in games.values():
        slots=g['order']
        teachers=[TEACHERS.get(g['strategies'][slot]['id']) for slot in slots]
        if None in teachers:continue
        l,r=teachers
        expected=1/(1+10**((rank[r]['rating']-rank[l]['rating'])/400.))
        score=g['scores'][slots[0]]-g['scores'][slots[1]]
        excess=(.5+.5*np.sign(score))-expected
        scores[l,r]+=excess;scores[r,l]-=excess;counts[l,r]+=1;counts[r,l]+=1
    model['matchup']=scores/(counts+5.)
    model['territory']=np.load(a.territory)['territory']
    np.savez_compressed(a.output,**model)


if __name__=='__main__':
    main()
