"""Apply tree labels to the archived evaluation data and reproduce actual sampling coverage."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from collect import write
from dataset import read_replay
from fit import assess, evaldata, sample, EPOCHS
from network import make, PARENT
from tree_labels import tree_flags, PLANTED, SEIZED


def main(data, out):
    teachers=json.loads((data/'teachers.json').read_text())
    ids={r['id']:i for i,r in enumerate(teachers)}
    maps=json.loads((data/'split.json').read_text())
    report={'samples':{}}
    for arm in ('top6','top6-plus'):
        seen=set();counts=[]
        for epoch in range(1,EPOCHS+1):
            sample(data,maps,arm,epoch,seen,fields=())
            counts.append(len(seen))
        report['samples'][arm]=counts
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(1)
    rows=json.loads((data/'forward-test.json').read_text())['maps']
    samples=evaldata(data,rows,'forward')
    offset=0
    for row in rows:
        replay=read_replay(data/'official'/(row['game']+'.html.gz'))
        frames=replay['steps']
        for teacher in row['teachers']:
            for side,entry in enumerate(row['seats']):
                if ids.get(entry)!=teacher:continue
                for t in range(0,len(frames)-1,6):
                    assert samples['T'][offset].item()==teacher
                    now=frames[t][0]['observation']['players']
                    then=frames[t+1][0]['observation']['players']
                    flags=tree_flags(now,then,side,frames[t+1][side]['action'] or {},replay['configuration'])
                    saved=samples['M'][offset].numpy()
                    assert np.array_equal((saved&2)!=0,(flags&(PLANTED|SEIZED))!=0)
                    saved|=flags
                    offset+=1
    assert offset==len(samples['X'])
    samples={k:v[samples['T']<6] for k,v in samples.items()}
    for size in ('main','small'):
        report[size]={}
        for arm in ('yan-v7','top6-plus'):
            text=(PARENT if arm=='yan-v7' else data/'top6-plus.py').read_text()
            model=make(text,size=='small',extended=arm!='yan-v7').to(device)
            result=assess(model,samples,device)
            report[size][arm]={k:v for k,v in result.items()
                               if k.split('/')[1] in ('planted','seized','left_plant','left_seize','loaded_return','low_cash','recruit','value')}
    write(out/'tree-diagnosis.json',report)


if __name__=='__main__':main(*[Path(x).resolve() for x in sys.argv[1:]])
