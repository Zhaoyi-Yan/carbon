"""Measure the deployment FP16 serialization error on legal actions on 13 independent maps."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from master_spatial_check import HERE,PARENT,ROOT,NAMES,load_net,contents,load

def main(data,reference,packed):
    torch.set_num_threads(1)
    both=[load_net(path) for path in (reference,packed)]
    rotations=load(PARENT)._V5_ROTATIONS
    rows=json.loads((data/'master-worker/index.json').read_text())
    values={k:[[],[]] for k in ('worker','base')}
    for row in [r for r in rows if r['fold']=='test']:
        with gzip.open(row['file']) as f:g=json.load(f)
        x,_,_,_,ts,rots=contents(data/'master-spatial'/row['cache'])
        out=[]
        for net in both:
            q=[]
            with torch.inference_mode():
                for v in np.array_split(x,max(1,len(x)//24)):
                    styles=torch.full((len(v),),NAMES.index(row['teacher']),dtype=torch.long)
                    q.extend(net(torch.from_numpy(v.astype('f')),styles).numpy())
            out.append(np.stack(q).transpose(0,2,1))
        for t in np.unique(ts):
            ids=np.flatnonzero(ts==t);assert list(rots[ids])==list(range(4))
            p=g['steps'][int(t)][0]['observation']['players'][row['side']]
            for key,pos,offset,count in [('worker',[z[0] for z in p[2].values()],0,5),('base',list(p[1].values()),5,3)]:
                if not pos or (key=='base' and (p[0]<30 or len(p[2])>=10)):continue
                for k in range(2):
                    frames=[]
                    for j,i in enumerate(ids):
                        cells,_,actions,_=rotations[j]
                        z=out[k][i,cells[pos],offset:offset+count]
                        frames.append(z[:,actions] if key=='worker' else z)
                    values[key][k].extend(np.mean(frames,axis=0))
    report=dict(states=sum(len(np.unique(np.load(data/'master-spatial'/r['cache'])['t'])) for r in rows if r['fold']=='test'),maps=13,
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),half_sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),half_bytes=packed.stat().st_size)
    for k,(a,b) in values.items():
        a=np.stack(a);b=np.stack(b);err=np.abs(a-b)
        report[k]=dict(n=len(a),differ=int((a.argmax(1)!=b.argmax(1)).sum()),mean_abs=float(err.mean()),p95_abs=float(np.quantile(err,.95)),max_abs=float(err.max()))
    (ROOT/'analysis/v7/master-spatial-quantization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('reference',type=Path);p.add_argument('packed',type=Path)
    a=p.parse_args();main(a.data,a.reference,a.packed)
