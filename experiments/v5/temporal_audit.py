"""Compare frozen policies against executed expert moves on a later public snapshot."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from backbone import ROOT,configure_engine,limit_math_threads,load_strategy
from extract import TEACHERS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('snapshot',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();limit_math_threads();configure_engine()
    base=load_strategy('temporal_base',ROOT/'opponents/yan-v4.py')
    cand=load_strategy('temporal_candidate',a.candidate.resolve())
    seen={r['match'] for r in json.loads((Path(__file__).parent/'replay-manifest.json').read_text())}
    ranks=json.loads((Path(__file__).parent/'design.json').read_text())['ranking']
    sums=np.zeros((3,3));games=[];seats=[]
    for g in json.loads((a.snapshot/'index.json').read_text()):
        path=a.snapshot/g['file'];name=path.name.split('.')[0]
        if name in seen or not path.exists() or name in games:continue
        frames=json.loads(gzip.decompress(path.read_bytes()))['steps'];games.append(name)
        for side,key in enumerate(g['order']):
            teacher=TEACHERS.get(g['strategies'][key]['id'])
            if teacher is None:continue
            seats.append(teacher+1)
            tr=base.BeliefTracker(base._Bel,side);prev=None;xs=[];units=[]
            def flush():
                if not xs:return
                x=np.stack(xs)
                predictions=[base._net()(x,base._SID)[:2],cand._net()(x,20)[:2],cand._net()(x,teacher)[:2]]
                for k,(ww,bb) in enumerate(predictions):
                    for j,objects in enumerate(units):
                        for pos,kind,label in objects:
                            logit=(ww if kind==0 else bb)[j,pos].astype(float)
                            l=logit-logit.max();loss=np.log(np.exp(l).sum())-l[label]
                            sums[k]+=[1,logit.argmax()==label,loss]
                xs.clear();units.clear()
            for t,(before,after) in enumerate(zip(frames,frames[1:])):
                shared=before[0]['observation'];pl=shared['players'];act=after[side]['action'] or {}
                est,known=tr.step(pl,t,before[side]['observation']['carbon'])
                if t%5==int(g['id'][:4],16)%5:
                    xs.append(base._quant(base.encode(pl,t,side,est,known,prev)))
                    obj=[(v[0],0,base.WIDX.get(act.get(uid),0)) for uid,v in pl[side][2].items()]
                    if pl[side][0]>=30 and len(pl[side][2])<10:
                        obj.extend((pos,1,base.BIDX.get(act.get(uid),0)) for uid,pos in pl[side][1].items())
                    units.append(obj)
                    if len(xs)==12:flush()
                tr.set_actions(pl,act);prev=pl
            flush()
    result=dict(matches=games,teacher_ranks=seats,units=int(sums[0,0]),
                metrics={key:dict(accuracy=r[1]/r[0],nll=r[2]/r[0]) for key,r in zip(('v4','new_shared_style','new_known_style'),sums)})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['units'],result['metrics'])


if __name__=='__main__':
    main()
