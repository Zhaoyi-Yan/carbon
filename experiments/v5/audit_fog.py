"""Measure both fog estimators against archived maps after computing each legal estimate."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from backbone import ROOT, load_strategy, configure_engine, limit_math_threads


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','independent','output'):p.add_argument(key,type=Path)
    p.add_argument('snapshots',type=Path,nargs='+')
    args=p.parse_args()
    configure_engine();limit_math_threads()
    m=load_strategy('v5_fog_audit',args.candidate.resolve())
    targets=set(json.loads(args.independent.read_text())['independent_maps'])
    files={}
    for root in args.snapshots:
        for g in json.loads((root/'index.json').read_text()):
            path=root/g['file']
            if path.exists() and path.name.split('.')[0] in targets:
                files[path.name.split('.')[0]]=path
    errors=np.zeros((4,2,3)); routed=np.zeros(20,dtype=int);switches=0;games=0
    for _,path in sorted(files.items()):
        frames=json.loads(gzip.decompress(path.read_bytes()))['steps']
        for side in (0,1):
            legacy=m.BeliefTracker(m._Bel,side)
            fog=m._V5Map(frames[0][0]['observation']['players'])
            for step,(before,after) in enumerate(zip(frames,frames[1:])):
                shared=before[0]['observation']
                pl=shared['players']
                obs={k:shared[k] for k in ('step','players','trees')}
                obs['carbon']=before[side]['observation']['carbon']
                est,known=legacy.step(pl,step,obs['carbon'])
                proposed=fog.observe(obs,side)
                action=after[side]['action'] or {}
                fog.moves=action
                legacy.set_actions(pl,action)
                # The target is accessed only after both predictions have been made.
                true=np.array(shared['full_carbon'])
                unchecked=~np.asarray(known,dtype=bool)
                for i,pred in enumerate((est,proposed)):
                    ae=np.abs(true-np.asarray(pred))
                    errors[step//75,i]+=[ae.sum(), ae[unchecked].sum(), unchecked.sum()]
            games+=1
    assert len(files)==len(targets)
    def list_mae(i):
        turns=np.array([75,75,75,74])*225*games
        return dict(all_cells=(errors[:,i,0]/turns).round(3).tolist(),
                    legacy_unknown_cells=(errors[:,i,1]/errors[:,i,2]).round(3).tolist())
    report=dict(maps=len(files),seats=games,phases=['0-74','75-149','150-224','225-298'],v4=list_mae(0),posterior=list_mae(1))
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    main()
