"""Recheck action accuracy on games absent from the v4 replay training pool."""
import argparse
import json
from pathlib import Path
import numpy as np
from backbone import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('cache','split','weights','legacy_index','output'):p.add_argument(arg,type=Path)
    args=p.parse_args()
    held=set(json.loads(args.split.read_text())['validation_matches'])
    legacy={Path(g['file']).name.split('.')[0] for g in json.loads(args.legacy_index.read_text())}
    heads=dict(np.load(args.weights));model=load()
    prior=model._POLICY.p
    ow=prior['hw.w'].reshape(5,-1);ob=prior['hb.w'].reshape(3,-1)
    totals=np.zeros((2,3))
    for name in sorted(held-legacy):
        with np.load(args.cache/(name+'.npz')) as z:
            for kind in range(3):
                mask=z['K']==kind
                h=z['H'][mask].astype(np.float32);y=z['A'][mask];t=z['T'][mask]
                w,b=(ow,prior['hw.b']) if kind!=2 else (ob,prior['hb.b'])
                logits=h@w.T+b
                delta=np.einsum('mac,mc->ma',heads['coef'][t,kind,:len(b)],h)+heads['bias'][t,kind,:len(b)]
                for i,pred in enumerate((logits,logits+delta)):
                    centered=pred-pred.max(1,keepdims=True)
                    ce=-centered[np.arange(len(y)),y]+np.log(np.exp(centered).sum(1))
                    totals[i]+=[len(y),int((pred.argmax(1)==y).sum()),ce.sum()]
    report=dict(held_out_maps=len(held),legacy_overlap=len(held&legacy),independent_maps=sorted(held-legacy),
                unit_decisions=int(totals[0,0]),v4=dict(accuracy=totals[0,1]/totals[0,0],nll=totals[0,2]/totals[0,0]),
                teachers=dict(accuracy=totals[1,1]/totals[1,0],nll=totals[1,2]/totals[1,0]))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='independent_maps'})


if __name__=='__main__':
    main()
