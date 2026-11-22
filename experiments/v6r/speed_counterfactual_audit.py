"""Compare the student's counterfactual moves with exact teacher predictions on the same simulated boards."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from speed_audit import load, probe


def js_and_match(reference, predictions, temperature):
    def normalize(logits):
        z=logits.astype(np.float64)/temperature
        z-=z.max(axis=1,keepdims=True)
        p=np.exp(z)
        return p/p.sum(axis=1,keepdims=True)
    p,q=normalize(reference),normalize(predictions)
    m=.5*(p+q)
    js=.5*((p*np.log(p/m)).sum(axis=1)+(q*np.log(q/m)).sum(axis=1))
    return np.array([len(p),js.sum(),(p.argmax(axis=1)==q.argmax(axis=1)).sum()])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path)
    p.add_argument('replay',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--turns',default='40,100,200')
    a=p.parse_args()
    model=load(a.candidate,'anchored_agreement',True)
    anchored=model._v6r_counterfactuals
    stats={}
    def checked(net,X,S,side,part,ours=None):
        result=anchored(net,X,S,side,part,ours)
        aw,ab,_=result
        if ours is not None:
            aw,ab=aw[1::2],ab[1::2]
            X=X[1::2]
        n=len(X)
        roles=X.reshape(n,model.NPL,225)
        exact_w,exact_b,_=net(np.ascontiguousarray(X),model._v5_ids([side]*n),play_sides=[side]*n)
        small_w,small_b,_=model._V6R_TINY(np.ascontiguousarray(X),model._v5_ids([side]*n))
        seen=set()
        distinct=[]
        for i,b in enumerate(part):
            if b[1] in seen:
                distinct.append(i)
            seen.add(b[1])
        for role,teacher,student,pred in ((2,exact_w,small_w,aw),(4,exact_w,small_w,aw),(12,exact_b,small_b,ab)):
            rr,cc=np.nonzero(roles[distinct,role]>.5)
            if not len(rr):
                continue
            row=np.asarray(distinct)[rr]
            truth=teacher[row,cc]
            for label,values in (('student',student),('anchored',pred)):
                key=f'{role}:{label}'
                stats.setdefault(key,np.zeros(3))
                stats[key]+=js_and_match(truth,values[row,cc],model._TEMP)
        return result
    model._v6r_counterfactuals=checked
    with gzip.open(a.replay) as f:
        replay=json.load(f)
    with a.output.open('x') as out:
        for i,t in enumerate(map(int,a.turns.split(','))):
            stats.clear()
            probe(model,replay,t,i%2)
            out.write(json.dumps({'turn':t,'seat':i%2,'seed':replay['configuration']['randomSeed'],
                                  'columns':['compared','sum_js','argmax_agreements'],
                                  'stats':{k:v.tolist() for k,v in stats.items()}})+'\n')
            out.flush()


if __name__=='__main__':
    main()
