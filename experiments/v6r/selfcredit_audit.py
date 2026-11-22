"""Control for every public landing option when testing the incremental signal of the recorded move."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from selfcredit_fit import fit,model_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('report',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();rep=json.loads(a.report.read_text())
    with np.load(a.data) as z:d=dict(z)
    eligible=d['seat']==d['candidate'];held=np.isin(d['seed'],[r['seed'] for r in rep['maps'] if r['split']=='test'])
    tr=~held & eligible
    x=d['features'].astype(float);n=len(x);chosen=x[np.arange(n),d['action'],12:]
    # This baseline sees all five candidate landing cells, threats and return distances,
    # in their original direction order. Only the actual chosen index is withheld.
    board=np.column_stack((x[:,0,:12],x[:,:,12:].reshape(n,-1)))
    with_action=np.column_stack((board,chosen))
    results={}
    for role,X in [('board_with_all_five_possible_actions',board),('board_and_recorded_choice',with_action)]:
        w,b=fit(X[tr],d['paid'][tr]);q=expit(X@w+b)
        results[role]={k:model_metrics(q[m],d['paid'][m]) for k,m in [('candidate',held&eligible),('v5',held&~eligible)]+[(str(s),d['seed']==s) for s in np.unique(d['seed'][held])]}
    with a.output.open('x') as f:json.dump(results,f,indent=2)
    for k in ('v5','candidate'):
        print(k,*[(w,r[k]['nll'],r[k]['auc']) for w,r in results.items()])


if __name__=='__main__':main()
