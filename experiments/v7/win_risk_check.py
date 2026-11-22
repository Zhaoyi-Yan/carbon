"""Check the stage-dependent collision choice on paired fixed-command economic counterfactuals."""
from pathlib import Path
import argparse,gzip,hashlib,json,os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import win_risk
from postcargo_check import fixed,run,select,ROOT
from postcargo_report import table

def extra(raw,corrected,w,options,t):
    d={}
    for label,values in [('risk_distance',raw),('risk_cargo',corrected)]:
        result=[]
        for vv,(_,penalty) in zip(values,options):
            vals=np.asarray(vv);left=.25;low=0.
            for i in np.argsort(vals):
                part=min(left,w[i]);low+=part*vals[i];left-=part
                if left<=1e-9:break
            result.append(win_risk._v7_risk_score(vals,w,low/.25,penalty,t))
        d[label]=int(np.argmax(result))
    return d


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('parent',type=Path);ap.add_argument('public',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    params=json.loads(Path(win_risk.__file__).with_name('win-risk-model.json').read_text())
    win_risk._V7_RISK_TURNS=params['turns'];win_risk._V7_RISK_SCALE=params['scales']
    idx={r['match']:r for r in json.loads((a.public/'index.json').read_text())}
    design=json.loads((ROOT/'analysis/v7/postcargo-validation.json').read_text())['maps'];a.out.mkdir(exist_ok=True,parents=True)
    m,orig=fixed(a.parent);out=[]
    for d in design:
        raw=json.load(gzip.open(a.public/idx[d['match']]['file']));path=a.out/(d['match']+'.json')
        if path.exists():games=json.loads(path.read_text())
        else:
            turns=[t for t in select(raw['steps'],d['side'],d['map'],34) if t>=140]
            rows,e=run(m,orig,raw,d['side'],turns,False,extra) if turns else ([],0.)
            games=dict(match=d['match'],map=d['map'],fold=d['fold'],side=d['side'],data=rows)
            path.write_text(json.dumps(games,separators=(',',':')));print(d['match'],len(rows),flush=True)
        out.append(games)
    metrics_all={}
    for fold in ('dev','confirm'):
        select_maps=[r for r in out if r['fold']==fold]
        metrics_all[fold]={}
        for base,alt in [('legacy','risk_distance'),('corrected','risk_cargo')]:
            metrics_all[fold][alt]=table(select_maps,alt,base)
    (ROOT/'analysis/v7/win-risk-branches.json').write_text(json.dumps(dict(design='At or after turn 140, fitted spread of terminal outcomes changes the utility of variable replies. Four candidate/two-ply reference search with calibrated scores. The real first rival command and the next nine pairs of recorded commands score the behavior change.',results=metrics_all),indent=2)+'\n')
if __name__=='__main__':main()
