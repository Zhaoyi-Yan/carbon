"""Fit 129 parameters on self-play returns, with separate calibration and held-out maps."""
import selfplay_value_data as data
import argparse,base64,hashlib,io,json,re
from collections import Counter
from pathlib import Path
import numpy as np

MODEL=data.HERE/'selfplay_value_model.npz'
def read(cache,rs):
    groups=[];cols={k:[] for k in ('h','old','y','win','s','t','horizons','sc')}
    for row in rs:
        d=dict(np.load(cache/row['file']));groups.extend([row['layout']]*len(d['old']))
        for k in cols:cols[k].append(d[k])
    return dict({k:np.concatenate(v) for k,v in cols.items()},group=np.array(groups))

def fit(d,reg):
    h=d['h'].astype('d');target=(d['y']-d['old']).astype('d');groups=d['group'];count=Counter(groups)
    wt=np.array([1/count[g] for g in groups]);wt/=wt.sum()
    mu=wt@h;sd=np.sqrt(wt@((h-mu)**2)+.015**2);x=(h-mu)/sd;inter=wt@target
    v=np.linalg.solve(x.T@(wt[:,None]*x)+reg*np.eye(x.shape[1]),x.T@(wt*(target-inter)))/sd
    return v,inter-mu@v

def measure(d,p):
    actual=3000*d['y'];pred=3000*p;err=np.abs(pred-actual);groups=sorted(set(d['group']))
    per=[dict(layout=g,mae=round(float(err[d['group']==g].mean()),3),n=int(np.sum(d['group']==g))) for g in groups]
    return dict(n=len(err),maps=len(per),mae=float(err.mean()),rmse=float(np.sqrt(np.mean((pred-actual)**2))),
                equal_map_mae=float(np.mean([r['mae'] for r in per])),end_winner_accuracy=float(np.mean((p>0)==d['win'])),
                horizon_mae=np.abs(pred[:,None]-d['horizons']).mean(0).round(2).tolist(),per_map=per)

def build(source,dw,db):
    text=source.read_text();pattern=r'^_VALUE = "([^"]+)"$';match=re.search(pattern,text,re.M);b=match[1]
    old=dict(np.load(io.BytesIO(base64.b64decode(b)),allow_pickle=False));fresh={k:v.copy() for k,v in old.items()}
    fresh['v3.weight']+=dw.astype(fresh['v3.weight'].dtype)[None,:]
    fresh['v3.bias']+=np.asarray(db,dtype=fresh['v3.bias'].dtype)
    out=io.BytesIO();np.savez_compressed(out,**fresh);new=base64.b64encode(out.getvalue()).decode()
    assert all(np.array_equal(old[k],fresh[k]) for k in old if not k.startswith('v3.'))
    result=text[:match.start(1)]+new+text[match.end(1):]
    assert re.sub(pattern,'VALUE_FIELD',result,flags=re.M)==re.sub(pattern,'VALUE_FIELD',text,flags=re.M)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path);p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--export-only',action='store_true',help='Rebuild the candidate from the published frozen parent and 129-parameter delta')
    a=p.parse_args()
    if a.export_only:
        if data.sha(data.SOURCE)!=data.FROZEN:raise ValueError('unexpected source')
        archived=json.loads((data.OUT/'selfplay-value-factual.json').read_text())
        if data.sha(MODEL)!=archived['model_sha256']:raise ValueError('unexpected residual weights')
        with np.load(MODEL) as fit:
            code=build(data.SOURCE,fit['dw'],fit['db'])
        if hashlib.sha256(code.encode()).hexdigest()!=archived['candidate_sha256']:raise ValueError('source regeneration mismatch')
        a.candidate.write_text(code);print(archived['candidate_sha256']);return
    if a.cache is None:p.error('supply --cache for fitting or --export-only for a rebuild')
    meta=json.loads((a.cache/'index.json').read_text());rows=meta['maps']
    tr=read(a.cache,[r for r in rows if r['batch']<=5]);dev=read(a.cache,[r for r in rows if r['batch']==6])
    grid=[(reg,shrink) for reg in (.02,.1,.5,2.,8.) for shrink in (.35,.65,1.)]
    candidates=[]
    for reg,shrink in grid:
        w,b=fit(tr,reg);pr=dev['old']+shrink*(dev['h']@w+b)
        candidates.append((measure(dev,pr)['equal_map_mae'],reg,shrink))
    best,reg,blend=min(candidates)
    group=read(a.cache,[r for r in rows if r['batch']<=6]);valid=read(a.cache,[r for r in rows if r['batch']>=7])
    w,b=fit(group,reg);w=(blend*w).astype('f');b=np.float32(blend*b)
    np.savez_compressed(MODEL,dw=w,db=b)
    report={'base_sha256':data.FROZEN,'model_sha256':data.sha(MODEL),'changed_parameters':len(w)+1,
            'label':'0.28 times the 20-turn cash margin plus 0.52 times the 52-turn margin plus 0.20 times the 100-turn margin, using observed final cash past the referee terminal frame; both legal observation perspectives',
            'schedule':rows,'excluded':meta['excluded'],'official_logs_sha256':meta['files'],
            'fit':'Map-balanced ridge on the frozen 128-dimensional last-layer activations; eigen directions standardized relative to their weighted train-only variability. Only v3.weight and v3.bias change.',
            'dev_grid':candidates,'selected':{'ridge':reg,'shrink':blend,'old_mae':measure(dev,dev['old'])['equal_map_mae'],'new_mae':best},
            'train_n':len(group['y']),'inner_train_n':len(tr['y']),'inner_dev_n':len(dev['y'])}
    for title,dd in [('holdout',valid),('seat0', {k:v[valid['s']==0] for k,v in valid.items()}),('seat1',{k:v[valid['s']==1] for k,v in valid.items()})]:
        new=dd['old']+dd['h']@w+b;report[title]={'base':measure(dd,dd['old']),'new':measure(dd,new)}
    old,res=report['holdout']['base'],report['holdout']['new']
    delta=np.asarray([n['mae']-o['mae'] for o,n in zip(old['per_map'],res['per_map'])]);rng=np.random.default_rng(69011)
    report['holdout']['map_bootstrap_p025_p975']=np.percentile(delta[rng.integers(0,len(delta),(4000,len(delta)))].mean(1),[2.5,97.5]).tolist()
    report['holdout']['maps_better']=int(sum(delta<0))
    qualified=res['mae']<old['mae'] and res['equal_map_mae']<old['equal_map_mae']
    report['fixed_development_trial_eligible']=qualified
    if qualified:
        code=build(data.SOURCE,w,b);compile(code,str(a.candidate),'exec');a.candidate.write_text(code)
        report['candidate_sha256']=data.sha(a.candidate);report['candidate_bytes']=a.candidate.stat().st_size
    (data.OUT/'selfplay-value-factual.json').write_text(json.dumps(report,indent=1)+'\n')
    print('held',old['mae'],res['mae'],'balanced',old['equal_map_mae'],res['equal_map_mae'],'maps improved',sum(delta<0),'eligibility',qualified,flush=True)

if __name__=='__main__':main()
