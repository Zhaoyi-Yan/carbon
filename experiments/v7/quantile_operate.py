"""Evaluate three predeclared gating conditions and the matched-opening historical stress cohort."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import quantile as qt
import distill_id
from quantile_fit import HERE,ROOT,LAND,read,summary,_ds_logprob

def main(data):
    fit=json.loads((HERE/'quantile-model.json').read_text());ds=json.loads((HERE/'distill-fast-model.json').read_text())
    ix=json.loads((data/'quantile-cache/index.json').read_text());op=json.loads((data/'quantile-cache/openings.json').read_text())
    extras=json.loads((data/'quantile-stress/index.json').read_text())
    distill_id._ROLLOUT_LANDING=qt._ROLLOUT_LANDING=list(map(list,LAND))
    pool=[]
    for r in ix['records']:
        if r['fold']!='test':continue
        key=r['source']+'-'+r['id']+('-other' if r['negative'] else '-v6')
        flag=op[key]['signature'];group='other' if r['negative'] else 'v6'
        pool.append((r['group'],[group,r['source']+'-'+group],flag,read(data,ds,r)))
    for ex in extras:
        with np.load(data/'quantile-stress'/ex['file']) as f: d={k:f[k] for k in f.files}
        pool.append((ex['layout'],['delayed_non_v6',ex['name']],True,d))
    sums=defaultdict(list)
    for g,scopes,match,d in pool:
        b=d['base'];u=d['u'][:,0];y=d['y'];i=np.arange(len(y));sample=qt._qt_soft(d['bare']/fit['temperature'])
        score=_ds_logprob(b+qt._qt_logratio(sample,u,fit['width'],fit['mixture']));logratio=score-b
        zero=np.zeros(len(y));confirmed=zero.copy();rng=zero.copy();mem={}
        for t in np.unique(d['t']):
            ix=np.flatnonzero(d['t']==t)
            units={int(d['uid'][j]):[int(d['pos'][j])] for j in ix}
            rng[ix]=qt._qt_gate(mem,int(t),units)
            c=int(ix[0]);confirmed[ix]=qt._qt_weight(match,d['count'][c],d['evidence'][c],mem['evidence'])
            mem['pending']={int(d['uid'][j]):(int(d['pos'][j]),logratio[j]) for j in ix}
        top=b.argmax(1);high=(d['cargo']>=65)&(d['role']==0)
        for k,w in dict(baseline=zero,opening_only=zero+float(match),seed_gate=rng,opening_and_evidence=confirmed).items():
            z=_ds_logprob(b+w[:,None]*logratio);nll=-z[i,y];alt=z.copy();alt[i,top]=-np.inf
            v=np.column_stack([nll+b[i,y],nll,z.argmax(1)==y,np.exp(-nll),alt.argmax(1)==y,w])
            for group,mask in [('all',i>=0),('loaded',high),('loaded_non_top',high&(y!=top))]:
                if mask.any():
                    for co in scopes:sums[co,group,k].append((g,v[mask]))
    out={}
    for (c,g,k),v in sums.items():out.setdefault(c,{}).setdefault(g,{})[k]=summary(v)
    (ROOT/'analysis/v7/quantile-operating.json').write_text(json.dumps(dict(stress=extras,metrics=out,
          method='Delayed opposite mixed recruit opening gated by exact reactive planter assignment. Confirmed additionally uses existing independent Fairlead class evidence and previously observed survivor motions. Controls predeclared before reading elite stress logits.'),indent=2)+'\n')
    for c in ('v6','local-v6','public-v6','other','delayed_non_v6','CarbonEdge_Artemis','CarbonEdge_Prometheus'):
        o=out[c]['all'];print(c,[(k,o[k]['delta_nll'],o[k]['map_delta95'],o[k]['gate']) for k in ('opening_only','opening_and_evidence')])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
