"""Fit on complete independent layouts, compare actual seed against predeclared placebo tapes and current reply logits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import quantile as qt
from distill import _ds_logprob
from distill_id import _ds_identify
import distill_id
from receipt_data import LAND
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def read(data,model,row):
    folder='distill-negative' if row.get('extra') else 'distill-cache'
    with np.load(data/folder/row['file']) as f:d={k:f[k] for k in f.files}
    with np.load(data/'quantile-cache'/row['tape']) as f:d.update({k:f[k] for k in f.files})
    x=d['x'];z=d['z'];role=d['role'];d['expert']=x[:,:,2].copy();x[:,:,[1,2,25]]=0.
    logits=[]
    for p in ('positive_',''):
        logits.append(np.asarray(model[p+'scale'])[role,None]*z+np.einsum('nak,nk->na',x,np.asarray(model[p+'coefficients'])[role]))
    odds=_ds_logprob(logits[0])-_ds_logprob(logits[1]);mem={};w=np.zeros(len(z))
    d['count']=np.zeros(len(z));d['evidence']=np.zeros(len(z))
    for t in np.unique(d['t']):
        ix=np.flatnonzero(d['t']==t)
        units={int(d['uid'][i]):[int(d['pos'][i])] for i in ix}
        w[ix]=_ds_identify(mem,int(t),units)
        d['count'][ix]=mem['count'];d['evidence'][ix]=mem['evidence']
        mem['pending']={int(d['uid'][i]):(int(d['pos'][i]),odds[i]) for i in ix}
    proposal=logits[0]*np.sqrt(1./(.4*np.asarray(model['positive_scale'])[role,None]))
    d['base']=_ds_logprob((1-w[:,None])*z/.4+w[:,None]*proposal)
    d['bare']=z
    del d['x']
    return d

def opt(train,kind):
    y=np.concatenate([d['y'] for d in train]);base=np.concatenate([d['base'] for d in train])
    raw=np.concatenate([d[kind] for d in train]);u=np.concatenate([d['u'][:,0] for d in train]);idx=np.arange(len(y))
    best=(0.,1.,.15,0.)
    for t in (.30,.40,.60,.90,1.25,1.65):
        sample=qt._qt_soft(raw/t)
        for h in (.065,.14,.25,.38):
            for m in (.06,.16,.32,.56,.76):
                score=_ds_logprob(base+qt._qt_logratio(sample,u,h,m))
                nll=(base[idx,y]-score[idx,y]).mean()
                if nll<best[0]:best=(float(nll),t,h,m)
    return best

def predictions(d,kind,fit,k):
    _,t,h,m=fit
    q=d['base'];sample=qt._qt_soft(d[kind]/t)
    logs=_ds_logprob(q+qt._qt_logratio(sample,d['u'][:,k],h,m))
    evidence=logs-q;mem={};weight=np.zeros(len(q))
    for turn in np.unique(d['t']):
        ix=np.flatnonzero(d['t']==turn);units={int(d['uid'][i]):[int(d['pos'][i])] for i in ix}
        weight[ix]=qt._qt_gate(mem,int(turn),units)
        mem['pending']={int(d['uid'][i]):(int(d['pos'][i]),evidence[i]) for i in ix}
    return logs,_ds_logprob(q+evidence*weight[:,None]),weight

def summary(items):
    bymap=defaultdict(list)
    for key,v in items:bymap[key].append(v)
    rows=np.concatenate([v for _,v in items]);maps=np.array([np.concatenate(v).mean(0) for v in bymap.values()])
    ci=np.random.RandomState(8107).choice(maps[:,0],(4000,len(maps))).mean(1)
    fields='delta_nll nll top1 mass alternate_hit gate'.split()
    return dict(zip(fields,rows.mean(0).round(6).tolist()),n=len(rows),maps=len(maps),
                map_delta=round(float(maps[:,0].mean()),6),gain_maps=int(np.sum(maps[:,0]<0)),
                map_delta95=np.quantile(ci,[.025,.975]).round(6).tolist())
def main(data):
    ix=json.loads((data/'quantile-cache/index.json').read_text())
    distill_id._ROLLOUT_LANDING=qt._ROLLOUT_LANDING=list(map(list,LAND))
    mod=json.loads((HERE/'distill-fast-model.json').read_text())
    games=[(r,read(data,mod,r)) for r in ix['records']]
    train=[(r,d) for r,d in games if r['fold']=='train' and not r['negative']]
    test=[(r,d) for r,d in games if r['fold']=='test']
    tr={r['group'] for r,d in train};te={r['group'] for r,d in test}
    assert tr.isdisjoint(te) and (tr|te).isdisjoint(ix['excluded'])
    fits={kind:opt([d for r,d in train],kind) for kind in ('bare','expert')}
    _,t,h,m=fits['bare']
    (HERE/'quantile-model.json').write_text(json.dumps(dict(sample='bare',temperature=t,width=h,mixture=m,training_layouts=sorted(tr)),indent=2)+'\n')
    results=defaultdict(list)
    for r,d in test:
        z=d['base'];idx=np.arange(len(z));y=d['y'];g=r['group'];zero=np.zeros(len(y))
        ps={'baseline':(z,zero)}
        for kind,fit in fits.items():
            for k,seed in enumerate(ix['seeds']):
                direct,gated,w=predictions(d,kind,fit,k)
                ps[f'{kind}-{seed}-known']=(direct,zero+1)
                ps[f'{kind}-{seed}-gated']=(gated,w)
        collections=['others' if r['negative'] else 'v6',r['source']+'-'+('other' if r['negative'] else 'v6')]
        top=z.argmax(1);high=(d['cargo']>=65)&(d['role']==0)
        for key,(score,w) in ps.items():
            nll=-score[idx,y];rk=score.copy();rk[idx,top]=-np.inf
            metrics=np.stack((nll+z[idx,y],nll,score.argmax(1)==y,np.exp(-nll),rk.argmax(1)==y,w),1)
            for group,mask in [('all',idx>=0),('loaded',high),('loaded_non_top',high&(y!=top)),('loaded_rare',high&(np.exp(z[idx,y])<.15))]:
                if mask.any():
                    for co in collections:results[co,group,key].append((g,metrics[mask]))
    output={}
    for (co,group,k),items in results.items():output.setdefault(co,{}).setdefault(group,{})[k]=summary(items)
    report=dict(train_layouts=sorted(tr),test_layouts=sorted(te),excluded=ix['excluded'],windows=ix['windows'],
      training_games=len(train),evaluation_seat_games=len(test),fit=fits,action='Actual post-search worker command, including subsequently killed workers.',
      baseline='Frozen Fairlead opponent worker logits with prospective fast distillation. Evaluation on cached contiguous windows. Confidence resets at each window; the RNG tape, separately, uses every turn from turn 0.',
      controls='Three independently seeded placebo tapes with identical agent counts, model, temperature, bandwidth, and gate. Raw bare predictions use 0.4 planning for baseline; elite proxy is only a diagnostic and costs additional network calls.',
      groups='Loaded collectors carry at least 65; loaded_non_top excludes the baseline top action based on the eventual label, descriptive only. alternate_hit ranks other four commands after removing the original baseline top.',
      gate='Minimum 10 previously seen surviving-worker motions, recency .965, posterior likelihood versus the untouched baseline. It recognizes action evidence and does not identify a version.',
      models=output)
    (ROOT/'analysis/v7/quantile-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(fits)
    for c in ('v6','local-v6','public-v6','local-other','public-other'):
        print(c,[(k,output[c]['all'][k]['delta_nll']) for k in ('bare-37-known','bare-37-gated','bare-36-known','expert-37-known')],output[c]['loaded_non_top']['bare-37-known']['delta_nll'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
