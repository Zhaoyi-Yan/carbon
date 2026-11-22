"""Fit role-specific search-action residuals, score the online gate on entire held-out layouts."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
import distill as model

def fit(x,y,offset,penalty):
    x=x.astype('d');offset=offset.astype('d')
    def objective(w):
        z=offset+x@w;q=model._ds_logprob(z)
        val=-q[np.arange(len(y)),y].mean()+.5*penalty*(w@w)
        prob=np.exp(q);prob[np.arange(len(y)),y]-=1
        return val,np.einsum('na,nak->k',prob,x)/len(y)+penalty*w
    res=minimize(objective,np.zeros(x.shape[-1]),method='L-BFGS-B',jac=True,options={'maxiter':160,'ftol':1.e-10})
    if not res.success:raise RuntimeError(res.message)
    return res.x

def load(cache,r):
    with np.load(cache/r['file']) as z:return {k:z[k] for k in z.files}
def predictions(game,w,small,scale,temp):
    roles=game['role'];z=game['z'];x=game['x'];y=game['y']
    ref=scale[roles,None]*z; full=ref+np.einsum('nak,nk->na',x,w[roles])
    slim=ref+np.einsum('nak,nk->na',x[:,:,:3],small[roles])
    evidence=np.clip(model._ds_logprob(full)-model._ds_logprob(ref),-3.,3.)
    tvalues=game['t']; prev=-10
    count=total=0.;gated=ref.copy();alpha=np.zeros(len(z))
    for t in np.unique(tvalues):
        now=np.flatnonzero(tvalues==t)
        if t!=prev+1:count=total=0.;last=[]
        count*=.93;total*=.93
        if len(last):
            v=evidence[last,y[last]];n=min(4,len(v))
            total+=n*np.clip(v,-1.,1.).mean();count+=n
        a=np.clip((total/max(1.,count)-.008)*4.,0,.78) if count>=12 else 0.
        gated[now]=(1.-a)*z[now]/temp+a*full[now]
        alpha[now]=a;last=now[game['alive'][now]];prev=t
    return {'operational':z/temp,'calibrated':ref,'specialists_only':slim,'distilled':full,'online':gated},alpha

def summaries(rows):
    vals=np.concatenate([x for _,x in rows]);maps=defaultdict(list)
    for g,v in rows:maps[g].append(v)
    per=np.stack([np.concatenate(sets).mean(0) for sets in maps.values()])
    return {'units':len(vals),'maps':len(maps),'nll':round(float(vals[:,0].mean()),5),
            'top1':round(float(vals[:,1].mean()),5),'probability_on_recorded_action':round(float(vals[:,2].mean()),5),'nll_macro_map':round(float(per[:,0].mean()),5)}
def main(a):
    manifest=json.loads((a.cache/'index.json').read_text());records=manifest['records']
    training=[r for r in records if r['fold']=='train' and not r['negative']]
    tg={r['group'] for r in training};te={r['group'] for r in records if r['fold']=='test'}
    assert tg.isdisjoint(te) and not (tg|te)&set(manifest['excluded_layouts'])
    ds=[load(a.cache,r) for r in training]
    cat=lambda key:np.concatenate([s[key] for s in ds])
    x,z,y,roles=map(cat,('x','z','y','role'))
    coefs=[];scales=[];small=[]
    for role in (0,1):
        use=roles==role;zr=z[use]
        scale=max(.05,fit(zr[:,:,None],y[use],np.zeros_like(zr),.0005)[0])
        scales.append(scale)
        coefs.append(fit(x[use],y[use],scale*zr,.014))
        small.append(fit(x[use,:,:3],y[use],scale*zr,.014))
    scale=np.array(scales);w=np.array(coefs);small=np.array(small)
    weights={'scale':scale.round(7).tolist(),'coefficients':w.round(7).tolist(),'trained_units':len(y),'train_layouts':sorted(tg)}
    a.model.write_text(json.dumps(weights,separators=(',',':'))+'\n')
    logs=defaultdict(list);permap=defaultdict(lambda:defaultdict(list));gates=defaultdict(list)
    for r in records:
        if r['fold']!='test':continue
        g=load(a.cache,r);ys=g['y'];pick=np.arange(len(ys));all_,alpha=predictions(g,w,small,scale,.4)
        cohort='other' if r['negative'] else 'v6';source=r['source'];map_=r['group']
        for scope in (('other',source+'-other') if r['negative'] else ('v6',source+'-v6')):
            gates[scope].append(alpha)
            for name,score in all_.items():
                lp=model._ds_logprob(score);observed=-lp[pick,ys]
                accuracy=score.argmax(1)==ys
                for group,use in (('all',np.ones(len(ys),bool)),('contact',g['x'][:,:,19].max(1)>.5),
                                  ('collector',g['role']==0),('planter',g['role']==1)):
                    if np.any(use):logs[scope,group,name].append((map_,np.column_stack((observed[use],accuracy[use],np.exp(-observed[use])))))
                permap[scope,map_][name].append(observed)
    result={}
    for (cohort,sub,model_),data in logs.items():result.setdefault(cohort,{}).setdefault(sub,{})[model_]=summaries(data)
    ci={}
    for cohort in gates:
        ci[cohort]={}
        for left,right in [('online','operational'),('distilled','calibrated'),('distilled','specialists_only')]:
            vals=[np.concatenate(d[left]).mean()-np.concatenate(d[right]).mean() for (k,g),d in permap.items() if k==cohort]
            vals=np.array(vals);rng=np.random.RandomState(3402)
            ci[cohort][left+'_vs_'+right]={'maps':len(vals),'improved':int(sum(vals<0)),'delta':round(float(vals.mean()),5),
                                          'map_bootstrap95':np.quantile(rng.choice(vals,(6000,len(vals))).mean(1),[.025,.975]).round(5).tolist()}
    out={'model':'Predict the actual worker command chosen by a frozen v6 policy including search. Opposing bases retain frozen priors.',
         'windows':manifest['windows'],'fog':'Observer fog and own replay actions on every turn; frozen neural router evaluated in contiguous windows. Rival action inferred only next turn from surviving movements for gate.',
         'sampling':'Operational baseline and deployed residual use frozen rollout temperature .4; a temperature-only model and a three-specialist-logit model also fitted to identical training data.',
         'train_games':len(training),'train_layouts':sorted(tg),'validation_layouts':sorted(te),'excluded_layouts':manifest['excluded_layouts'],
         'sources':[{k:v for k,v in r.items() if k!='file'} for r in records],'training_workers':len(y),'scale':scales,
         'gate':{k:dict(share_active=round(float((np.concatenate(v)>0).mean()),4),mean_weight=round(float(np.concatenate(v).mean()),4)) for k,v in gates.items()},
         'statistics':result,'paired_nll':ci}
    a.report.write_text(json.dumps(out,indent=2)+'\n')
    for k in ('v6','local-v6','public-v6','other'):
        print(k,result[k]['all'],ci[k],out['gate'][k])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('model',type=Path);p.add_argument('report',type=Path)
    main(p.parse_args())
