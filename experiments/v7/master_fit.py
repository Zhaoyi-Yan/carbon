"""Fit rotation-pooled low-cost shared heads and evaluate commands on unseen whole layouts."""
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from master_data import NAMES
from distill import _ds_logprob

def fit(x,style,y,physical,style_width):
    # Action-independent means cancel in a softmax, including most hidden noise.
    x=x.astype(np.float32);x-=x.mean(1,keepdims=True)
    scale=np.maximum(.15,np.sqrt(np.mean(x*x,axis=(0,1))))
    x/=scale
    d=x.shape[-1];small=min(style_width,d);style_rows=[np.where(style==i)[0] for i in range(3)]
    penalty=np.full(d,.095);penalty[:physical]=.012
    def obj(vector):
        shared=vector[:d];special=vector[d:].reshape(3,small)
        sc=x@shared.astype(np.float32)
        for i,ii in enumerate(style_rows):sc[ii]+=x[ii,:,:small]@special[i].astype(np.float32)
        lp=_ds_logprob(sc);p=np.exp(lp);n=len(y);loss=-lp[np.arange(n),y].mean()
        p[np.arange(n),y]-=1.;p/=n
        grad=np.einsum('na,nak->k',p,x).astype(float)+shared*penalty
        grad2=np.stack([np.einsum('na,nak->k',p[ii],x[ii,:,:small]) for ii in style_rows])+.055*special
        return float(loss+.5*(shared*shared@penalty)+.0275*(special*special).sum()),np.r_[grad,grad2.ravel()]
    opt=minimize(obj,np.zeros(d+3*small),jac=True,method='L-BFGS-B',options={'maxiter':90,'ftol':2.e-8})
    return opt.x[:d]/scale,opt.x[d:].reshape(3,small)/scale[:small]

def predict(w,x,s):
    shared,specific=w
    return x@shared.astype(np.float32)+np.einsum('nak,nk->na',x[:,:,:specific.shape[1]],specific[s].astype(np.float32))
def temp(x,y):
    def fn(z):
        lp=_ds_logprob(x*z[0]);probs=np.exp(lp);probs[np.arange(len(y)),y]-=1.
        return float(-lp[np.arange(len(y)),y].mean()+.0005*z[0]**2),np.array([(probs*x).sum()/len(y)+.001*z[0]])
    return minimize(fn,np.ones(1),jac=True,method='L-BFGS-B',bounds=[(.025,5.)]).x[0]
def reads(cache,row):
    with np.load(cache/row['cache']) as n:return dict(n)
def summary(sets):
    rows=np.concatenate([v for k,v in sets]);groups=defaultdict(list)
    for k,v in sets:groups[k].append(v)
    maps=np.array([np.concatenate(v).mean(0) for v in groups.values()])
    delta=maps[:,0]-maps[:,2]
    ci=np.quantile(np.random.RandomState(139).choice(delta,(4000,len(maps))).mean(1),[.025,.975]).round(4)
    mean=rows.mean(0)
    return dict(maps=len(groups),units=len(rows),model_nll=round(mean[0],5),model_hit1=round(mean[1],5),
                strong_mix_nll=round(mean[2],5),strong_mix_hit1=round(mean[3],5),frozen_root_nll=round(mean[4],5),root_hit1=round(mean[5],5),
                prior_style_nll=round(mean[6],5),prior_style_hit1=round(mean[7],5),macro_delta=round(delta.mean(),5),
                maps_beat_strong=int((delta<0).sum()),ci95=ci.tolist())
def main(a):
    rows=json.loads((a.cache/'index.json').read_text());train=[v for v in rows if v['fold']=='train'];test=[v for v in rows if v['fold']=='test']
    A={r['group'] for r in train};B={r['group'] for r in test};assert A.isdisjoint(B)
    trainset=[(r,reads(a.cache,r)) for r in train];testset=[(r,reads(a.cache,r)) for r in test]
    allweights={};scores=defaultdict(list);tempmodel={}
    rawrow=lambda v,role:(np.arange(len(v['by'])) if role==2 else np.nonzero(v['wr']==role)[0])
    for role in (0,1,2):
        prefix='b' if role==2 else 'w'
        def get(rowspec):
            sets=[]
            for row,v in rowspec:
                ii=rawrow(v,role);n=len(ii);s=np.full(n,NAMES.index(row['teacher']),int)
                sets.append((v[prefix+'x'][ii].astype(np.float32),v[prefix+'z'][ii].transpose(0,2,1),v[prefix+'y'][ii],s))
            return tuple(np.concatenate([zz[j] for zz in sets]) for j in range(4))
        x,z,y,st=get(trainset)
        weights=fit(x,st,y,25 if role<2 else 32,12)
        # Stronger comparison can invoke two extra deep styles and combine them with all three incumbent heads.
        oracle=fit(z,st,y,5,5)
        scales=np.array([[temp(z[st==s,:,j],y[st==s]) for j in range(z.shape[2])] for s in range(3)])
        tempmodel[str(role)]=scales.round(4).tolist()
        for n,ws in zip(('w','s'),weights):allweights[n+str(role)]=np.asarray(ws,dtype=np.float32)
        for r,v in testset:
            x,z,y,s=get([(r,v)]);ii=np.arange(len(y))
            if not len(ii):continue
            sid=int(s[0]);prior_col=3+sid if sid<2 else 3
            model=predict(weights,x,s);strong=predict(oracle,z,s)
            root=z[:,:,0]*scales[sid,0]
            incumbent=z[:,:,prior_col]*scales[sid,prior_col]
            matrices=[]
            for logits in (model,strong,root,incumbent):
                matrices.extend((-_ds_logprob(logits)[ii,y],logits.argmax(1)==y))
            tab=np.stack(matrices,1);sel=rawrow(v,role);t=v[prefix+'t'][sel];poor=v[prefix+'poor'][sel]
            for label,mask in (('all',t>=0),('cash_open',(t<=75)&poor),('newborn',t<=12),('later',t>75)):
                if np.any(mask):
                    scores[role,label,'all'].append((r['group'],tab[mask]));scores[role,label,r['teacher']].append((r['group'],tab[mask]))
            if role==0:
                blocked=(t>75)&v['blocked'][sel]
                if np.any(blocked):
                    scores[role,'blocked','all'].append((r['group'],tab[blocked]));scores[role,'blocked',r['teacher']].append((r['group'],tab[blocked]))
    with a.model.open('wb') as f:np.savez_compressed(f,**allweights)
    statistics={}
    for (k,scope,teacher),data in scores.items():statistics.setdefault(('collector','planter','base')[k],{}).setdefault(scope,{})[teacher]=summary(data)
    economy={}
    for teacher in NAMES:
        rs=[r['economy'] for r in rows if r['teacher']==teacher]
        economy[teacher]=dict(games=len(rs),opening=dict(Counter('/'.join(e['opening']) for e in rs)),
            median_first_deposit=float(np.median([e['first_deposit'] for e in rs if e['first_deposit'] is not None])),
            median_first_tree=float(np.median([e['first_tree'] for e in rs if e['first_tree'] is not None])),
            checkpoints={t:np.array([e['steps'][t] for e in rs if t in e['steps']]).mean(0).round(2).tolist() for t in ('10','20','35','50','80')})
    report=dict(training_games=len(train),validation_games=len(test),training_maps=sorted(A),validation_maps=sorted(B),
                cohorts={t:{f:sum(r['teacher']==t and r['fold']==f for r in rows) for f in ('train','test')} for t in NAMES},
                match_ids=[{k:r[k] for k in ('match','game','seed','group','teacher','side','fold')} for r in rows],
                economy=economy,economic_units='Cumulative gross transfers, current cash, collector count, tree count. Descriptive across all maps.',
                measurement='Teacher-facing fog updated on every turn. Existing root router evaluated on sampled frames. Base commands only if affordable with free worker capacity. An incumbent style may have seen some historical maps in older pretraining; new heads have never trained on any validation layout.',
                comparisons='Frozen root and matched dedicated deep style (D=1, B2=2, E proxy=1) independently temperature calibrated on training. Strong mix is independently fitted to root, two existing offsets, both dedicated deep styles. Learned model only needs normal root activations.',
                model_bytes=a.model.stat().st_size,statistics=statistics)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(report['cohorts'],economy)
    for k in ('collector','planter','base'):print(k,{s:statistics[k][s]['all'] for s in statistics[k]})
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('model',type=Path);p.add_argument('report',type=Path)
    main(p.parse_args())
