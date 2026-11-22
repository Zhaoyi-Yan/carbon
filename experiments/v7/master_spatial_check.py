"""Evaluate the isolated action expert on the original 13 held-out Master maps and old-head timestamps."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import log_softmax
import torch
from dual_weights import arrays,encoded
import master_fit as old_base
import master_worker as mw
from master_spatial_fit import Spatial,contents,HERE,ROOT,PARENT,NAMES
from distill_data import load

def load_net(path):
    data=dict(np.load(path));kind=Spatial
    if 'query.weight' in data:
        from master_spatial_goal import GoalNet
        kind=GoalNet
    net=kind(data['mean'],data['inv'],arrays(encoded(PARENT)))
    dst=net.state_dict();dst.update({k:torch.from_numpy(q) for k,q in data.items()})
    net.load_state_dict(dst);net.eval();return net
def scored(z,y):
    y=np.asarray(y)
    if not len(y):return []
    lp=log_softmax(z,1);p=z.argmax(1)
    return np.column_stack((-lp[np.arange(len(y)),y],p==y,p!=0,y!=0,(p!=0)&(y!=0))).tolist()
def metric(v):
    d=np.array(v);assert d.ndim==2
    nll,acc,want,truth,ok=d.sum(0)
    return dict(n=len(d),nll=round(float(nll/len(d)),5),hit=round(float(acc/len(d)),5),
                nonidle_precision=round(float(ok/max(1,want)),4),nonidle_recall=round(float(ok/max(1,truth)),4))
def main(data,model):
    torch.set_num_threads(1);net=load_net(model);m=load(PARENT);mw._V5_ROTATIONS=m._V5_ROTATIONS
    workers=dict(np.load(HERE/'master-worker-model.npz'));hiring=dict(np.load(HERE/'master-model.npz'))
    rows=json.loads((data/'master-worker/index.json').read_text());train={r['group'] for r in rows if r['fold']=='train'}
    hold=[r for r in rows if r['fold']=='test'];assert len(train)==42 and len({r['group'] for r in hold})==13 and not (train&{r['group'] for r in hold})
    terms=defaultdict(list);per=[]
    for r in hold:
        sid=NAMES.index(r['teacher']);x,y,goal,mask,t,rot=contents(data/'master-spatial'/r['cache'])
        with torch.inference_mode():
            allq=np.concatenate([net(torch.from_numpy(v.astype('f')),torch.full((len(v),),sid,dtype=torch.long)).numpy() for v in np.array_split(x,max(1,len(x)//32))])
        with gzip.open(r['file']) as f:g=json.load(f)
        wnew=[];w1=[];wg=[];wy=[];wtype=[];bnew=[];b1=[];by=[]
        for ts in np.unique(t):
            ids=np.flatnonzero(t==ts);assert len(ids)==4 and np.array_equal(rot[ids],[0,1,2,3])
            p=g['steps'][int(ts)][0]['observation']['players'][r['side']]
            positions=[a[0] for a in p[2].values()]
            sites=list(p[1].values());w=[];b=[]
            for i in ids:
                cells,inv,actions,chs=m._V5_ROTATIONS[int(rot[i])];q=allq[i].T
                w.append(q[cells[positions],:5][:,actions]);b.append(q[cells[sites],5:8])
            if positions:
                wnew.extend(np.mean(w,axis=0));w1.extend(w[0]);wy.extend(y[ids[0],0,positions]);wtype.extend(mask[ids[0],positions])
                # Remove the learned spatial residual to assess the inherited native prior.
            if p[0]>=30 and len(positions)<10:
                bnew.extend(np.mean(b,axis=0));b1.extend(b[0]);by.extend(y[ids[0],1,sites])
        wtype=np.asarray(wtype);wnew=np.asarray(wnew);wy=np.asarray(wy);by=np.asarray(by)
        with np.load(data/'master-worker'/r['cache']) as f:
            base=f['base'].astype('f');assert np.array_equal(wy,f['y'])
            nr=f['r'];styles=np.eye(3,dtype='f')[np.full(len(nr),sid)]
            gate=mw._mw_router(f['gs'].astype('f'),workers)[np.searchsorted(f['tgs'],f['t'])]
            soft=np.stack((1-gate,np.zeros_like(gate),gate),axis=1)
            args=[f[k].astype('f') for k in ('h','l','k')]
            old=mw._mw_logits(*args,nr,base,styles,workers);routed=mw._mw_logits(*args,nr,base,soft,workers)
            assert np.array_equal((wtype&2)>0,f['heavy']) and np.array_equal((wtype&8)>0,f['pressure'])
        with np.load(data/'master-cache'/r['cache']) as f:
            assert np.array_equal(by,f['by'])
            oldb=old_base.predict((hiring['w2'],hiring['s2']),f['bx'].astype('f'),np.full(len(by),sid))
            ba=f['bz'][:,0]
        include=dict(all=np.ones(len(wy),bool),collector=(wtype&1)>0,planter=(wtype&1)==0,
                     heavy_home=(wtype&2)>0,light_enemy_base=(wtype&4)>0,light_pressure=(wtype&8)>0)
        local=defaultdict(dict)
        for scope,ii in include.items():
            if not ii.any():continue
            for k,z in dict(spatial4=wnew,spatial1=np.asarray(w1),directional_known=old,directional_routed=routed,frozen_root=base,
                            blend_root=.7*wnew+.3*base).items():
                vv=scored(z[ii],wy[ii]);local[scope][k]=metric(vv)
                for cohort in (r['teacher'],'D_E' if sid!=1 else 'B2_only','all'):terms[cohort,scope,k].extend(vv)
        for scope,select in [('base_affordable',np.ones(len(by),bool)),('reference_hires',by!=0)]:
            if not select.any():continue
            for k,z in dict(spatial4=np.asarray(bnew),spatial1=np.asarray(b1),base_known=oldb,frozen_root=ba,
                            blend_root=.7*np.asarray(bnew)+.3*ba).items():
                vv=scored(z[select],by[select]);local[scope][k]=metric(vv)
                for cohort in (r['teacher'],'D_E' if sid!=1 else 'B2_only','all'):terms[cohort,scope,k].extend(vv)
        per.append(dict(match=r['match'],group=r['group'],seat=r['side'],seed=r['seed'],teacher=r['teacher'],metrics=dict(local)))
    combined={}
    for (co,scope,k),v in terms.items():combined.setdefault(co,{}).setdefault(scope,{})[k]=metric(v)
    ci={}
    for group in ('Master7sE','Master7sD','D_E'):
        rr=[r for r in per if r['teacher']==group or (group=='D_E' and r['teacher']!='Master7sB2')]
        ci[group]={}
        for scope in ('all','heavy_home','light_enemy_base','light_pressure','base_affordable'):
            vs='base_known' if scope=='base_affordable' else 'directional_known';d=[]
            for r in rr:
                if scope not in r['metrics']:continue
                a,b=(r['metrics'][scope][n] for n in ('spatial4',vs))
                d.append([a['nll']-b['nll'],a['hit']-b['hit']])
            d=np.array(d);rng=np.random.RandomState(9106);boot=d[rng.randint(len(d),size=(6000,len(d)))].mean(1)
            ci[group][scope]=dict(maps=len(d),macro_delta=d.mean(0).round(5).tolist(),interval95=np.quantile(boot,[.025,.975],axis=0).round(5).tolist(),
                                  maps_nll_improved=int((d[:,0]<0).sum()),maps_hit_improved=int((d[:,1]>0).sum()))
    out=dict(parent_sha=hashlib.sha256(PARENT.read_bytes()).hexdigest(),weight_sha=hashlib.sha256(model.read_bytes()).hexdigest(),weight_bytes=model.stat().st_size,
        train_maps=sorted(train),independent_maps=per,combined=combined,paired=ci,
        definitions='All predictors use the identical old sparse timestamps and labels. light_enemy_base = collector cargo <=15 and distance to an enemy depot <=4. heavy_home = collector cargo >=65, distance to own depot <=4. light_pressure includes eligible loaded enemy units within 3. Base samples are the same affordable non-cap-capped decisions as the previous study. Expert IDs given to both learned competitors. Macro delta columns are NLL and hit rate, lower NLL/higher hit is better. E has only three independent maps.')
    f='master-spatial-goal-validation.json' if hasattr(net,'query') else 'master-spatial-validation.json'
    (ROOT/'analysis/v7'/f).write_text(json.dumps(out,indent=2)+'\n')
    for who in ('Master7sE','Master7sD','D_E'):
        for sc in ('all','heavy_home','light_enemy_base','base_affordable'):
            d=combined[who][sc];old='base_known' if sc=='base_affordable' else 'directional_known'
            print(who,sc,d[old]['n'],[(k,d[k]['nll'],d[k]['hit']) for k in (old,'spatial1','spatial4')],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('model',type=Path);a=p.parse_args();main(a.data,a.model)
